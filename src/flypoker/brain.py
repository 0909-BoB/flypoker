"""Leaky integrate-and-fire simulator over a (real or synthetic) fly connectome
subgraph. Directly adapted from the flappy-fly project's brain.py: same update
rule, same idea of injecting external current into named neuron populations
and reading spike counts back out. Generalized here to an arbitrary set of
named input/output populations instead of one visual-input / one DN-output
group, since poker needs several distinct numeric channels in and one
multi-way decision out.
"""
from __future__ import annotations

import json
import os
from collections import deque

import numpy as np
import scipy.sparse as sp

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
MATRIX_PATH = os.path.join(DATA_DIR, "circuit.npz")
META_PATH = os.path.join(DATA_DIR, "circuit_meta.json")

# flappy-fly's 1/400 was tuned for a sparser subgraph; this project's ~22-avg-degree
# extraction needs much less attenuation to get the output population to fire at
# all without saturating -- picked empirically, see README "Tuning" section.
WEIGHT_GAIN = 1.0 / 3.0
TAU_MS = 10.0
V_REST = 0.0
V_THRESH = 1.0
V_RESET = 0.0
REFRACTORY_MS = 3.0
SPIKE_WINDOW_MS = 50.0


def load_circuit(matrix_path=MATRIX_PATH, meta_path=META_PATH):
    W = sp.load_npz(matrix_path).tocsr().astype(np.float64) * WEIGHT_GAIN
    with open(meta_path) as f:
        meta = json.load(f)
    return W, meta["nodes"]


class LIFNetwork:
    def __init__(self, W=None, nodes=None, matrix_path=MATRIX_PATH, meta_path=META_PATH):
        if W is None or nodes is None:
            W, nodes = load_circuit(matrix_path, meta_path)
        self.W = W
        self.nodes = nodes
        self.n = len(self.nodes)

        # role is either "relay" or an input/output channel name, e.g.
        # "in:equity", "in:pot_odds", "out:action". Grouped by exact role.
        self.idx_by_role: dict[str, np.ndarray] = {}
        for m in self.nodes:
            self.idx_by_role.setdefault(m["role"], []).append(m["index"])
        self.idx_by_role = {k: np.array(v, dtype=np.int64) for k, v in self.idx_by_role.items()}

        self.reset()

    def input_roles(self) -> list[str]:
        return sorted(r for r in self.idx_by_role if r.startswith("in:"))

    def output_roles(self) -> list[str]:
        return sorted(r for r in self.idx_by_role if r.startswith("out:"))

    def reset(self):
        self.V = np.full(self.n, V_REST, dtype=np.float64)
        self.refractory_until = np.zeros(self.n, dtype=np.float64)
        self.I_ext = np.zeros(self.n, dtype=np.float64)
        self.spikes = np.zeros(self.n, dtype=np.float64)
        self.t = 0.0
        self._spike_log = deque()
        self.last_recording: list = []

    def inject(self, role: str, current: float):
        idx = self.idx_by_role.get(role)
        if idx is not None and len(idx):
            self.I_ext[idx] += current

    def step(self, dt_ms: float = 1.0, recording: list | None = None):
        I_syn = self.W @ self.spikes
        dV = (dt_ms / TAU_MS) * (-(self.V - V_REST) + I_syn + self.I_ext)
        self.V += dV
        self.V[self.t < self.refractory_until] = V_RESET

        can_spike = self.t >= self.refractory_until
        spiking = can_spike & (self.V >= V_THRESH)

        self.spikes = spiking.astype(np.float64)
        self.V[spiking] = V_RESET
        self.refractory_until[spiking] = self.t + REFRACTORY_MS

        self.t += dt_ms
        self.I_ext[:] = 0.0

        if spiking.any():
            spiked_idx = np.flatnonzero(spiking)
            for idx in spiked_idx:
                self._spike_log.append((self.t, idx))
            # Unlike _spike_log above (a rolling window for the readout's
            # spike-count features), `recording` -- when the caller wants
            # one, e.g. for the web UI's live circuit view -- keeps every
            # spike of this run untrimmed, so a full decision's activity
            # can be played back rather than only its trailing window.
            if recording is not None:
                for idx in spiked_idx:
                    recording.append((self.t, int(idx)))
        cutoff = self.t - SPIKE_WINDOW_MS
        while self._spike_log and self._spike_log[0][0] < cutoff:
            self._spike_log.popleft()

    def spike_counts(self, indices: np.ndarray, window_ms: float = SPIKE_WINDOW_MS) -> int:
        cutoff = self.t - window_ms
        idx_set = set(indices.tolist()) if len(indices) else set()
        return sum(1 for (t, idx) in self._spike_log if t >= cutoff and idx in idx_set)

    def output_activity(self, window_ms: float = SPIKE_WINDOW_MS) -> dict[str, int]:
        return {role: self.spike_counts(idx, window_ms) for role, idx in self.idx_by_role.items()
                if role.startswith("out:")}

    def role_activity(self, roles: list[str] | None = None, window_ms: float = SPIKE_WINDOW_MS) -> dict[str, int]:
        """Spike counts per named population, for visualization. Defaults to
        every role (inputs, outputs, and the relay pool)."""
        roles = roles if roles is not None else list(self.idx_by_role.keys())
        return {role: self.spike_counts(self.idx_by_role[role], window_ms)
                for role in roles if role in self.idx_by_role}

    def output_spike_vector(self, role: str, window_ms: float = SPIKE_WINDOW_MS) -> np.ndarray:
        """Per-neuron spike counts for one output population, for use as a
        feature vector into a trainable readout (a single aggregate count
        throws away which neurons fired, which is what the decoder needs)."""
        idx = self.idx_by_role.get(role, np.array([], dtype=np.int64))
        cutoff = self.t - window_ms
        counts = np.zeros(len(idx), dtype=np.float64)
        pos = {int(i): k for k, i in enumerate(idx)}
        for t, i in self._spike_log:
            if t >= cutoff and i in pos:
                counts[pos[i]] += 1.0
        return counts

    def output_window_features(self, role: str, n_windows: int, total_ms: float) -> np.ndarray:
        """Per-neuron spike counts of one output population in `n_windows`
        equal time slices of the last run(record=True), flattened. The
        trailing-window vector above only sees the last 50ms; splitting the
        whole run into slices keeps the response's time course too, which
        carries noticeably more about the inputs (linear equity R^2 rose from
        ~0.42 to ~0.64 in testing)."""
        idx = self.idx_by_role.get(role, np.array([], dtype=np.int64))
        pos = {int(i): k for k, i in enumerate(idx)}
        counts = np.zeros((n_windows, len(idx)))
        width = total_ms / n_windows
        for t, i in self.last_recording:
            k = pos.get(i)
            if k is not None:
                counts[min(int(t // width), n_windows - 1), k] += 1.0
        return counts.reshape(-1)

    def run(self, inputs: dict[str, float], n_steps: int = 200, dt_ms: float = 1.0,
            record: bool = False) -> dict[str, int]:
        """Inject each `inputs[role]` current every step for n_steps, then
        return spike counts per output role over the trailing window.
        `record=True` also captures every spike of this run (not just the
        trailing window) into `self.last_recording` as (t_ms, neuron_idx)
        pairs, for playback in the web UI's live circuit view."""
        recording = [] if record else None
        for _ in range(n_steps):
            for role, current in inputs.items():
                self.inject(role, current)
            self.step(dt_ms, recording=recording)
        if record:
            self.last_recording = recording
        return self.output_activity()
