"""Generates a placeholder circuit with the same file schema build_circuit.py
would produce from real MaleCNS data (circuit.npz + circuit_meta.json), so
the rest of the pipeline can be built and tested before a real neuPrint
token is available. This is NOT real connectome data -- it's a random
small-world-ish directed graph with the right role labels wired in. Swap it
out with scripts/build_circuit.py once you have real data.
"""
from __future__ import annotations

import json
import os

import numpy as np
import scipy.sparse as sp

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")

# Sensory channels we invented for poker (no literal biological analogue --
# see README for the honest framing). Each gets its own small input
# population feeding into the shared relay pool.
INPUT_CHANNELS = [
    "in:equity", "in:pot_odds", "in:spr", "in:street",
    "in:position", "in:opp_aggression",
]
OUTPUT_CHANNELS = ["out:action"]

N_INPUT_PER_CHANNEL = 15
N_OUTPUT = 60
N_RELAY = 900
MEAN_OUT_DEGREE = 12
INHIBITORY_FRACTION = 0.3


def build(seed: int = 0, out_dir: str = DATA_DIR):
    rng = np.random.default_rng(seed)

    nodes = []
    idx = 0
    role_ranges: dict[str, tuple[int, int]] = {}
    for ch in INPUT_CHANNELS:
        start = idx
        for _ in range(N_INPUT_PER_CHANNEL):
            nodes.append({"index": idx, "bodyId": None, "type": "synthetic", "role": ch})
            idx += 1
        role_ranges[ch] = (start, idx)

    start = idx
    for _ in range(N_RELAY):
        nodes.append({"index": idx, "bodyId": None, "type": "synthetic", "role": "relay"})
        idx += 1
    role_ranges["relay"] = (start, idx)

    for ch in OUTPUT_CHANNELS:
        start = idx
        for _ in range(N_OUTPUT):
            nodes.append({"index": idx, "bodyId": None, "type": "synthetic", "role": ch})
            idx += 1
        role_ranges[ch] = (start, idx)

    n = idx
    is_inhibitory = rng.random(n) < INHIBITORY_FRACTION

    rows, cols, data = [], [], []

    def connect(src_range, dst_range, mean_degree):
        src_lo, src_hi = src_range
        dst_lo, dst_hi = dst_range
        for s in range(src_lo, src_hi):
            k = rng.poisson(mean_degree)
            k = min(k, dst_hi - dst_lo)
            if k <= 0:
                continue
            targets = rng.choice(np.arange(dst_lo, dst_hi), size=k, replace=False)
            weight = rng.integers(1, 20, size=k)
            sign = -1 if is_inhibitory[s] else 1
            for t, w in zip(targets, weight):
                rows.append(t)  # W[post, pre] convention, matches brain.py's W @ spikes
                cols.append(s)
                data.append(int(w) * sign)

    relay_range = role_ranges["relay"]
    for ch in INPUT_CHANNELS:
        connect(role_ranges[ch], relay_range, MEAN_OUT_DEGREE)
    connect(relay_range, relay_range, MEAN_OUT_DEGREE // 2)
    for ch in OUTPUT_CHANNELS:
        connect(relay_range, role_ranges[ch], MEAN_OUT_DEGREE)

    W = sp.csr_matrix((data, (rows, cols)), shape=(n, n))

    os.makedirs(out_dir, exist_ok=True)
    sp.save_npz(os.path.join(out_dir, "circuit.npz"), W)
    with open(os.path.join(out_dir, "circuit_meta.json"), "w") as f:
        json.dump({"source": "synthetic", "seed": seed, "nodes": nodes}, f)

    print(f"Synthetic circuit: {n} neurons, {W.nnz} directed edges")
    print(f"Input channels: {INPUT_CHANNELS} ({N_INPUT_PER_CHANNEL} neurons each)")
    print(f"Output channels: {OUTPUT_CHANNELS} ({N_OUTPUT} neurons)")
    print(f"Relay pool: {N_RELAY} neurons")
    return W, nodes


if __name__ == "__main__":
    build()
