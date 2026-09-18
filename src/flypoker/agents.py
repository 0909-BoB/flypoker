"""Agent implementations satisfying poker.py's `.act(obs) -> Action` interface."""
from __future__ import annotations

import numpy as np

from .brain import LIFNetwork
from .decoder import SoftmaxDecoder, N_ACTIONS
from .encoder import encode
from .poker import Action, Observation, monte_carlo_equity

N_SIM_STEPS = 150


class FlyAgent:
    """A fly connectome (fixed) + trainable softmax decoder on its
    out:action spike pattern. Call `.act()` during play; trajectory of
    (features, action_idx, probs) for the hand is kept in `self.trajectory`
    for the training loop to consume after the hand resolves."""

    def __init__(self, net: LIFNetwork, decoder: SoftmaxDecoder | None = None,
                 seed: int = 0, reset_brain_each_decision: bool = True):
        self.net = net
        n_out = len(net.idx_by_role.get("out:action", []))
        self.decoder = decoder or SoftmaxDecoder(n_features=n_out, seed=seed)
        self.rng = np.random.default_rng(seed)
        self.reset_brain_each_decision = reset_brain_each_decision
        self.trajectory: list[dict] = []

    def new_hand(self):
        self.trajectory = []

    def act(self, obs: Observation) -> Action:
        if self.reset_brain_each_decision:
            self.net.reset()
        inputs, equity = encode(obs)
        self.net.run(inputs, n_steps=N_SIM_STEPS)
        features = self.net.output_spike_vector("out:action")

        legal_mask = np.ones(N_ACTIONS)
        if obs.to_call == 0:
            legal_mask[Action.FOLD] = 0.0

        action, probs = self.decoder.act(features, self.rng, legal_mask)
        self.trajectory.append({
            "features": features, "action_idx": int(action), "probs": probs,
            "equity": equity, "street": obs.street,
        })
        return action


class EquityBot:
    """Simple pot-odds baseline opponent: calls/raises based on Monte Carlo
    equity, with a small randomized bluff frequency so it isn't purely
    exploitable by a fold-when-behind strategy."""

    def __init__(self, seed: int = 0, bluff_freq: float = 0.08):
        self.rng = np.random.default_rng(seed)
        self.bluff_freq = bluff_freq

    def act(self, obs: Observation) -> Action:
        equity = monte_carlo_equity(obs.hole, obs.board, [], n_opponents=1, trials=150)
        pot_after_call = obs.pot + obs.to_call
        pot_odds = obs.to_call / pot_after_call if pot_after_call > 0 else 0.0

        bluffing = self.rng.random() < self.bluff_freq

        if obs.to_call == 0:
            if equity > 0.7 or bluffing:
                return Action.RAISE_BIG if equity > 0.85 else Action.RAISE_SMALL
            return Action.CHECK_CALL

        if equity < pot_odds and not bluffing:
            return Action.FOLD
        if equity > 0.75:
            return Action.RAISE_BIG if equity > 0.9 else Action.RAISE_SMALL
        if bluffing and equity < 0.4:
            return Action.RAISE_SMALL
        return Action.CHECK_CALL
