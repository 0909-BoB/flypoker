"""Agent implementations satisfying poker.py's `.act(obs) -> Action` interface."""
from __future__ import annotations

import numpy as np

from .brain import LIFNetwork
from .decoder import SoftmaxDecoder, N_ACTIONS
from .encoder import encode
from .poker import Action, Observation, monte_carlo_equity

N_SIM_STEPS = 150
N_FEATURE_WINDOWS = 3


def feature_dim(net: LIFNetwork) -> int:
    return len(net.idx_by_role.get("out:action", [])) * N_FEATURE_WINDOWS


def brain_features(net: LIFNetwork) -> np.ndarray:
    """Decoder input for the run that just finished (needs run(record=True))."""
    return net.output_window_features("out:action", N_FEATURE_WINDOWS, N_SIM_STEPS)


class FlyAgent:
    """A fly connectome (fixed) + trainable softmax decoder on its
    out:action spike pattern. Call `.act()` during play; trajectory of
    (features, action_idx, probs) for the hand is kept in `self.trajectory`
    for the training loop to consume after the hand resolves."""

    def __init__(self, net: LIFNetwork, decoder: SoftmaxDecoder | None = None,
                 seed: int = 0, reset_brain_each_decision: bool = True, on_decide=None):
        self.net = net
        self.decoder = decoder or SoftmaxDecoder(n_features=feature_dim(net), seed=seed)
        self.rng = np.random.default_rng(seed)
        self.reset_brain_each_decision = reset_brain_each_decision
        self.trajectory: list[dict] = []
        self.last_activity: dict[str, int] = {}
        # optional callback(obs, action, probs, equity, activity, spikes) --
        # `spikes` is a list of (t_ms, neuron_idx) covering the whole
        # decision (for the web UI's live circuit view), or None if nobody
        # asked for it (self-play training leaves on_decide unset, so this
        # recording -- cheap, but not free -- never runs then).
        self.on_decide = on_decide

    def new_hand(self):
        self.trajectory = []

    def act(self, obs: Observation) -> Action:
        if self.reset_brain_each_decision:
            self.net.reset()
        inputs, equity = encode(obs)
        self.net.run(inputs, n_steps=N_SIM_STEPS, record=True)
        features = brain_features(self.net)

        legal_mask = np.ones(N_ACTIONS)
        if obs.to_call == 0:
            legal_mask[Action.FOLD] = 0.0
        # No point offering a sized raise when the call already commits the
        # whole stack, or the stack is a big blind or less -- those can only
        # ever be fold / call / shove.
        if obs.to_call >= obs.my_stack or obs.my_stack <= obs.big_blind:
            legal_mask[Action.RAISE_SMALL] = 0.0
            legal_mask[Action.RAISE_BIG] = 0.0

        action, probs = self.decoder.act(features, self.rng, legal_mask)
        self.last_activity = self.net.role_activity()
        self.trajectory.append({
            "features": features, "action_idx": int(action), "probs": probs,
            "equity": equity, "street": obs.street, "to_call": obs.to_call,
            "my_stack": obs.my_stack, "big_blind": obs.big_blind,
        })
        if self.on_decide:
            self.on_decide(obs, action, probs, equity, self.last_activity, self.net.last_recording)
        return action


class EquityBot:
    """Simple pot-odds baseline opponent: calls/raises based on Monte Carlo
    equity, with a small randomized bluff frequency so it isn't purely
    exploitable by a fold-when-behind strategy.

    Raise sizing is randomized across a wide range of pot multiples (not
    just two fixed presets) so a decoder trained against this bot sees --
    and learns to read -- the same variety of bet sizes a human free to
    pick their own amount will actually throw at it. Training only against
    fixed 0.5x/1.0x-pot raises left the decoder's read of "opponent
    aggression" poorly calibrated for anything else, part of why it kept
    over-folding once humans could bet arbitrary amounts.
    """

    def __init__(self, seed: int = 0, bluff_freq: float = 0.08):
        self.rng = np.random.default_rng(seed)
        self.bluff_freq = bluff_freq

    def _raise_to(self, obs: Observation, strength: float) -> tuple[Action, int]:
        # strength in [0, 1]: how big a bet this decision calls for. Mapped
        # to a randomized pot multiple so sizing varies hand to hand instead
        # of landing on the same one or two amounts every time.
        mult = self.rng.uniform(0.3, 0.9) if strength < 0.5 else self.rng.uniform(0.7, 2.0)
        raise_amt = max(int(obs.pot * mult), 1)
        total = obs.to_call + raise_amt
        raise_to = min(obs.my_bet + total, obs.my_bet + obs.my_stack)
        return Action.RAISE_SMALL, raise_to

    def act(self, obs: Observation):
        equity = monte_carlo_equity(obs.hole, obs.board, [], n_opponents=1, trials=150)
        pot_after_call = obs.pot + obs.to_call
        pot_odds = obs.to_call / pot_after_call if pot_after_call > 0 else 0.0

        bluffing = self.rng.random() < self.bluff_freq

        if obs.to_call == 0:
            if equity > 0.7 or bluffing:
                return self._raise_to(obs, equity if equity > 0.7 else 0.9)
            return Action.CHECK_CALL

        if equity < pot_odds and not bluffing:
            return Action.FOLD
        if equity > 0.75:
            return self._raise_to(obs, equity)
        if bluffing and equity < 0.4:
            return self._raise_to(obs, 0.3)
        return Action.CHECK_CALL
