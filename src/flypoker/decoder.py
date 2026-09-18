"""Trainable readout: turns the brain's out:action spike-count vector into a
poker decision. The connectome itself is fixed (not differentiable, not
trained) -- this small softmax layer on top is the only thing that learns,
via REINFORCE. See README for why: real fly wiring can't be taught pot odds,
but a linear decoder reading its spike patterns can be.
"""
from __future__ import annotations

import numpy as np

from .poker import Action

N_ACTIONS = len(Action)

# However well training goes, the decoder must never become fully
# deterministic: mixing a small uniform floor into the sampling
# distribution guarantees no legal action's probability can collapse all
# the way to 0, independent of what the logits say. This is a hard,
# training-independent backstop for the fold-collapse pathology (see
# reinforce_update's comment) -- entropy regularization alone slows the
# collapse but its own gradient vanishes exactly as probabilities
# saturate, so it isn't a reliable guarantee by itself.
ACTION_FLOOR = 0.12


class SoftmaxDecoder:
    def __init__(self, n_features: int, n_actions: int = N_ACTIONS, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(scale=0.05, size=(n_actions, n_features))
        self.b = np.zeros(n_actions)
        self.n_features = n_features

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        # spike counts can vary a lot in scale; squash so training is stable
        return np.tanh(features / 5.0)

    def policy(self, features: np.ndarray) -> np.ndarray:
        x = self._normalize(features)
        logits = self.W @ x + self.b
        logits -= logits.max()
        p = np.exp(logits)
        return p / p.sum()

    def act(self, features: np.ndarray, rng: np.random.Generator, legal_mask: np.ndarray | None = None):
        probs = self.policy(features)
        mask = legal_mask if legal_mask is not None else np.ones_like(probs)
        probs = probs * mask
        if probs.sum() <= 0:
            probs = mask / mask.sum()
        else:
            probs = probs / probs.sum()
        probs = (1 - ACTION_FLOOR) * probs + ACTION_FLOOR * (mask / mask.sum())
        action_idx = rng.choice(len(probs), p=probs)
        return Action(action_idx), probs

    def reinforce_update(self, features: np.ndarray, action_idx: int, probs: np.ndarray,
                          advantage: float, lr: float = 0.015, entropy_coef: float = 0.05,
                          weight_decay: float = 0.01):
        x = self._normalize(features)
        grad_logits = -probs
        grad_logits[action_idx] += 1.0

        # Plain REINFORCE with a single shared hand-level reward per decision
        # is prone to collapsing onto one action (e.g. "always fold to any
        # raise", regardless of equity) once that action's variance-avoidance
        # gets reinforced a few times in a row -- there's nothing pulling the
        # policy back toward exploring once probs saturate near 0/1. An
        # entropy bonus (gradient of softmax entropy wrt logits) counteracts
        # that: it pushes low-probability actions back up a bit every step,
        # independent of the sampled action's advantage.
        log_probs = np.log(np.clip(probs, 1e-8, 1.0))
        entropy = -np.sum(probs * log_probs)
        entropy_grad = -probs * (log_probs + entropy)

        total_grad = advantage * grad_logits + entropy_coef * entropy_grad
        self.W += lr * np.outer(total_grad, x)
        self.b += lr * total_grad

        # The entropy bonus above only pushes back on saturation once it's
        # already happening (and its own gradient vanishes right as probs
        # approach 0/1, which is exactly when it's needed most). Weight
        # decay bounds ||W|| directly, so logits can't grow large enough to
        # saturate the softmax in the first place -- observed in practice as
        # the decoder outputting the *same* near-certain action regardless
        # of wildly different equity inputs once W got large enough that the
        # bias/weight dot-products dwarfed genuine input variation.
        #
        # Decaying only W turned out to open a worse escape hatch: with W
        # suppressed, the bias b (never decayed) became the dominant term in
        # the logits, so the decoder still ignored the input entirely -- it
        # just converged on a different input-independent verdict (always
        # call instead of always fold). b needs the same pressure.
        self.W -= lr * weight_decay * self.W
        self.b -= lr * weight_decay * self.b

    def supervised_update(self, features: np.ndarray, action_idx: int,
                           class_mask: np.ndarray, lr: float = 0.05):
        """One cross-entropy gradient step toward `action_idx`, restricted
        to the classes where `class_mask` is truthy -- the rows for masked-
        out classes get *exactly* zero update. Used by pretrain.py to
        imitate real players' actions from hand history data that can only
        ever supply non-fold labels (see hand_history_parser.py's
        docstring): masking out FOLD means this can never move the
        decoder's fold behavior in either direction, good or bad."""
        x = self._normalize(features)
        logits = self.W @ x + self.b
        logits = np.where(class_mask, logits, -np.inf)
        logits -= np.max(logits)
        p = np.where(class_mask, np.exp(logits), 0.0)
        p = p / p.sum()

        grad_logits = -p
        grad_logits[action_idx] += 1.0
        grad_logits = grad_logits * class_mask

        self.W += lr * np.outer(grad_logits, x)
        self.b += lr * grad_logits

    def save(self, path: str):
        np.savez(path, W=self.W, b=self.b)

    def load(self, path: str):
        d = np.load(path)
        self.W, self.b = d["W"], d["b"]
