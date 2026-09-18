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
        if legal_mask is not None:
            probs = probs * legal_mask
            if probs.sum() <= 0:
                probs = legal_mask / legal_mask.sum()
            else:
                probs = probs / probs.sum()
        action_idx = rng.choice(len(probs), p=probs)
        return Action(action_idx), probs

    def reinforce_update(self, features: np.ndarray, action_idx: int, probs: np.ndarray,
                          advantage: float, lr: float = 0.02):
        x = self._normalize(features)
        grad_logits = -probs
        grad_logits[action_idx] += 1.0
        self.W += lr * advantage * np.outer(grad_logits, x)
        self.b += lr * advantage * grad_logits

    def save(self, path: str):
        np.savez(path, W=self.W, b=self.b)

    def load(self, path: str):
        d = np.load(path)
        self.W, self.b = d["W"], d["b"]
