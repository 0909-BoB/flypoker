"""Online learning across games: every finished hand is logged, and the flies'
shared decoder is nudged by REINFORCE from the chips each fly won or lost.

What learns: only the decoder's readout weights (the connectome stays fixed,
as everywhere else in this project). All three flies share one learner; each
fly's personality bias is layered on top of the shared weights at sync time,
so personalities stay distinct while what they *know* accumulates.

Why it's conservative -- plain REINFORCE against a noisy table collapsed into
over-folding / over-shoving every time it was tried at scale (see README), so:
  * small learning rate, advantages clipped, baselines per (stack depth,
    equity bucket) so a good call with a strong hand isn't blamed for a bad
    runout;
  * an anchor pulls the weights back toward the distilled prior every step;
  * total drift from the prior is capped;
  * a health guard watches recent behavior (folding strong hands, folding or
    shoving too much overall) and pauses learning -- while the anchor keeps
    pulling back -- if the policy starts to degenerate.

Files (under runs/):
    decoder.npz            distilled prior (never modified here)
    decoder.learned.npz    current learned weights
    learning_state.json    baselines, counters, recent-behavior window
    hand_log.jsonl         one JSON line per hand (see record_hand)

    .venv/bin/python -m flypoker.learning replay   # rebuild learned weights
                                                   # from hand_log.jsonl
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import deque

import numpy as np

from .decoder import N_ACTIONS, SoftmaxDecoder
from .poker import Action

RUNS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "runs")
BASE_PATH = os.path.join(RUNS_DIR, "decoder.npz")
LEARNED_PATH = os.path.join(RUNS_DIR, "decoder.learned.npz")
STATE_PATH = os.path.join(RUNS_DIR, "learning_state.json")
LOG_PATH = os.path.join(RUNS_DIR, "hand_log.jsonl")

LR = 0.002
ANCHOR = 0.0005          # per-update pull of the weights back toward the prior
UNHEALTHY_ANCHOR = 0.01  # much stronger pull while the guard has learning paused
# Behavioral trust region: on any state a fly has seen, an action's probability
# may not be pushed above RATIO_MAX x, or below RATIO_MIN x, what the prior
# would give it. (Weight-space drift alone doesn't bound behavior: a 5% weight
# change swung shoving from 10% to 51% in simulation.)
RATIO_MAX = 1.5
RATIO_MIN = 1.0 / 1.5
MAX_REL_DRIFT = 0.30     # ||W - W0|| may not exceed this fraction of ||W0||
ADV_CLIP = 3.0
BASELINE_BETA = 0.02
WINDOW = 300

# Health guard thresholds over the recent-decision window.
MAX_STRONG_FOLD = 0.35   # folding equity>0.65 hands facing a bet
MAX_FOLD_RATE = 0.60
MAX_SHOVE_RATE = 0.22
MIN_SAMPLES = 60


def _depth_bucket(stack: float, big_blind: float) -> int:
    bb = stack / max(big_blind, 1)
    return 0 if bb < 25 else (1 if bb < 80 else 2)


def _equity_bucket(equity: float) -> int:
    return min(int(equity * 5), 4)


class LearningHub:
    def __init__(self, base_path=BASE_PATH, learned_path=None, state_path=None, log_path=None):
        # FLYPOKER_LEARN_DIR redirects everything the hub *writes* (tests use
        # it so they never touch the real learned weights / hand log); the
        # prior is always read from base_path.
        out_dir = os.environ.get("FLYPOKER_LEARN_DIR", RUNS_DIR)
        self.learned_path = learned_path or os.path.join(out_dir, "decoder.learned.npz")
        self.state_path = state_path or os.path.join(out_dir, "learning_state.json")
        self.log_path = log_path or os.path.join(out_dir, "hand_log.jsonl")
        self.lock = threading.Lock()

        base = np.load(base_path)
        self.W0, self.b0 = base["W"].copy(), base["b"].copy()
        self.decoder = SoftmaxDecoder(n_features=self.W0.shape[1])
        self.decoder.W, self.decoder.b = self.W0.copy(), self.b0.copy()

        self.baselines = [[0.0] * 5 for _ in range(3)]
        self.hands = 0
        self.updates = 0
        self.skipped = 0
        self.recent = deque(maxlen=WINDOW)  # (equity, faced_bet, action_idx)
        self._load_state()

    # -- persistence ---------------------------------------------------------
    def _load_state(self):
        if os.path.exists(self.learned_path):
            d = np.load(self.learned_path)
            if d["W"].shape == self.W0.shape:
                self.decoder.W, self.decoder.b = d["W"].copy(), d["b"].copy()
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path) as f:
                    s = json.load(f)
                self.baselines = s["baselines"]
                self.hands, self.updates, self.skipped = s["hands"], s["updates"], s["skipped"]
                self.recent.extend(tuple(x) for x in s.get("recent", []))
            except (ValueError, KeyError):
                pass

    def save(self):
        os.makedirs(os.path.dirname(self.learned_path), exist_ok=True)
        np.savez(self.learned_path, W=self.decoder.W, b=self.decoder.b)
        with open(self.state_path, "w") as f:
            json.dump({"baselines": self.baselines, "hands": self.hands, "updates": self.updates,
                       "skipped": self.skipped, "recent": [list(x) for x in self.recent]}, f)

    # -- reading -------------------------------------------------------------
    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        with self.lock:
            return self.decoder.W.copy(), self.decoder.b.copy()

    def drift(self) -> float:
        num = np.linalg.norm(self.decoder.W - self.W0)
        return float(num / max(np.linalg.norm(self.W0), 1e-9))

    def healthy(self) -> bool:
        if len(self.recent) < MIN_SAMPLES:
            return True
        arr = list(self.recent)
        fold = np.mean([a == Action.FOLD for _, _, a in arr])
        shove = np.mean([a == Action.ALL_IN for _, _, a in arr])
        strong = [a == Action.FOLD for e, faced, a in arr if faced and e > 0.65]
        strong_fold = np.mean(strong) if len(strong) >= 20 else 0.0
        return bool(strong_fold <= MAX_STRONG_FOLD and fold <= MAX_FOLD_RATE and shove <= MAX_SHOVE_RATE)

    def status(self) -> dict:
        return {"hands": self.hands, "updates": self.updates, "skipped": self.skipped,
                "drift": round(self.drift(), 3), "healthy": self.healthy()}

    # -- learning ------------------------------------------------------------
    def learn_from_hand(self, steps: list[dict], payoff_bb: float):
        """One fly's decisions in one hand, credited with that fly's net
        result (in big blinds). Steps are FlyAgent.trajectory entries."""
        if not steps:
            return
        with self.lock:
            dec = self.decoder
            for st in steps:
                faced = st["to_call"] > 0
                self.recent.append((float(st["equity"]), faced, int(st["action_idx"])))
            learning_ok = self.healthy()
            for st in steps:
                anchor = ANCHOR
                if learning_ok:
                    d = _depth_bucket(st["my_stack"], st["big_blind"])
                    e = _equity_bucket(st["equity"])
                    advantage = float(np.clip(payoff_bb - self.baselines[d][e], -ADV_CLIP, ADV_CLIP))
                    self.baselines[d][e] = (1 - BASELINE_BETA) * self.baselines[d][e] + BASELINE_BETA * payoff_bb
                    x = np.asarray(st["features"])
                    a = int(st["action_idx"])
                    ratio = self._policy(self.decoder.W, self.decoder.b, x)[a] / max(
                        self._policy(self.W0, self.b0, x)[a], 1e-6)
                    blocked = (advantage > 0 and ratio > RATIO_MAX) or (advantage < 0 and ratio < RATIO_MIN)
                    if blocked:
                        self.skipped += 1
                    else:
                        dec.reinforce_update(x, a, np.array(st["probs"], dtype=float), advantage,
                                             lr=LR, weight_decay=0.0)
                        self.updates += 1
                else:
                    self.skipped += 1
                    anchor = UNHEALTHY_ANCHOR
                # Anchor toward the prior every step, learning or paused.
                dec.W -= anchor * (dec.W - self.W0)
                dec.b -= anchor * (dec.b - self.b0)
            self._cap_drift()

    def _policy(self, W, b, features) -> np.ndarray:
        x = np.tanh(features / 5.0)
        logits = W @ x + b
        logits -= logits.max()
        p = np.exp(logits)
        return p / p.sum()

    def _cap_drift(self):
        dec = self.decoder
        dW = dec.W - self.W0
        limit = MAX_REL_DRIFT * np.linalg.norm(self.W0)
        n = np.linalg.norm(dW)
        if n > limit:
            dec.W = self.W0 + dW * (limit / n)
        db = dec.b - self.b0
        b_limit = MAX_REL_DRIFT * max(np.linalg.norm(self.b0), 1.0)
        nb = np.linalg.norm(db)
        if nb > b_limit:
            dec.b = self.b0 + db * (b_limit / nb)

    # -- recording -----------------------------------------------------------
    def record_hand(self, record: dict):
        """Append one hand to hand_log.jsonl and count it. Feature vectors are
        stored as small ints so a hand can be replayed later."""
        record = dict(record, t=time.strftime("%Y-%m-%dT%H:%M:%S"))
        with self.lock:
            self.hands += 1
            record["n"] = self.hands
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(json.dumps(record, separators=(",", ":")) + "\n")
            self.save()


def step_to_log(st: dict) -> dict:
    return {
        "street": st["street"], "action": int(st["action_idx"]),
        "probs": [round(float(p), 3) for p in st["probs"]],
        "equity": round(float(st["equity"]), 3), "to_call": int(st["to_call"]),
        "my_stack": int(st["my_stack"]), "bb": int(st["big_blind"]),
        "features": [int(x) for x in st["features"]],
    }


def step_from_log(d: dict) -> dict:
    return {
        "street": d["street"], "action_idx": d["action"], "probs": d["probs"],
        "equity": d["equity"], "to_call": d["to_call"], "my_stack": d["my_stack"],
        "big_blind": d["bb"], "features": d["features"],
    }


_hub: LearningHub | None = None
_hub_lock = threading.Lock()


def get_hub() -> LearningHub:
    global _hub
    with _hub_lock:
        if _hub is None:
            _hub = LearningHub()
        return _hub


def replay(log_path: str | None = None):
    """Rebuild the learned weights from scratch by replaying the hand log in
    order (a fresh start from the prior, one pass)."""
    out_dir = os.environ.get("FLYPOKER_LEARN_DIR", RUNS_DIR)
    for name in ("decoder.learned.npz", "learning_state.json"):
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            os.remove(path)
    hub = LearningHub(log_path=os.devnull)
    n = 0
    with open(log_path or os.path.join(out_dir, "hand_log.jsonl")) as f:
        for line in f:
            rec = json.loads(line)
            for fly in rec.get("flies", {}).values():
                hub.learn_from_hand([step_from_log(s) for s in fly["steps"]], fly["payoff_bb"])
            hub.hands += 1
            n += 1
    hub.save()
    print(f"replayed {n} hands -> {hub.status()}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "replay":
        replay()
    else:
        print(json.dumps(get_hub().status(), indent=2))
