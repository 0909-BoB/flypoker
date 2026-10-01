"""Supervised pretraining ("behavior cloning") for the decoder, using real
players' actions from data/hand_histories/IRCdata.tgz (see
hand_history_parser.py) instead of starting train.py's self-play REINFORCE
from fully random weights.

Only teaches the four *non-fold* actions -- see hand_history_parser.py's
module docstring for why fold labels structurally can't come from this
data (a folded hand never reveals hole cards, so there's no equity to
learn from). `decoder.SoftmaxDecoder.supervised_update` masks the FOLD row
out of every gradient step here, so this script can never move fold
behavior in either direction; train.py's REINFORCE loop remains the only
thing that ever touches it.

Usage:
    .venv/bin/python -m flypoker.pretrain --hands 20000 --out runs/decoder.pretrained.npz

    # then warm-start REINFORCE self-play from it:
    .venv/bin/python -m flypoker.train --hands 6000 --decoder-in runs/decoder.pretrained.npz --out runs/decoder.npz
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from .agents import N_SIM_STEPS, brain_features, feature_dim
from .brain import LIFNetwork
from .decoder import SoftmaxDecoder
from .encoder import INJECT_GAIN
from .hand_history_parser import iter_showdown_examples
from .poker import Action, STREETS, monte_carlo_equity

NON_FOLD_MASK = np.array([0.0 if a == Action.FOLD else 1.0 for a in Action])


def _encode_example(ex: dict) -> dict[str, float]:
    """Mirrors encoder.py's channel set as closely as this dataset allows.
    Real limitations, disclosed rather than smoothed over:
      - equity uses the same Monte Carlo rollout encoder.py itself uses
        (only needs this player's hole cards + board, not the actual
        opponent's cards), so it's on equal footing with in-engine play.
      - pot_odds/spr are built from the per-street pot checkpoints and
        starting bankroll this format actually records. There's no exact
        per-action to-call in the source data, so pot_odds approximates it
        as half of the pot's growth across the street (see
        hand_history_parser.py's `_action_to_enum` for the matching
        raise-size heuristic).
      - opp_aggression has no reliable source here (the format records
        each player's own action string, not the interleaved back-and-forth
        within a street) and is left at a neutral 0 rather than guessed.
    """
    equity = monte_carlo_equity(ex["hole"], ex["board"], [], n_opponents=1, trials=150)
    pot_before, pot_after = ex["pot_before"], ex["pot_after"]
    to_call_approx = max((pot_after - pot_before) / 2, 0)
    pot_odds = to_call_approx / (pot_before + to_call_approx) if (pot_before + to_call_approx) > 0 else 0.0
    spr = ex["bankroll"] / pot_before if pot_before > 0 else 5.0
    spr_norm = min(spr / 10.0, 1.0)
    street_norm = STREETS.index(ex["street"]) / (len(STREETS) - 1)
    return {
        "in:equity": equity * INJECT_GAIN,
        "in:pot_odds": pot_odds * INJECT_GAIN,
        "in:spr": spr_norm * INJECT_GAIN,
        # Stack-aware channels: only commit has a usable source here (the
        # dataset has no big-blind or opponent-stack info), the rest stay
        # neutral so pretraining doesn't bake in a made-up stack signal.
        "in:commit": min(to_call_approx / ex["bankroll"], 1.0) * INJECT_GAIN if ex["bankroll"] > 0 else 0.0,
        "in:depth": 0.0,
        "in:rel_stack": 0.5 * INJECT_GAIN,
        "in:street": street_norm * INJECT_GAIN,
        "in:position": (1.0 if ex["is_button"] else 0.0) * INJECT_GAIN,
        "in:opp_aggression": 0.0,
    }


def pretrain(tgz_path: str, max_hands: int, out_path: str, seed: int = 0,
             lr: float = 0.05, log_every: int = 500) -> SoftmaxDecoder:
    net = LIFNetwork()
    decoder = SoftmaxDecoder(n_features=feature_dim(net), seed=seed)

    t0 = time.time()
    n = 0
    correct = 0
    for ex in iter_showdown_examples(tgz_path, max_hands=max_hands):
        inputs = _encode_example(ex)
        net.reset()
        net.run(inputs, n_steps=N_SIM_STEPS, record=True)
        features = brain_features(net)

        probs = decoder.policy(features)
        predicted = int(np.argmax(np.where(NON_FOLD_MASK > 0, probs, -1.0)))
        correct += int(predicted == int(ex["action"]))

        decoder.supervised_update(features, int(ex["action"]), NON_FOLD_MASK, lr=lr)
        n += 1
        if n % log_every == 0:
            elapsed = time.time() - t0
            print(f"examples {n:6d}  train_acc(non-fold, running)={correct / n:.2%}  {n / elapsed:.1f} ex/s")

    if n == 0:
        raise RuntimeError(
            "No training examples extracted -- check --tgz points at a real "
            "IRCdata.tgz and that it contains 'nolimit' game-type archives."
        )
    print(f"done: {n} examples, final running train_acc(non-fold)={correct / n:.2%}")
    decoder.save(out_path)
    print(f"saved pretrained decoder to {out_path}")
    return decoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tgz", type=str, default="data/hand_histories/IRCdata.tgz")
    ap.add_argument("--hands", type=int, default=10_000,
                     help="cap on hands consumed from the dataset (not examples; "
                          "each qualifying hand yields several street-level examples)")
    ap.add_argument("--out", type=str, default="runs/decoder.pretrained.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--log-every", type=int, default=500)
    args = ap.parse_args()
    pretrain(args.tgz, args.hands, args.out, args.seed, args.lr, args.log_every)


if __name__ == "__main__":
    main()
