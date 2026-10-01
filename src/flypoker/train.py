"""Self-play training loop: the fly agent plays many heads-up hands against
EquityBot, and its decoder is updated via REINFORCE (Monte Carlo policy
gradient, one shared reward per hand across all of that hand's decisions,
with a running-mean baseline to cut variance). The connectome itself never
changes -- only the readout weights do.

Usage:
    python -m flypoker.train --hands 5000 --out runs/decoder.npz
"""
from __future__ import annotations

import argparse
import math
import random
import time

import numpy as np

from .agents import EquityBot, FlyAgent, feature_dim
from .brain import LIFNetwork
from .decoder import SoftmaxDecoder
from .poker import Action, HeadsUpHand

BIG_BLIND = 20
MIN_DEPTH_BB, MAX_DEPTH_BB = 8, 200


def sample_stack(rng: random.Random) -> int:
    """Log-uniform depth between 8BB and 200BB, so short stacks are seen as
    often as deep ones -- the decoder has to learn stack-aware play, which it
    can't do if every training hand starts at the same 100BB."""
    depth_bb = math.exp(rng.uniform(math.log(MIN_DEPTH_BB), math.log(MAX_DEPTH_BB)))
    return int(depth_bb * BIG_BLIND)


def depth_bucket(stack: int) -> int:
    bb = stack / BIG_BLIND
    return 0 if bb < 25 else (1 if bb < 80 else 2)


def run_hand(fly: FlyAgent, bot: EquityBot, rng: random.Random, button_seat: int):
    fly.new_hand()
    fly_seat = 0 if button_seat == 0 else 1
    stacks = [0, 0]
    stacks[fly_seat] = sample_stack(rng)
    stacks[1 - fly_seat] = sample_stack(rng)
    agents = [fly, bot] if button_seat == 0 else [bot, fly]
    hand = HeadsUpHand(stacks, small_blind=BIG_BLIND // 2, big_blind=BIG_BLIND, rng=rng)
    result = hand.play(agents)
    fly_payoff = result.payoff[fly_seat]
    return fly_payoff, result, stacks[fly_seat]


def train(n_hands: int, out_path: str | None, log_every: int = 200, seed: int = 0,
          decoder_in: str | None = None):
    net = LIFNetwork()
    decoder = SoftmaxDecoder(n_features=feature_dim(net), seed=seed)
    if decoder_in:
        # Warm-start from pretrain.py's behavior-cloned weights instead of
        # random init. That script never touches the FOLD row (see its
        # docstring for why), so REINFORCE here is still the only thing
        # that ever teaches fold/continue -- it just no longer also has to
        # learn "what does a made hand look like" from scratch at the same
        # time, which is what kept collapsing (see README).
        decoder.load(decoder_in)
        print(f"warm-started decoder from {decoder_in}")
    fly = FlyAgent(net, decoder=decoder, seed=seed)
    bot = EquityBot(seed=seed + 1)
    rng = random.Random(seed)

    # A single global baseline mixes together every decision ever made,
    # regardless of how strong the hand was at the time. That silently
    # miscredits variance: a *correct* call with AA gets the hand's whole
    # final payoff as its advantage, so if the hand later loses to a bad
    # runout (which happens plenty even with AA), the good call gets
    # punished -- while folding never gets blamed for anything, since
    # folding ends the trajectory immediately and never sees the runout.
    # Repeated over thousands of hands this reliably teaches "fold is
    # always safe," independent of equity. Conditioning the baseline on
    # the equity bucket at decision time compares each decision against
    # the average outcome of *similar-strength* hands instead, so a call
    # with AA is judged against other strong-hand outcomes, not diluted
    # by the mass of weak preflop folds.
    n_equity_buckets = 5
    # Also keyed by starting depth (short / mid / deep): a 10BB stack's
    # typical outcome distribution is nothing like a 150BB stack's, so one
    # shared baseline per equity bucket would mis-credit stack-driven variance.
    baselines = [[0.0] * n_equity_buckets for _ in range(3)]
    baseline_beta = 0.02

    def equity_bucket(equity: float) -> int:
        return min(int(equity * n_equity_buckets), n_equity_buckets - 1)

    recent_payoffs = []
    commit_stats = {"hi": [0, 0], "lo": [0, 0]}  # [opportunities, folds], mid-equity, faced a bet
    shove_stats = [[0, 0], [0, 0], [0, 0]]       # per depth bucket: [decisions, all-ins]
    bluff_count = 0
    bluff_opportunities = 0
    aggressive_actions = {Action.RAISE_SMALL, Action.RAISE_BIG, Action.ALL_IN}
    strong_fold_count = 0
    strong_fold_opportunities = 0

    t0 = time.time()
    for i in range(1, n_hands + 1):
        button = i % 2
        payoff_bb = 0.0
        payoff, _, fly_stack = run_hand(fly, bot, rng, button_seat=button)
        d_bucket = depth_bucket(fly_stack)
        payoff_bb = payoff / BIG_BLIND
        recent_payoffs.append(payoff_bb)

        for step in fly.trajectory:
            bucket = equity_bucket(step["equity"])
            # Clip so one huge-pot outlier hand can't dominate thousands of
            # updates in one direction -- a further source of the
            # fold-collapse pathology (see decoder.reinforce_update's
            # entropy comment).
            advantage = np.clip(payoff_bb - baselines[d_bucket][bucket], -3.0, 3.0)
            decoder.reinforce_update(step["features"], step["action_idx"], step["probs"], advantage)
            baselines[d_bucket][bucket] = (1 - baseline_beta) * baselines[d_bucket][bucket] + baseline_beta * payoff_bb

            if step["to_call"] > 0 and 0.35 <= step["equity"] <= 0.65:
                commit = step["to_call"] / max(step["my_stack"], 1)
                key = "hi" if commit > 0.4 else ("lo" if commit < 0.1 else None)
                if key:
                    commit_stats[key][0] += 1
                    commit_stats[key][1] += int(step["action_idx"] == Action.FOLD)
            if step["action_idx"] == Action.ALL_IN:
                shove_stats[d_bucket][1] += 1
            shove_stats[d_bucket][0] += 1

            if step["equity"] < 0.35:
                bluff_opportunities += 1
                if step["action_idx"] in aggressive_actions:
                    bluff_count += 1
            if step["equity"] > 0.65 and step["to_call"] > 0:
                strong_fold_opportunities += 1
                if step["action_idx"] == Action.FOLD:
                    strong_fold_count += 1

        if i % log_every == 0:
            avg = np.mean(recent_payoffs[-log_every:])
            bluff_rate = bluff_count / max(bluff_opportunities, 1)
            strong_fold_rate = strong_fold_count / max(strong_fold_opportunities, 1)
            elapsed = time.time() - t0
            print(f"hand {i:6d}  avg_bb/hand={avg:+.2f}  bluff_rate(eq<0.35)={bluff_rate:.2%}  "
                  f"fold_rate(eq>0.65,faced bet)={strong_fold_rate:.2%}  {i / elapsed:.0f} hands/s")
            fold_hi = commit_stats["hi"][1] / max(commit_stats["hi"][0], 1)
            fold_lo = commit_stats["lo"][1] / max(commit_stats["lo"][0], 1)
            shoves = [f"{a / max(n, 1):.0%}" for n, a in shove_stats]
            print(f"             stack-awareness: fold@commit>40%={fold_hi:.0%} vs fold@commit<10%={fold_lo:.0%}"
                  f"  shove_rate short/mid/deep={'/'.join(shoves)}")

    if out_path:
        decoder.save(out_path)
        print(f"saved decoder to {out_path}")
    return decoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=3000)
    ap.add_argument("--out", type=str, default="runs/decoder.npz")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--decoder-in", type=str, default=None,
                     help="warm-start from a decoder checkpoint (e.g. pretrain.py's output) "
                          "instead of random init")
    args = ap.parse_args()
    train(args.hands, args.out, args.log_every, args.seed, args.decoder_in)


if __name__ == "__main__":
    main()
