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
import random
import time

import numpy as np

from .agents import EquityBot, FlyAgent
from .brain import LIFNetwork
from .decoder import SoftmaxDecoder
from .poker import Action, HeadsUpHand

BIG_BLIND = 20
STARTING_STACK = 2000


def run_hand(fly: FlyAgent, bot: EquityBot, rng: random.Random, button_seat: int):
    fly.new_hand()
    stacks = [STARTING_STACK, STARTING_STACK]
    agents = [fly, bot] if button_seat == 0 else [bot, fly]
    hand = HeadsUpHand(stacks, small_blind=BIG_BLIND // 2, big_blind=BIG_BLIND, rng=rng)
    result = hand.play(agents)
    fly_seat = 0 if button_seat == 0 else 1
    fly_payoff = result.payoff[fly_seat]
    return fly_payoff, result


def train(n_hands: int, out_path: str | None, log_every: int = 200, seed: int = 0,
          decoder_in: str | None = None):
    net = LIFNetwork()
    n_features = len(net.idx_by_role.get("out:action", []))
    decoder = SoftmaxDecoder(n_features=n_features, seed=seed)
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
    baselines = [0.0] * n_equity_buckets
    baseline_beta = 0.02

    def equity_bucket(equity: float) -> int:
        return min(int(equity * n_equity_buckets), n_equity_buckets - 1)

    recent_payoffs = []
    bluff_count = 0
    bluff_opportunities = 0
    aggressive_actions = {Action.RAISE_SMALL, Action.RAISE_BIG, Action.ALL_IN}
    strong_fold_count = 0
    strong_fold_opportunities = 0

    t0 = time.time()
    for i in range(1, n_hands + 1):
        button = i % 2
        payoff_bb = 0.0
        payoff, _ = run_hand(fly, bot, rng, button_seat=button)
        payoff_bb = payoff / BIG_BLIND
        recent_payoffs.append(payoff_bb)

        for step in fly.trajectory:
            bucket = equity_bucket(step["equity"])
            # Clip so one huge-pot outlier hand can't dominate thousands of
            # updates in one direction -- a further source of the
            # fold-collapse pathology (see decoder.reinforce_update's
            # entropy comment).
            advantage = np.clip(payoff_bb - baselines[bucket], -3.0, 3.0)
            decoder.reinforce_update(step["features"], step["action_idx"], step["probs"], advantage)
            baselines[bucket] = (1 - baseline_beta) * baselines[bucket] + baseline_beta * payoff_bb

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
