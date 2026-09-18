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


def train(n_hands: int, out_path: str | None, log_every: int = 200, seed: int = 0):
    net = LIFNetwork()
    n_features = len(net.idx_by_role.get("out:action", []))
    decoder = SoftmaxDecoder(n_features=n_features, seed=seed)
    fly = FlyAgent(net, decoder=decoder, seed=seed)
    bot = EquityBot(seed=seed + 1)
    rng = random.Random(seed)

    baseline = 0.0
    baseline_beta = 0.02
    recent_payoffs = []
    bluff_count = 0
    bluff_opportunities = 0
    aggressive_actions = {Action.RAISE_SMALL, Action.RAISE_BIG, Action.ALL_IN}

    t0 = time.time()
    for i in range(1, n_hands + 1):
        button = i % 2
        payoff_bb = 0.0
        payoff, _ = run_hand(fly, bot, rng, button_seat=button)
        payoff_bb = payoff / BIG_BLIND
        recent_payoffs.append(payoff_bb)

        advantage = payoff_bb - baseline
        for step in fly.trajectory:
            decoder.reinforce_update(step["features"], step["action_idx"], step["probs"], advantage)
            if step["equity"] < 0.35:
                bluff_opportunities += 1
                if step["action_idx"] in aggressive_actions:
                    bluff_count += 1

        baseline = (1 - baseline_beta) * baseline + baseline_beta * payoff_bb

        if i % log_every == 0:
            avg = np.mean(recent_payoffs[-log_every:])
            bluff_rate = bluff_count / max(bluff_opportunities, 1)
            elapsed = time.time() - t0
            print(f"hand {i:6d}  avg_bb/hand={avg:+.2f}  bluff_rate(eq<0.35)={bluff_rate:.2%}  "
                  f"{i / elapsed:.0f} hands/s")

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
    args = ap.parse_args()
    train(args.hands, args.out, args.log_every, args.seed)


if __name__ == "__main__":
    main()
