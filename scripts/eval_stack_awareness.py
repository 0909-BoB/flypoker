"""Behavior check for stack-awareness: same cards, same pot, only the stack
sizes change. If the fly reads stack depth / commitment, fold and shove
probabilities should move with them. Prints averages over a set of random
mid-strength spots, per decoder checkpoint.

    .venv/bin/python scripts/eval_stack_awareness.py runs/decoder.v1.npz runs/decoder.v2.npz
"""
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from flypoker.agents import FlyAgent, feature_dim
from flypoker.brain import LIFNetwork
from flypoker.decoder import SoftmaxDecoder
from flypoker.poker import Action, Observation, make_deck, monte_carlo_equity

BB = 20


def make_spots(n, seed=7):
    rng = random.Random(seed)
    spots = []
    while len(spots) < n:
        deck = make_deck()
        rng.shuffle(deck)
        hole, board = deck[:2], deck[2:5]
        eq = monte_carlo_equity(hole, board, [], n_opponents=1, trials=200, rng=rng)
        if 0.35 <= eq <= 0.65:
            spots.append((hole, board))
    return spots


def probs_for(agent, hole, board, pot, to_call, my_stack, opp_stack):
    obs = Observation(seat=0, hole=hole, board=board, street="flop", pot=pot, to_call=to_call,
                      my_stack=my_stack, opp_stack=opp_stack, is_button=False, my_bet=0,
                      big_blind=BB, max_opp_stack=opp_stack,
                      street_actions=[Action.RAISE_SMALL] if to_call > 0 else [])
    agent.new_hand()
    agent.act(obs)
    return agent.trajectory[-1]["probs"]


def main(paths):
    net = LIFNetwork()
    spots = make_spots(10)
    for path in paths:
        dec = SoftmaxDecoder(n_features=feature_dim(net))
        dec.load(path)
        agent = FlyAgent(net, decoder=dec, seed=1)
        print(f"\n== {path}")

        # facing a 60-chip bet into a 200 pot; only my stack changes
        print("facing a bet of 60 (pot 200): P(fold) as the call eats more of my stack")
        for my_stack in (3000, 600, 240, 120, 60):
            p = np.mean([probs_for(agent, h, b, 200, 60, my_stack, 3000)[Action.FOLD] for h, b in spots])
            print(f"  stack {my_stack:5d} ({my_stack / BB:5.0f}BB, commit {60 / my_stack:4.0%}): P(fold)={p:.2f}")

        print("unopened pot (pot 60): P(all-in) and P(fold-ish=passive) by depth")
        for my_stack in (3000, 1000, 400, 200):
            ps = np.array([probs_for(agent, h, b, 60, 0, my_stack, 3000) for h, b in spots]).mean(axis=0)
            print(f"  stack {my_stack:5d} ({my_stack / BB:5.0f}BB): P(all-in)={ps[Action.ALL_IN]:.2f}  "
                  f"P(raise)={ps[Action.RAISE_SMALL] + ps[Action.RAISE_BIG]:.2f}  P(check)={ps[Action.CHECK_CALL]:.2f}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["runs/decoder.npz"])
