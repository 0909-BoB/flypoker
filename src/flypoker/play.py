"""CLI to watch (or play against) the fly agent.

Usage:
    python -m flypoker.play --hands 5           # watch fly vs EquityBot
    python -m flypoker.play --hands 20 --decoder runs/decoder.npz
    python -m flypoker.play --human             # you play against the fly
"""
from __future__ import annotations

import argparse
import random

from .agents import EquityBot, FlyAgent
from .brain import LIFNetwork
from .decoder import SoftmaxDecoder
from .poker import Action, HeadsUpHand, Observation

BIG_BLIND = 20
STARTING_STACK = 2000

ACTION_PROMPT = "  [0] fold  [1] check/call  [2] raise small  [3] raise big  [4] all-in > "


class HumanAgent:
    def act(self, obs: Observation) -> Action:
        print(f"\n  street={obs.street}  board={obs.board}  your hole={obs.hole}")
        print(f"  pot={obs.pot}  to_call={obs.to_call}  your_stack={obs.my_stack}  opp_stack={obs.opp_stack}")
        while True:
            try:
                choice = int(input(ACTION_PROMPT).strip())
                action = Action(choice)
                if action == Action.FOLD and obs.to_call == 0:
                    print("  can't fold when there's nothing to call")
                    continue
                return action
            except (ValueError, KeyError):
                print("  invalid, try again")


def play_match(n_hands: int, decoder_path: str | None, human: bool, seed: int = 0):
    net = LIFNetwork()
    n_features = len(net.idx_by_role.get("out:action", []))
    decoder = SoftmaxDecoder(n_features=n_features, seed=seed)
    if decoder_path:
        decoder.load(decoder_path)
        print(f"loaded trained decoder from {decoder_path}")
    else:
        print("using an UNTRAINED decoder (random policy) -- pass --decoder to load a trained one")

    fly = FlyAgent(net, decoder=decoder, seed=seed)
    opponent = HumanAgent() if human else EquityBot(seed=seed + 1)
    rng = random.Random(seed)

    fly_bankroll = 0
    for i in range(n_hands):
        button = i % 2
        fly.new_hand()
        stacks = [STARTING_STACK, STARTING_STACK]
        agents = [fly, opponent] if button == 0 else [opponent, fly]
        hand = HeadsUpHand(stacks, small_blind=BIG_BLIND // 2, big_blind=BIG_BLIND, rng=rng)
        result = hand.play(agents)
        fly_seat = 0 if button == 0 else 1
        fly_bankroll += result.payoff[fly_seat]

        print(f"\n=== hand {i + 1} ===")
        for line in result.log:
            print(" ", line)
        for step in fly.trajectory:
            print(f"  fly[{step['street']}] action={Action(step['action_idx']).name} "
                  f"equity={step['equity']:.2f} probs={[round(p, 2) for p in step['probs']]}")
        print(f"  fly payoff this hand: {result.payoff[fly_seat]:+d}  running total: {fly_bankroll:+d}")

    print(f"\nFinal fly bankroll over {n_hands} hands: {fly_bankroll:+d} ({fly_bankroll / BIG_BLIND:+.1f} bb)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=5)
    ap.add_argument("--decoder", type=str, default=None)
    ap.add_argument("--human", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    play_match(args.hands, args.decoder, args.human, args.seed)


if __name__ == "__main__":
    main()
