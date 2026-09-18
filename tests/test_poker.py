import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from flypoker.poker import Action, Card, HeadsUpHand, best_hand, evaluate_5, hand_category_name
import random


def C(s):
    r = "23456789TJQKA".index(s[0]) + 2
    su = "cdhs".index(s[1])
    return Card(r, su)


def test_straight_flush_beats_quads():
    sf = best_hand([C("As"), C("Ks"), C("Qs"), C("Js"), C("Ts"), C("2c"), C("3d")])
    quads = best_hand([C("Ah"), C("Ad"), C("As"), C("Ac"), C("Kd"), C("2c"), C("3d")])
    assert sf > quads
    assert hand_category_name(sf) == "straight_flush"


def test_wheel_straight():
    h = best_hand([C("Ah"), C("2d"), C("3s"), C("4c"), C("5d"), C("9c"), C("Jd")])
    assert hand_category_name(h) == "straight"


class AlwaysCall:
    def act(self, obs):
        return Action.FOLD if obs.to_call > obs.my_stack + 1000 else Action.CHECK_CALL


def test_hand_runs_to_showdown_without_crashing():
    rng = random.Random(42)
    for _ in range(20):
        hand = HeadsUpHand([500, 500], small_blind=10, big_blind=20, rng=rng)
        result = hand.play([AlwaysCall(), AlwaysCall()])
        assert sum(result.payoff) == 0
        assert result.final_pot >= 0


if __name__ == "__main__":
    test_straight_flush_beats_quads()
    test_wheel_straight()
    test_hand_runs_to_showdown_without_crashing()
    print("all tests passed")
