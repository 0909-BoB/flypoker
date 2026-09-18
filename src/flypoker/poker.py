"""Heads-up no-limit Texas Hold'em with a discretized action space.

Kept deliberately small: two players, five actions (fold / check-call /
raise-small / raise-big / all-in), pot-relative bet sizing. That's enough
surface for bluffing and stack-awareness to matter without needing a full
continuous-bet-sizing engine.
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from enum import Enum, IntEnum

RANKS = "23456789TJQKA"
SUITS = "cdhs"


class Action(IntEnum):
    FOLD = 0
    CHECK_CALL = 1
    RAISE_SMALL = 2  # to (pot * 0.5) on top of the call
    RAISE_BIG = 3    # to (pot * 1.0) on top of the call
    ALL_IN = 4


ACTION_NAMES = {a: a.name for a in Action}

HAND_CATEGORY_NAMES = [
    "high_card", "pair", "two_pair", "trips", "straight",
    "flush", "full_house", "quads", "straight_flush",
]


@dataclass(frozen=True)
class Card:
    rank: int  # 2..14
    suit: int  # 0..3

    def __repr__(self):
        return f"{RANKS[self.rank - 2]}{SUITS[self.suit]}"


def make_deck() -> list[Card]:
    return [Card(r, s) for r in range(2, 15) for s in range(4)]


def evaluate_5(cards: tuple[Card, ...]) -> tuple:
    ranks = sorted((c.rank for c in cards), reverse=True)
    suits = [c.suit for c in cards]
    is_flush = len(set(suits)) == 1

    rank_counts: dict[int, int] = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1
    by_count = sorted(rank_counts.items(), key=lambda kv: (-kv[1], -kv[0]))
    counts = [c for _, c in by_count]
    ordered_ranks = [r for r, _ in by_count]

    unique_ranks = sorted(set(ranks), reverse=True)
    is_straight = False
    straight_high = None
    if len(unique_ranks) == 5:
        if unique_ranks[0] - unique_ranks[4] == 4:
            is_straight = True
            straight_high = unique_ranks[0]
        elif unique_ranks == [14, 5, 4, 3, 2]:
            is_straight = True
            straight_high = 5  # wheel

    if is_straight and is_flush:
        return (8, straight_high)
    if counts == [4, 1]:
        return (7, ordered_ranks[0], ordered_ranks[1])
    if counts == [3, 2]:
        return (6, ordered_ranks[0], ordered_ranks[1])
    if is_flush:
        return (5, *ranks)
    if is_straight:
        return (4, straight_high)
    if counts == [3, 1, 1]:
        return (3, *ordered_ranks)
    if counts == [2, 2, 1]:
        return (2, *ordered_ranks)
    if counts == [2, 1, 1, 1]:
        return (1, *ordered_ranks)
    return (0, *ranks)


def best_hand(cards: list[Card]) -> tuple:
    return max(evaluate_5(combo) for combo in itertools.combinations(cards, 5))


def hand_category_name(score: tuple) -> str:
    return HAND_CATEGORY_NAMES[score[0]]


def monte_carlo_equity(hole: list[Card], board: list[Card], dead: list[Card],
                        n_opponents: int = 1, trials: int = 200,
                        rng: random.Random | None = None) -> float:
    """Approximate win probability via random rollout against random opponent hands."""
    rng = rng or random
    used = set(hole) | set(board) | set(dead)
    deck = [c for c in make_deck() if c not in used]
    needed_board = 5 - len(board)

    wins = 0.0
    for _ in range(trials):
        rng.shuffle(deck)
        draw = deck[: n_opponents * 2 + needed_board]
        opp_holes = [draw[i * 2:i * 2 + 2] for i in range(n_opponents)]
        rest_board = board + draw[n_opponents * 2:]
        my_score = best_hand(hole + rest_board)
        opp_scores = [best_hand(h + rest_board) for h in opp_holes]
        best_opp = max(opp_scores)
        if my_score > best_opp:
            wins += 1.0
        elif my_score == best_opp:
            wins += 1.0 / (1 + sum(1 for s in opp_scores if s == my_score))
    return wins / trials


@dataclass
class Observation:
    seat: int
    hole: list[Card]
    board: list[Card]
    street: str
    pot: int
    to_call: int
    my_stack: int
    opp_stack: int
    is_button: bool
    street_actions: list[Action] = field(default_factory=list)


@dataclass
class HandResult:
    winner: int | None  # 0, 1, or None for split
    payoff: list[int]  # chip delta per seat
    showdown: bool
    final_pot: int
    log: list[str]


STREETS = ["preflop", "flop", "turn", "river"]


class HeadsUpHand:
    """Runs one hand of heads-up NLHE between two agents with a .act(obs) method.

    Seat 0 is the button/small blind, seat 1 is the big blind (standard
    heads-up convention: button acts first preflop, last postflop).
    """

    def __init__(self, stacks: list[int], small_blind: int = 10, big_blind: int = 20,
                 rng: random.Random | None = None):
        self.rng = rng or random.Random()
        deck = make_deck()
        self.rng.shuffle(deck)
        self.hole = [deck[0:2], deck[2:4]]
        self.board_deck = deck[4:]
        self.stacks = list(stacks)
        self.sb, self.bb = small_blind, big_blind

    def play(self, agents: list) -> HandResult:
        log: list[str] = []
        bets = [0, 0]
        stacks = self.stacks
        sb_amt = min(self.sb, stacks[0])
        bb_amt = min(self.bb, stacks[1])
        stacks[0] -= sb_amt
        stacks[1] -= bb_amt
        bets[0] += sb_amt
        bets[1] += bb_amt
        pot = 0
        board: list[Card] = []
        folded = [False, False]

        for street_i, street in enumerate(STREETS):
            if street != "preflop":
                n_new = {"flop": 3, "turn": 1, "river": 1}[street]
                board.extend(self.board_deck[:n_new])
                self.board_deck = self.board_deck[n_new:]
            if any(folded) or all(s == 0 for s in stacks) and street_i > 0:
                pass
            if not any(folded):
                folded_this_street, bets = self._betting_round(
                    agents, street, board, bets, stacks, folded, log
                )
                folded = folded_this_street
            if any(folded):
                break
            if stacks[0] == 0 or stacks[1] == 0:
                # all-in: run out remaining streets with no more betting
                continue

        pot = bets[0] + bets[1]
        while len(board) < 5 and not any(folded):
            n_new = min(1, 5 - len(board))
            board.extend(self.board_deck[:n_new])
            self.board_deck = self.board_deck[n_new:]

        if any(folded):
            winner = 1 if folded[0] else 0
            payoff = [0, 0]
            payoff[winner] = pot - bets[winner]
            payoff[1 - winner] = -bets[1 - winner]
            log.append(f"seat {1 - winner} folds, seat {winner} wins pot {pot}")
            return HandResult(winner, payoff, False, pot, log)

        score0 = best_hand(self.hole[0] + board)
        score1 = best_hand(self.hole[1] + board)
        if score0 > score1:
            winner = 0
        elif score1 > score0:
            winner = 1
        else:
            winner = None

        payoff = [-bets[0], -bets[1]]
        if winner is None:
            payoff[0] += pot / 2
            payoff[1] += pot / 2
        else:
            payoff[winner] += pot
        log.append(
            f"showdown board={board} h0={self.hole[0]}({hand_category_name(score0)}) "
            f"h1={self.hole[1]}({hand_category_name(score1)}) pot={pot} winner={winner}"
        )
        return HandResult(winner, payoff, True, pot, log)

    def _betting_round(self, agents, street, board, bets, stacks, folded, log):
        order = [0, 1] if street == "preflop" else [1, 0]
        actions_taken: list[Action] = []
        acted = {0: False, 1: False}
        current_bet = max(bets)

        while True:
            progressed = False
            for seat in order:
                if folded[seat]:
                    continue
                other = 1 - seat
                to_call = bets[other] - bets[seat]
                if acted[seat] and to_call == 0:
                    continue
                if stacks[seat] == 0:
                    acted[seat] = True
                    continue

                obs = Observation(
                    seat=seat, hole=self.hole[seat], board=list(board), street=street,
                    pot=bets[0] + bets[1], to_call=to_call, my_stack=stacks[seat],
                    opp_stack=stacks[other], is_button=(seat == 0),
                    street_actions=list(actions_taken),
                )
                action = agents[seat].act(obs)
                progressed = True
                acted[seat] = True

                if action == Action.FOLD and to_call > 0:
                    folded[seat] = True
                    actions_taken.append(action)
                    log.append(f"[{street}] seat{seat} folds")
                    return folded, bets

                if action == Action.CHECK_CALL:
                    call_amt = min(to_call, stacks[seat])
                    stacks[seat] -= call_amt
                    bets[seat] += call_amt
                    log.append(f"[{street}] seat{seat} " + ("checks" if to_call == 0 else f"calls {call_amt}"))
                else:
                    pot_now = bets[0] + bets[1]
                    if action == Action.RAISE_SMALL:
                        raise_amt = int(pot_now * 0.5)
                    elif action == Action.RAISE_BIG:
                        raise_amt = int(pot_now * 1.0)
                    else:  # ALL_IN
                        raise_amt = stacks[seat]
                    total = min(to_call + max(raise_amt, self.bb), stacks[seat])
                    stacks[seat] -= total
                    bets[seat] += total
                    if total < to_call + raise_amt:
                        acted[other] = False  # short all-in still lets opp act once more only if it was a raise
                    else:
                        acted[other] = False
                    log.append(f"[{street}] seat{seat} raises to {bets[seat]} ({action.name})")
                actions_taken.append(action)

            if bets[0] == bets[1] and all(acted.values()):
                break
            if not progressed:
                break
        return folded, bets
