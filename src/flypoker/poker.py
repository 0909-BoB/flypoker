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
    my_bet: int = 0  # this seat's cumulative chips committed so far this hand
    active_opponents: int = 1  # non-folded opponents left; >1 at a multi-way table
    big_blind: int = 20
    max_opp_stack: int = 0  # deepest non-folded opponent's stack; 0 = unknown (fall back to opp_stack)


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

    def play(self, agents: list, on_street_dealt=None) -> HandResult:
        """Play one hand. `on_street_dealt(street, board)`, if given, fires
        for each street dealt *after* betting has already finished for the
        hand (both players all-in, or everyone already acted) -- i.e. the
        streets that would otherwise get dealt out silently in one shot with
        no agent ever seeing them. It does not fire for streets that still
        have a real betting round, since those are already visible via each
        agent's own .act(obs) call."""
        log: list[str] = []
        bets = [0, 0]
        stacks = self.stacks
        sb_amt = min(self.sb, stacks[0])
        bb_amt = min(self.bb, stacks[1])
        stacks[0] -= sb_amt
        stacks[1] -= bb_amt
        bets[0] += sb_amt
        bets[1] += bb_amt
        board: list[Card] = []
        folded = [False, False]
        all_in_runout = False

        for street in STREETS:
            if street != "preflop":
                n_new = {"flop": 3, "turn": 1, "river": 1}[street]
                board.extend(self.board_deck[:n_new])
                self.board_deck = self.board_deck[n_new:]
                if all_in_runout and on_street_dealt:
                    on_street_dealt(street, list(board))
            if not any(folded):
                folded_this_street, bets = self._betting_round(
                    agents, street, board, bets, stacks, folded, log
                )
                folded = folded_this_street
            if any(folded):
                break
            if stacks[0] == 0 or stacks[1] == 0:
                # all-in: remaining streets get dealt with no more betting
                all_in_runout = True

        pot = bets[0] + bets[1]

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
                    street_actions=list(actions_taken), my_bet=bets[seat],
                    big_blind=self.bb, max_opp_stack=stacks[other],
                )
                result = agents[seat].act(obs)
                # An agent may return a bare Action (fly/bot: fixed pot-relative
                # sizing below) or (Action, raise_to) to name an exact total
                # chip amount to raise to -- how a human player picks their own
                # bet size from a slider instead of two fixed presets.
                if isinstance(result, tuple):
                    action, raise_to = result
                else:
                    action, raise_to = result, None
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
                    if raise_to is not None:
                        total = max(to_call + self.bb, raise_to - bets[seat])
                        total = min(total, stacks[seat])
                    else:
                        pot_now = bets[0] + bets[1]
                        if action == Action.RAISE_SMALL:
                            raise_amt = int(pot_now * 0.5)
                        elif action == Action.RAISE_BIG:
                            raise_amt = int(pot_now * 1.0)
                        else:  # ALL_IN
                            raise_amt = stacks[seat]
                        total = min(to_call + max(raise_amt, self.bb), stacks[seat])
                        if stacks[seat] - total < max(self.bb, stacks[seat] / 3):
                            total = stacks[seat]  # same shove-instead-of-sliver rule as MultiWayHand
                    stacks[seat] -= total
                    bets[seat] += total
                    acted[other] = False
                    tag = f" ({action.name})" if raise_to is None else ""
                    log.append(f"[{street}] seat{seat} raises to {bets[seat]}{tag}")
                actions_taken.append(action)

            if bets[0] == bets[1] and all(acted.values()):
                break
            if not progressed:
                break
        return folded, bets


@dataclass
class SidePot:
    amount: int
    eligible_seats: list[int]


@dataclass
class MultiHandResult:
    payoff: list[int]  # chip delta per seat
    showdown: bool
    final_pot: int
    log: list[str]
    winners_by_pot: list[list[int]]
    showdown_seats: list[int] = field(default_factory=list)  # non-folded seats to reveal, if showdown


def compute_side_pots(bets: list[int], folded: list[bool]) -> list[SidePot]:
    """Standard side-pot layering: sort the distinct contribution levels,
    and for each layer charge everyone who put in at least that much (a
    player who went all-in for less doesn't pay into -- or win -- the
    layers above their own contribution)."""
    levels = sorted(set(b for b in bets if b > 0))
    pots: list[SidePot] = []
    prev = 0
    for level in levels:
        payers = [i for i, b in enumerate(bets) if b >= level]
        layer_amount = (level - prev) * len(payers)
        if layer_amount > 0:
            eligible = [i for i in payers if not folded[i]]
            # Two adjacent layers with the exact same eligible seats (e.g.
            # everyone who could contest the lower layer also folded or
            # matched exactly at the same point, common once only two
            # players are left contesting a pot no one else can win any
            # part of) split the money identically either way -- merge
            # them so the log doesn't show two "different" pots that are
            # really the same contest counted twice.
            if pots and pots[-1].eligible_seats == eligible:
                pots[-1] = SidePot(pots[-1].amount + layer_amount, eligible)
            else:
                pots.append(SidePot(layer_amount, eligible))
        prev = level
    return pots


class MultiWayHand:
    """N-player (N >= 2) no-limit Hold'em with side pots, for a table of
    more than one opponent. Seat 0 is the button, seat 1 the small blind,
    seat 2 the big blind, seats 3..N-1 the rest in table order (so seat 3
    is under the gun and acts first preflop, for N=4). Heads-up (N=2) is
    handled the same way HeadsUpHand does it (button/SB acts first
    preflop, last postflop) since the "everyone after the blinds" rotation
    used for N>2 doesn't apply with no seats past the blinds.

    Kept as a separate class from HeadsUpHand rather than folding N=2 into
    it, so the well-exercised 1-on-1 engine (and everything trained/tested
    against it) is untouched by this."""

    def __init__(self, n_players: int, stacks: list[int], small_blind: int = 10,
                 big_blind: int = 20, rng: random.Random | None = None):
        assert n_players >= 2
        self.n = n_players
        self.rng = rng or random.Random()
        deck = make_deck()
        self.rng.shuffle(deck)
        self.hole = [deck[2 * i:2 * i + 2] for i in range(n_players)]
        self.board_deck = deck[2 * n_players:]
        self.stacks = list(stacks)
        self.sb, self.bb = small_blind, big_blind

    def _preflop_order(self) -> list[int]:
        if self.n == 2:
            return [0, 1]
        return list(range(3, self.n)) + [0, 1, 2]

    def _postflop_order(self) -> list[int]:
        if self.n == 2:
            return [1, 0]
        return list(range(1, self.n)) + [0]

    def play(self, agents: list, on_street_dealt=None) -> MultiHandResult:
        log: list[str] = []
        bets = [0] * self.n
        stacks = self.stacks
        sb_seat, bb_seat = (0, 1) if self.n == 2 else (1, 2)
        sb_amt = min(self.sb, stacks[sb_seat])
        bb_amt = min(self.bb, stacks[bb_seat])
        stacks[sb_seat] -= sb_amt
        stacks[bb_seat] -= bb_amt
        bets[sb_seat] += sb_amt
        bets[bb_seat] += bb_amt
        board: list[Card] = []
        folded = [False] * self.n
        all_in_runout = False

        for street in STREETS:
            if street != "preflop":
                n_new = {"flop": 3, "turn": 1, "river": 1}[street]
                board.extend(self.board_deck[:n_new])
                self.board_deck = self.board_deck[n_new:]
                if all_in_runout and on_street_dealt:
                    on_street_dealt(street, list(board))
            active = [i for i in range(self.n) if not folded[i]]
            if len(active) > 1:
                folded, bets = self._betting_round(agents, street, board, bets, stacks, folded, log)
            active = [i for i in range(self.n) if not folded[i]]
            if len(active) <= 1:
                break
            if sum(1 for i in active if stacks[i] > 0) <= 1:
                all_in_runout = True

        pot = sum(bets)
        active = [i for i in range(self.n) if not folded[i]]

        if len(active) <= 1:
            winner = active[0]
            payoff = [-b for b in bets]
            payoff[winner] += pot
            log.append(f"seat {winner} wins uncontested pot {pot} (all others folded)")
            return MultiHandResult(payoff, False, pot, log, [[winner]])

        pots = compute_side_pots(bets, folded)
        payoff = [-b for b in bets]
        scores: dict[int, tuple] = {}
        winners_by_pot: list[list[int]] = []
        for side_pot in pots:
            for i in side_pot.eligible_seats:
                if i not in scores:
                    scores[i] = best_hand(self.hole[i] + board)
            best_score = max(scores[i] for i in side_pot.eligible_seats)
            winners = [i for i in side_pot.eligible_seats if scores[i] == best_score]
            share, remainder = divmod(side_pot.amount, len(winners))
            for k, w in enumerate(winners):
                payoff[w] += share + (1 if k < remainder else 0)
            winners_by_pot.append(winners)

        hands_desc = ", ".join(
            f"seat{i}={self.hole[i]}({hand_category_name(scores[i])})" for i in sorted(scores)
        )
        log.append(
            f"showdown board={board} {hands_desc} pots="
            f"{[(p.amount, p.eligible_seats) for p in pots]} winners_by_pot={winners_by_pot}"
        )
        return MultiHandResult(payoff, True, pot, log, winners_by_pot, showdown_seats=active)

    def _betting_round(self, agents, street, board, bets, stacks, folded, log):
        order = self._preflop_order() if street == "preflop" else self._postflop_order()
        actions_taken: list[Action] = []
        acted = {i: False for i in range(self.n)}

        while True:
            progressed = False
            for seat in order:
                if folded[seat]:
                    continue
                active_seats = [i for i in range(self.n) if not folded[i]]
                current_max = max(bets[i] for i in active_seats)
                to_call = current_max - bets[seat]
                if acted[seat] and to_call == 0:
                    continue
                if stacks[seat] == 0:
                    acted[seat] = True
                    continue

                others = [i for i in active_seats if i != seat]
                obs = Observation(
                    seat=seat, hole=self.hole[seat], board=list(board), street=street,
                    pot=sum(bets), to_call=to_call, my_stack=stacks[seat],
                    opp_stack=sum(stacks[i] for i in others), is_button=(seat == 0),
                    street_actions=list(actions_taken), my_bet=bets[seat],
                    active_opponents=len(others), big_blind=self.bb,
                    max_opp_stack=max(stacks[i] for i in others),
                )
                result = agents[seat].act(obs)
                if isinstance(result, tuple):
                    action, raise_to = result
                else:
                    action, raise_to = result, None
                progressed = True
                acted[seat] = True

                if action == Action.FOLD and to_call > 0:
                    folded[seat] = True
                    actions_taken.append(action)
                    log.append(f"[{street}] seat{seat} folds")
                    if sum(1 for i in range(self.n) if not folded[i]) <= 1:
                        return folded, bets
                    continue

                if action == Action.CHECK_CALL:
                    call_amt = min(to_call, stacks[seat])
                    stacks[seat] -= call_amt
                    bets[seat] += call_amt
                    log.append(f"[{street}] seat{seat} " + ("checks" if to_call == 0 else f"calls {call_amt}"))
                else:
                    if raise_to is not None:
                        total = max(to_call + self.bb, raise_to - bets[seat])
                        total = min(total, stacks[seat])
                    else:
                        pot_now = sum(bets)
                        if action == Action.RAISE_SMALL:
                            raise_amt = int(pot_now * 0.5)
                        elif action == Action.RAISE_BIG:
                            raise_amt = int(pot_now * 1.0)
                        else:  # ALL_IN
                            raise_amt = stacks[seat]
                        total = min(to_call + max(raise_amt, self.bb), stacks[seat])
                        # A preset raise that would leave only a sliver
                        # behind (under a big blind, or under a third of the
                        # stack) is effectively a shove already -- treat it
                        # as one instead of stranding a meaningless tail.
                        leftover = stacks[seat] - total
                        if leftover < max(self.bb, stacks[seat] / 3):
                            total = stacks[seat]
                    stacks[seat] -= total
                    bets[seat] += total
                    for i in range(self.n):
                        if i != seat and not folded[i]:
                            acted[i] = False
                    acted[seat] = True
                    tag = f" ({action.name})" if raise_to is None else ""
                    log.append(f"[{street}] seat{seat} raises to {bets[seat]}{tag}")
                actions_taken.append(action)

            active_seats = [i for i in range(self.n) if not folded[i]]
            current_max = max(bets[i] for i in active_seats)
            if all((stacks[i] == 0) or (acted[i] and bets[i] == current_max) for i in active_seats):
                break
            if not progressed:
                break
        return folded, bets
