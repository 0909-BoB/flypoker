"""Parses the raw IRC Poker Database format into training examples that
match flypoker's own encoder.py channels, for behavior-cloning the decoder
on real players' actions instead of only self-play.

Format (verified against the real files in data/hand_histories/IRCdata.tgz,
not just the CPRG documentation page -- game types are separate
<game_type>.<yyyymm>.tgz archives inside the outer tarball, each containing:

    <game_type>/<yyyymm>/hdb
    <game_type>/<yyyymm>/hroster
    <game_type>/<yyyymm>/pdb/pdb.<player>

hdb line (one per hand):
    timestamp  dealer  hand_num  num_players  flop  turn  river  showdown  [board cards]
    820783242  12      2         2            2/200 0/0   0/0    1/400    8h Qh Kh
  each stage field is "num_players_remaining/pot_size_at_start_of_stage".

hroster line (one per hand, same timestamp as hdb):
    timestamp  num_players  player1  player2  ...

pdb.<player> line (one per hand this player was dealt into):
    player  timestamp  num_players  position  preflop  flop  turn  river  bankroll  total_bet  total_win  [hole cards]
    gfw     820783242  2            2         Bk       f     -     -      295820    100        0
  action strings are per-street, one character per action this player took
  that street: B=post blind, f=fold, k=check, c=call, b=bet, r=raise,
  A=all-in, '-'=street not reached (already folded, or hand ended earlier).
  `position` 1 is the button/small blind, 2 is the big blind, matching this
  project's seat 0 / seat 1 convention.

IMPORTANT, real limitation of this data (not a parsing bug): a player's
hole cards are only ever recorded when they're shown, and IRC only shows
cards at genuine showdown. In heads-up, a fold always ends the hand *before*
showdown, so a folded player's hole cards are never in the data -- meaning
this dataset structurally cannot supply "this equity -> fold" examples.
Every example this module yields is for showdown hands only, so it only
carries information about calibrating aggression among hands that got
played out, not about when to give up. See pretrain.py for how the training
loop accounts for that (it never updates the FOLD row of the decoder from
this data).
"""
from __future__ import annotations

import tarfile
from dataclasses import dataclass
from io import BufferedReader

from .poker import Action, Card, RANKS, STREETS, SUITS

# The IRC network also logged several fixed-limit tables (holdem,
# holdem1/2/3, holdemii, holdempot, tourney) with pot-limit/fixed-limit
# betting rules that don't share this project's pot-relative no-limit
# sizing semantics. Only "nolimit" matches flypoker's own engine.
DEFAULT_GAME_TYPES = ("nolimit",)


def parse_card(token: str) -> Card:
    return Card(RANKS.index(token[0]) + 2, SUITS.index(token[1]))


@dataclass
class HdbRecord:
    timestamp: int
    num_players: int
    stage_pots: dict[str, int]  # street -> pot size at the *start* of that street
    board: list[Card]


def _parse_hdb_line(line: str) -> HdbRecord | None:
    parts = line.split()
    if len(parts) < 8:
        return None
    try:
        timestamp, _dealer, _hand_num, num_players = (int(x) for x in parts[:4])
        stage_pots = {}
        for street, tok in zip(("flop", "turn", "river", "showdown"), parts[4:8]):
            _n, pot = tok.split("/")
            stage_pots[street] = int(pot)
        board = [parse_card(c) for c in parts[8:]]
    except (ValueError, IndexError):
        return None
    return HdbRecord(timestamp, num_players, stage_pots, board)


@dataclass
class PdbRecord:
    player: str
    timestamp: int
    position: int
    street_actions: dict[str, str]
    bankroll: int
    cards: list[Card]


def _parse_pdb_line(line: str) -> PdbRecord | None:
    parts = line.split()
    if len(parts) < 11:
        return None
    try:
        player = parts[0]
        timestamp, _num_players, position = (int(x) for x in parts[1:4])
        street_actions = dict(zip(STREETS, parts[4:8]))
        bankroll = int(parts[8])
        cards = [parse_card(c) for c in parts[11:13]]
    except (ValueError, IndexError):
        return None
    return PdbRecord(player, timestamp, position, street_actions, bankroll, cards)


def _parse_hroster_line(line: str) -> tuple[int, list[str]] | None:
    parts = line.split()
    if len(parts) < 3:
        return None
    try:
        timestamp, num_players = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    return timestamp, parts[2:2 + num_players]


def _action_to_enum(raw: str, pot_before: int, pot_after: int) -> Action | None:
    """Map a raw per-street action string to this project's discretized
    Action, or None if it's not a real voluntary decision (blind-only,
    street not reached, or an action code this parser doesn't recognize --
    a few rare fixed-limit-only codes like 'Q'/'K' show up in the format's
    grammar but not in the no-limit tables; skip rather than guess)."""
    body = raw.replace("B", "", 1) if raw.startswith("B") else raw
    if not body or body == "-":
        return None
    if "A" in body:
        return Action.ALL_IN
    if body.endswith("f"):
        return Action.FOLD
    if all(c in "kc" for c in body):
        return Action.CHECK_CALL
    if all(c in "kcbr" for c in body):
        # a bet/raise happened this street; size it off how much the pot
        # grew relative to itself -- the closest thing to bet-relative
        # sizing this per-street-pot-only data supports.
        growth = (pot_after - pot_before) / max(pot_before, 1)
        return Action.RAISE_BIG if growth >= 1.0 else Action.RAISE_SMALL
    return None


def iter_showdown_examples(tgz_path: str, game_types=DEFAULT_GAME_TYPES, max_hands: int | None = None):
    """Yields dicts: {hole, board, street, pot_before, pot_after, is_button,
    action} for every real per-street decision made by either player in
    hands that reached showdown (see module docstring for why only
    showdown hands are usable). Only heads-up (num_players==2) hands are
    considered, matching this project's engine. `max_hands` caps the number
    of *hands* (not examples) consumed across the whole dataset."""
    hands_seen = 0
    with tarfile.open(tgz_path) as outer:
        for member in outer.getmembers():
            if max_hands and hands_seen >= max_hands:
                return
            if not member.name.endswith(".tgz"):
                continue
            fname = member.name.rsplit("/", 1)[-1]
            game_type = fname.split(".", 1)[0]
            if game_type not in game_types:
                continue
            folder = fname[: -len(".tgz")].replace(".", "/")
            group_file = outer.extractfile(member)
            if group_file is None:
                continue
            remaining = (max_hands - hands_seen) if max_hands else None
            with tarfile.open(fileobj=group_file) as inner:
                for example, is_new_hand in _iter_group(inner, folder, remaining):
                    if is_new_hand:
                        hands_seen += 1
                        if max_hands and hands_seen > max_hands:
                            return
                    else:
                        yield example


def _read_lines(tar: tarfile.TarFile, path: str) -> list[str]:
    f = tar.extractfile(path)
    if f is None:
        return []
    return f.read().decode("latin-1").splitlines()


def _iter_group(inner: tarfile.TarFile, folder: str, max_hands: int | None):
    """Yields (example_dict, False) for each training example and
    (None, True) once per hand processed (a sentinel so the caller can cap
    on hand count without this generator needing to know the running total
    across other groups)."""
    hdb_lines = _read_lines(inner, f"{folder}/hdb")
    hroster_lines = _read_lines(inner, f"{folder}/hroster")
    if not hdb_lines or not hroster_lines:
        return

    roster_by_ts: dict[int, list[str]] = {}
    for line in hroster_lines:
        parsed = _parse_hroster_line(line)
        if parsed:
            roster_by_ts[parsed[0]] = parsed[1]

    names = inner.getnames()
    pdb_by_player: dict[str, dict[int, PdbRecord]] = {}
    prefix = f"{folder}/pdb/pdb."
    for name in names:
        if not name.startswith(prefix):
            continue
        player = name[len(prefix):]
        by_ts: dict[int, PdbRecord] = {}
        for line in _read_lines(inner, name):
            rec = _parse_pdb_line(line)
            if rec:
                by_ts[rec.timestamp] = rec
        pdb_by_player[player] = by_ts

    hands_yielded = 0
    for line in hdb_lines:
        if max_hands and hands_yielded >= max_hands:
            return
        hdb = _parse_hdb_line(line)
        if hdb is None or hdb.num_players != 2:
            continue
        players = roster_by_ts.get(hdb.timestamp)
        if not players or len(players) != 2:
            continue
        recs = [pdb_by_player.get(p, {}).get(hdb.timestamp) for p in players]
        if any(r is None for r in recs):
            continue
        # Showdown filter: both players' hole cards known == hand went the
        # distance (see module docstring -- folded hands never show cards).
        if any(len(r.cards) != 2 for r in recs):
            continue

        # hdb's four pot fields are checkpoints *after* preflop/flop/turn/
        # river action respectively (named "flop"/"turn"/"river"/"showdown"
        # for what street that pot carries into) -- not one per STREETS
        # entry directly, so a decision on `street` reads the *next*
        # checkpoint for its "pot after this street's action" value.
        next_checkpoint = {"preflop": "flop", "flop": "turn", "turn": "river", "river": "showdown"}
        pot_before = 0
        for street in STREETS:
            pot_after = hdb.stage_pots.get(next_checkpoint[street], pot_before)
            board_so_far = hdb.board[: {"preflop": 0, "flop": 3, "turn": 4, "river": 5}[street]]
            for rec in recs:
                action = _action_to_enum(rec.street_actions.get(street, "-"), pot_before, pot_after)
                if action is None or action == Action.FOLD:
                    continue  # see module docstring: fold has no known equity here
                yield {
                    "hole": rec.cards,
                    "board": board_so_far,
                    "street": street,
                    "pot_before": max(pot_before, 1),
                    "pot_after": pot_after,
                    "bankroll": rec.bankroll,
                    "is_button": rec.position == 1,
                    "action": action,
                }, False
            pot_before = pot_after

        hands_yielded += 1
        yield None, True
