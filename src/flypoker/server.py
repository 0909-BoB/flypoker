"""FastAPI + WebSocket bridge: lets a browser play 4-handed NLHE (you + 3
flies) against the fly connectome. Each browser tab gets its own game loop
running in a background thread (the poker engine and brain sim are
synchronous); the thread blocks on a plain queue.Queue while waiting for
the human's action, and pushes JSON messages onto an outgoing queue that an
async task drains and sends over the websocket.

Run with:
    .venv/bin/python -m flypoker.server
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import random
import sys
import threading
import time

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from .agents import N_SIM_STEPS, FlyAgent, feature_dim
from .brain import LIFNetwork
from .decoder import SoftmaxDecoder
from .encoder import encode
from .learning import LearningHub, get_hub, step_to_log
from .sanity import SAN_START, after_decision, after_hand
from .poker import Action, MultiWayHand, Observation, best_hand, hand_category_name

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
RUNS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "runs")
DEFAULT_DECODER_PATH = os.path.join(RUNS_DIR, "decoder.npz")

DEFAULT_BIG_BLIND = 20
DEFAULT_STARTING_STACK = 2000
MAX_STARTING_STACK = 10_000_000

# Each connection runs its own game loop in a background thread and holds a
# LearningHub lock briefly per hand -- fine at hobby scale, but nothing was
# stopping an unbounded number of tabs/bots from each opening one and running
# it forever. A simple process-wide cap is the cheapest real guard against
# that before this is reachable from the open internet; MAX_CONCURRENT_GAMES
# is overridable per deployment (e.g. a bigger host might raise it).
MAX_CONCURRENT_GAMES = int(os.environ.get("MAX_CONCURRENT_GAMES", "60"))
_active_games_lock = threading.Lock()
_active_games = 0
FLY_NAMES = ["fly1", "fly2", "fly3"]
TURN_TIME_LIMIT = 15  # seconds the human gets per decision
TURN_ACK_GRACE = 6    # extra wait for the browser to confirm it showed the turn


def validate_game_config(raw_stack, raw_bb) -> tuple[int | None, int | None, str | None]:
    """The setup screen (frontend/index.html) lets the player pick a
    starting stack and big blind before the table opens; this is the
    server-side check for it -- the frontend validates too, but a client
    is never trusted to be the only guard. Returns (starting_stack,
    big_blind, None) on success or (None, None, message) on failure."""
    try:
        starting_stack = int(raw_stack)
        big_blind = int(raw_bb)
    except (TypeError, ValueError):
        return None, None, "起始籌碼跟大盲必須是整數"
    if starting_stack <= 0 or starting_stack > MAX_STARTING_STACK:
        return None, None, f"起始籌碼必須在 1 到 {MAX_STARTING_STACK} 之間"
    if big_blind <= 0:
        return None, None, "大盲必須大於 0"
    if big_blind > starting_stack:
        return None, None, "大盲不能大於起始籌碼"
    return starting_stack, big_blind, None

app = FastAPI()


def _json_default(obj):
    """numpy scalars/arrays sneak into messages (e.g. np.bool_ from a
    comparison); json.dumps rejects them."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


def card_str(c) -> str:
    return repr(c)


def human_hand_rank(hole: list, board: list) -> str | None:
    """Category name (e.g. "two_pair") of the best 5-card hand `hole` +
    `board` makes, for the live "牌面大小" display -- None before the flop,
    since best_hand() needs 5+ cards to evaluate."""
    if len(hole) + len(board) < 5:
        return None
    return hand_category_name(best_hand(hole + board))


# Personality anchors: the "aggressive" and "tight" bias presets this
# project already validated for stability (see README's "Tuning / known
# rough edges" -- getting a decoder that both tracks equity *and* doesn't
# collapse into an input-independent verdict took a lot of trial and
# error). Each fly's personality is now sampled fresh every game as a
# continuous blend between these two validated extremes, rather than
# picked from a few fixed named archetypes -- see sample_personality_bias.
_AGGRESSIVE_BIAS = {
    Action.FOLD: -0.4, Action.CHECK_CALL: -0.6,
    Action.RAISE_SMALL: 1.3, Action.RAISE_BIG: 1.6, Action.ALL_IN: 0.9,
}
_TIGHT_BIAS = {
    Action.FOLD: 1.8, Action.CHECK_CALL: 0.0,
    Action.RAISE_SMALL: -0.6, Action.RAISE_BIG: -0.9, Action.ALL_IN: -0.9,
}


def sample_personality_bias(rng: random.Random) -> tuple[dict[Action, float], float, float]:
    """Two independent axes, each sampled uniformly in [-1, 1]:
      - aggression: shifts RAISE_SMALL/RAISE_BIG/ALL_IN/CHECK_CALL bias
        toward the validated "aggressive" (+1) or "tight" (-1) extreme.
      - fold_tendency: same idea, independently, for FOLD bias (+1 folds
        more like "tight", -1 folds less like "aggressive").
    Interpolating between the two validated anchors -- rather than
    sampling arbitrary bias values from scratch -- keeps every draw inside
    the range already confirmed not to collapse the decoder.
    decoder.py's ACTION_FLOOR is still the hard backstop underneath this
    regardless of what gets sampled."""
    aggression = rng.uniform(-1.0, 1.0)
    fold_tendency = rng.uniform(-1.0, 1.0)

    def blend(action: Action, t: float) -> float:
        return (_AGGRESSIVE_BIAS[action] * t) if t >= 0 else (_TIGHT_BIAS[action] * -t)

    bias = {
        Action.RAISE_SMALL: blend(Action.RAISE_SMALL, aggression),
        Action.RAISE_BIG: blend(Action.RAISE_BIG, aggression),
        Action.ALL_IN: blend(Action.ALL_IN, aggression),
        Action.CHECK_CALL: blend(Action.CHECK_CALL, aggression),
        Action.FOLD: blend(Action.FOLD, fold_tendency),
    }
    return bias, aggression, fold_tendency


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
CIRCUIT_POSITIONS_PATH = os.path.join(DATA_DIR, "circuit_positions.json")
CIRCUIT_META_PATH = os.path.join(DATA_DIR, "circuit_meta.json")
CIRCUIT_LAYOUT_PATH = os.path.join(DATA_DIR, "circuit_layout.json")
_circuit_nodes_cache = None
_circuit_edges_cache = None


def load_circuit_nodes():
    """Each circuit neuron's role plus its real 3D position (see
    scripts/build_brain_3d.py), sent once per websocket connection for the
    3D brain view. None if positions haven't been built."""
    global _circuit_nodes_cache
    if _circuit_nodes_cache is None and os.path.exists(CIRCUIT_POSITIONS_PATH):
        with open(CIRCUIT_POSITIONS_PATH) as f:
            positions = json.load(f)["positions"]
        with open(CIRCUIT_META_PATH) as f:
            meta_nodes = json.load(f)["nodes"]
        _circuit_nodes_cache = [
            {"index": n["index"], "role": n["role"],
             "x": positions[str(n["index"])][0], "y": positions[str(n["index"])][1],
             "z": positions[str(n["index"])][2]}
            for n in meta_nodes if str(n["index"]) in positions
        ]
    return _circuit_nodes_cache


def load_circuit_edges() -> list[list[int]]:
    """The sampled real synapses (source, target) from the layout build, sent
    once so the 3D view can send light packets along actual connections."""
    global _circuit_edges_cache
    if _circuit_edges_cache is None:
        _circuit_edges_cache = []
        if os.path.exists(CIRCUIT_LAYOUT_PATH):
            with open(CIRCUIT_LAYOUT_PATH) as f:
                _circuit_edges_cache = [[e["source"], e["target"]] for e in json.load(f)["edges"]]
    return _circuit_edges_cache


def spike_events_for_ui(recording: list, max_total: int = 2500) -> list[list[int]]:
    """A decision's full spike recording (brain.py's LIFNetwork.run(...,
    record=True), routinely 15-20k events over 150 steps) is far too much to
    ship per decision. This keeps, per neuron, its *first* spike time (ms) and
    its total spike count (the animation's intensity), as [index, t_ms, count]
    rows sorted by time. If more than `max_total` neurons fired, it *strides*
    through the first-spike-ordered list instead of truncating, so the sample
    still spans the whole decision -- truncating to the earliest N would cut
    off whatever fires late, which can include the out:action neurons the
    decision itself reads off of."""
    if not recording:
        return []
    first: dict[int, float] = {}
    count: dict[int, int] = {}
    for t, idx in recording:
        if idx not in first:
            first[idx] = t
        count[idx] = count.get(idx, 0) + 1
    ordered = sorted(first.items(), key=lambda kv: kv[1])
    if len(ordered) > max_total:
        stride = len(ordered) / max_total
        ordered = [ordered[int(i * stride)] for i in range(max_total)]
    return [[int(idx), int(round(t)), int(count[idx])] for idx, t in ordered]


def raise_to_for_pot_mult(obs: Observation, mult: float, big_blind: int) -> int:
    """The "raise to" total for a pot-relative preset (0.5x pot, 1x pot,
    ...), using the same math as the engine's own RAISE_SMALL/RAISE_BIG
    sizing, so the UI's quick buttons land on the same amounts those names
    used to mean."""
    raise_amt = int(obs.pot * mult)
    total = obs.to_call + max(raise_amt, big_blind)
    max_raise_to = obs.my_bet + obs.my_stack
    return min(obs.my_bet + total, max_raise_to)


def obs_to_dict(obs: Observation, big_blind: int) -> dict:
    max_raise = obs.my_bet + obs.my_stack
    min_raise = raise_to_for_pot_mult(obs, 0.0, big_blind)  # smallest legal raise (to_call + BB)
    return {
        "street": obs.street,
        "board": [card_str(c) for c in obs.board],
        "hole": [card_str(c) for c in obs.hole],
        "hand_rank": human_hand_rank(obs.hole, obs.board),
        "pot": obs.pot,
        "to_call": obs.to_call,
        "my_stack": obs.my_stack,
        "opp_stack": obs.opp_stack,
        "my_bet": obs.my_bet,
        "is_button": obs.is_button,
        "legal_actions": [a.name for a in Action if not (a == Action.FOLD and obs.to_call == 0)],
        "min_raise": min(min_raise, max_raise),
        "max_raise": max_raise,
        "raise_presets": {
            "half_pot": raise_to_for_pot_mult(obs, 0.5, big_blind),
            "pot": raise_to_for_pot_mult(obs, 1.0, big_blind),
            "two_pot": raise_to_for_pot_mult(obs, 2.0, big_blind),
        },
    }


class QueueHumanAgent:
    def __init__(self, outgoing: queue.Queue, big_blind: int, session: dict | None = None):
        self.outgoing = outgoing
        self.incoming: queue.Queue = queue.Queue()
        self.big_blind = big_blind
        self.session = session
        self.turn_id = 0
        # Set by the game loop at the start of each hand so "your_turn" can
        # report every seat's live stack (not just this player's own), so
        # the other seats' stack displays update turn by turn instead of
        # only at hand_result.
        self.current_hand = None

    def act(self, obs: Observation):
        """One decision under a TURN_TIME_LIMIT clock. The clock starts when
        the browser acks that it actually *showed* this turn (the UI paces
        messages, so the turn can be queued behind animations for a few
        seconds -- starting the clock at send time would eat the player's
        time). Each turn has an id so a click that lands just after the
        timeout can't be mistaken for the answer to the *next* turn."""
        self.turn_id += 1
        msg = {"type": "your_turn", "turn_id": self.turn_id, "time_limit": TURN_TIME_LIMIT,
               "obs": obs_to_dict(obs, self.big_blind)}
        if self.current_hand is not None:
            msg["stacks"] = list(self.current_hand.stacks)
        self.outgoing.put(msg)

        fallback = Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD
        deadline = time.monotonic() + TURN_ACK_GRACE + TURN_TIME_LIMIT
        acked = False
        while True:
            if self.session is not None and self.session["stop"]:
                return fallback
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.outgoing.put({"type": "turn_timeout", "turn_id": self.turn_id, "action": fallback.name})
                return fallback
            try:
                reply = self.incoming.get(timeout=min(remaining, 0.5))
            except queue.Empty:
                continue
            if reply.get("turn_id") not in (None, self.turn_id):
                continue  # answer to an earlier, already-timed-out turn
            if reply.get("type") == "turn_ack":
                if not acked:
                    acked = True
                    deadline = time.monotonic() + TURN_TIME_LIMIT
                continue
            try:
                action = Action[reply["action"]]
            except KeyError:
                continue
            amount = reply.get("amount")
            if amount is not None and action != Action.ALL_IN:
                return action, int(amount)
            return action


def sync_fly_from_hub(agent: FlyAgent, hub: LearningHub):
    """Refresh a fly's decoder from the shared learned weights, with its own
    personality bias layered on top."""
    W, b = hub.snapshot()
    agent.decoder.W = W
    agent.decoder.b = b + agent.bias_vec


def make_fly_agent(outgoing: queue.Queue, seed: int, bias_delta: dict[Action, float] | None = None,
                    tag: str = "fly", hub: LearningHub | None = None) -> FlyAgent:
    net = LIFNetwork()
    decoder = SoftmaxDecoder(n_features=feature_dim(net), seed=seed)
    if hub is not None:
        decoder.W, decoder.b = hub.snapshot()
    elif os.path.exists(DEFAULT_DECODER_PATH):
        decoder.load(DEFAULT_DECODER_PATH)
    # Every fly shares the same trained W (equity-sensitivity) -- only the
    # bias shifts per personality, so a call with AA still reads as a call
    # with AA no matter which fly is looking at it.
    bias_vec = np.zeros(decoder.b.shape)
    for action, delta in (bias_delta or {}).items():
        bias_vec[int(action)] = delta
    decoder.b = decoder.b + bias_vec
    # Set by the game loop each hand so fly_decision can report live table
    # stacks (see QueueHumanAgent.current_hand above for the same idea).
    hand_ref = {"hand": None}

    def on_decide(obs, action, probs, equity, activity, spikes):
        agent.san, san_reason = after_decision(agent.san, obs, equity)
        msg = {
            "type": "fly_decision",
            "fly": tag,
            "seat": obs.seat,
            "street": obs.street,
            "board": [card_str(c) for c in obs.board],
            "action": action.name,
            "probs": [round(float(p), 3) for p in probs],
            "equity": round(float(equity), 3),
            "activity": activity,
            "spikes": spike_events_for_ui(spikes),
            "san": round(agent.san, 1),
            "san_reason": san_reason,
            "depth_bb": round(obs.my_stack / max(obs.big_blind, 1), 1),
            "commit_ratio": round(min(obs.to_call / obs.my_stack, 1.0), 2) if obs.my_stack > 0 else 1.0,
        }
        if hand_ref["hand"] is not None:
            msg["stacks"] = list(hand_ref["hand"].stacks)
            human_seat = hand_ref.get("human_seat")
            if human_seat is not None:
                msg["hand_rank"] = human_hand_rank(hand_ref["hand"].hole[human_seat], obs.board)
        outgoing.put(msg)

    agent = FlyAgent(net, decoder=decoder, seed=seed, on_decide=on_decide)
    agent.hand_ref = hand_ref
    agent.bias_vec = bias_vec
    agent.san = SAN_START
    return agent


def validate_rebuy(raw_amount, big_blind: int) -> tuple[int | None, str | None]:
    try:
        amount = int(raw_amount)
    except (TypeError, ValueError):
        return None, "補碼金額必須是整數"
    if amount < big_blind or amount > MAX_STARTING_STACK:
        return None, f"補碼金額必須在 {big_blind} 到 {MAX_STARTING_STACK} 之間"
    return amount, None


def wait_for_rebuy(session: dict, big_blind: int, outgoing: queue.Queue) -> int | None:
    """Blocks the game thread until the busted human rebuys (returns the
    validated amount) or leaves / disconnects (returns None)."""
    while not session["stop"]:
        try:
            msg = session["control"].get(timeout=0.5)
        except queue.Empty:
            continue
        if msg.get("type") == "leave":
            return None
        amount, err = validate_rebuy(msg.get("amount"), big_blind)
        if err:
            outgoing.put({"type": "error", "message": err})
            continue
        return amount
    return None


def game_loop(outgoing: queue.Queue, session: dict, starting_stack: int, big_blind: int):
    """You + three flies at a table, using MultiWayHand (side pots).
    Each fly's personality (sample_personality_bias) is randomized fresh
    for this game, so it's a new lineup of opponents every time. Identities
    keep a fixed relative seating order and the button rotates through all
    four each hand, same as a real table -- seat 0 isn't always the same
    identity. `starting_stack`/`big_blind` come from the setup screen (see
    ws_endpoint/validate_game_config) rather than being fixed constants."""
    identities = ["human"] + FLY_NAMES
    seed_base = random.randint(0, 1_000_000)

    personality_rng = random.Random()  # unseeded: a genuinely fresh draw every game
    try:
        hub = get_hub()
    except Exception as exc:  # no prior decoder on disk: play without learning
        print(f"[flypoker] learning disabled: {exc}", file=sys.stderr)
        hub = None
    flies = {}
    for i, name in enumerate(FLY_NAMES):
        bias, aggression, fold_tendency = sample_personality_bias(personality_rng)
        print(f"[flypoker] {name}: aggression={aggression:+.2f} fold_tendency={fold_tendency:+.2f}")
        flies[name] = make_fly_agent(outgoing, seed_base + i + 1, bias_delta=bias, tag=name, hub=hub)

    human = QueueHumanAgent(outgoing, big_blind, session)
    session["human"] = human

    circuit_nodes = load_circuit_nodes()
    if circuit_nodes is not None:
        # Sent once: neuron positions never change hand to hand, only which
        # neurons are active (each fly_decision/runout_street message's
        # spikes) does.
        outgoing.put({"type": "circuit", "nodes": circuit_nodes, "edges": load_circuit_edges()})

    rng = random.Random(seed_base)
    # Chips carry over from hand to hand like a real cash game: everyone
    # buys in for `starting_stack`, and a busted player has to rebuy (the
    # human is asked; flies just reload automatically) before the next deal.
    table_stacks = {name: starting_stack for name in identities}
    buy_ins = {name: starting_stack for name in identities}
    last_human_buy_in = starting_stack
    hand_i = 0

    # This runs in a background thread; an uncaught exception here would
    # otherwise just kill the thread silently and leave the browser's
    # websocket sender blocked forever on an empty queue -- the UI would
    # sit on "thinking..." forever with no indication anything went wrong.
    # Surface it to the frontend instead of hanging.
    try:
        while not session["stop"]:
            for name in FLY_NAMES:
                if table_stacks[name] <= 0:
                    table_stacks[name] = starting_stack
                    buy_ins[name] += starting_stack
                    outgoing.put({"type": "fly_rebuy", "fly": name, "amount": starting_stack})
            if table_stacks["human"] <= 0:
                outgoing.put({
                    "type": "busted", "suggested": last_human_buy_in,
                    "min": big_blind, "max": MAX_STARTING_STACK,
                })
                decision = wait_for_rebuy(session, big_blind, outgoing)
                if decision is None:
                    outgoing.put({"type": "left"})
                    break
                table_stacks["human"] = decision
                buy_ins["human"] += decision
                last_human_buy_in = decision

            for fly in flies.values():
                fly.new_hand()
                if hub is not None:
                    sync_fly_from_hub(fly, hub)  # pick up whatever has been learned so far
            seat_identity = [identities[(s + hand_i) % 4] for s in range(4)]
            agents = [human if ident == "human" else flies[ident] for ident in seat_identity]
            human_seat = seat_identity.index("human")
            stacks = [table_stacks[ident] for ident in seat_identity]
            hand = MultiWayHand(4, stacks, small_blind=big_blind // 2, big_blind=big_blind, rng=rng)
            human.current_hand = hand
            for fly in flies.values():
                fly.hand_ref["hand"] = hand
                fly.hand_ref["human_seat"] = human_seat

            outgoing.put({
                "type": "new_hand", "hand_number": hand_i + 1,
                "seats": seat_identity, "human_seat": human_seat,
                "human_hole": [card_str(c) for c in hand.hole[human_seat]],
                "small_blind": big_blind // 2, "big_blind": big_blind,
                "san": {name: round(flies[name].san, 1) for name in FLY_NAMES},
                "stacks": list(hand.stacks),
                "bankroll": {ident: table_stacks[ident] - buy_ins[ident] for ident in identities},
            })

            def on_street_dealt(street, board, hand=hand, seat_identity=seat_identity, human_seat=human_seat,
                                start_stacks=tuple(stacks)):
                # All-in run-out: cosmetic circuit/brain-activity display
                # only (no decision made), shown from whichever fly seat
                # comes first -- there's no single "the" fly at a 4-way
                # table the way heads-up has exactly one opponent.
                fly_seat = next(s for s in range(4) if seat_identity[s] != "human")
                fly_name = seat_identity[fly_seat]
                fly_agent = flies[fly_name]
                others = [i for i in range(4) if i != fly_seat]
                pot = sum(start_stacks) - sum(hand.stacks)
                obs = Observation(
                    seat=fly_seat, hole=hand.hole[fly_seat], board=board, street=street,
                    pot=pot, to_call=0, my_stack=hand.stacks[fly_seat],
                    opp_stack=sum(hand.stacks[i] for i in others), is_button=(fly_seat == 0),
                    active_opponents=len(others), big_blind=big_blind,
                    max_opp_stack=max(hand.stacks[i] for i in others),
                )
                inputs, equity = encode(obs)
                fly_agent.net.reset()
                fly_agent.net.run(inputs, n_steps=N_SIM_STEPS, record=True)
                outgoing.put({
                    "type": "runout_street",
                    "fly": fly_name,
                    "street": street,
                    "board": [card_str(c) for c in board],
                    "hand_rank": human_hand_rank(hand.hole[human_seat], board),
                    "activity": fly_agent.net.role_activity(),
                    "equity": round(float(equity), 3),
                    "spikes": spike_events_for_ui(fly_agent.net.last_recording),
                    "stacks": list(hand.stacks),
                })

            result = hand.play(agents, on_street_dealt=on_street_dealt)
            for s in range(4):
                table_stacks[seat_identity[s]] = stacks[s] + result.payoff[s]
            bankroll = {ident: table_stacks[ident] - buy_ins[ident] for ident in identities}

            showdown_holes = None
            if result.showdown:
                showdown_holes = {
                    seat_identity[s]: [card_str(c) for c in hand.hole[s]]
                    for s in result.showdown_seats if seat_identity[s] != "human"
                }

            for s in range(4):
                if seat_identity[s] != "human":
                    fly_agent = flies[seat_identity[s]]
                    fly_agent.san = after_hand(fly_agent.san, result.payoff[s] / big_blind)

            learning = None
            if hub is not None:
                try:
                    flies_log = {}
                    for s in range(4):
                        ident = seat_identity[s]
                        if ident == "human":
                            continue
                        steps = flies[ident].trajectory
                        payoff_bb = result.payoff[s] / big_blind
                        hub.learn_from_hand(steps, payoff_bb)
                        flies_log[ident] = {"payoff_bb": round(payoff_bb, 3),
                                            "steps": [step_to_log(st) for st in steps]}
                    hub.record_hand({
                        "config": {"stack": starting_stack, "bb": big_blind},
                        "seats": seat_identity, "human_seat": human_seat,
                        "holes": {seat_identity[s]: [card_str(c) for c in hand.hole[s]] for s in range(4)},
                        "payoffs": {seat_identity[s]: result.payoff[s] for s in range(4)},
                        "showdown": result.showdown, "log": result.log, "flies": flies_log,
                    })
                    learning = hub.status()
                except Exception as exc:  # disk / numeric trouble must never stall the table
                    print(f"[flypoker] learning step failed: {exc!r}", file=sys.stderr)

            outgoing.put({
                "type": "hand_result",
                "learning": learning,
                "san": {name: round(flies[name].san, 1) for name in FLY_NAMES},
                "log": result.log,
                "showdown": result.showdown,
                "showdown_holes": showdown_holes,
                "payoffs": {seat_identity[s]: result.payoff[s] for s in range(4)},
                "human_payoff": result.payoff[human_seat],
                "bankroll": bankroll,
                "final_stacks": {ident: table_stacks[ident] for ident in identities},
            })

            if session["stop"]:
                break
            hand_i += 1
    except Exception as exc:
        outgoing.put({"type": "error", "message": str(exc)})


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()

    global _active_games
    with _active_games_lock:
        if _active_games >= MAX_CONCURRENT_GAMES:
            await websocket.send_json({
                "type": "error",
                "message": "伺服器現在太多人在玩,請稍後再試一次。",
            })
            await websocket.close()
            return
        _active_games += 1

    try:
        await _run_game_session(websocket)
    finally:
        with _active_games_lock:
            _active_games -= 1


async def _run_game_session(websocket: WebSocket):
    # The setup screen (frontend/index.html) opens this connection and then
    # sends {"type": "start_game", starting_stack, big_blind} as its first
    # message, before anything is dealt -- wait for that (re-validating
    # server-side, never trusting the client alone) rather than dealing
    # with hardcoded defaults.
    starting_stack: int | None = None
    big_blind: int | None = None
    try:
        while starting_stack is None:
            data = await websocket.receive_json()
            if data.get("type") != "start_game":
                continue
            starting_stack, big_blind, err = validate_game_config(
                data.get("starting_stack"), data.get("big_blind")
            )
            if err:
                await websocket.send_json({"type": "error", "message": err})
                starting_stack = None
    except WebSocketDisconnect:
        return

    outgoing: queue.Queue = queue.Queue()
    session = {"human": None, "stop": False, "control": queue.Queue()}
    thread = threading.Thread(
        target=game_loop, args=(outgoing, session, starting_stack, big_blind), daemon=True
    )
    thread.start()

    async def sender():
        # If this task dies, the game thread keeps running but the browser
        # never hears about it again (a silent freeze) -- so a message that
        # can't be serialized is converted or dropped with a log line, never
        # allowed to kill the loop.
        while True:
            msg = await asyncio.to_thread(outgoing.get)
            try:
                await websocket.send_text(json.dumps(msg, default=_json_default))
            except (TypeError, ValueError) as exc:
                print(f"[flypoker] dropped unserializable {msg.get('type')!r} message: {exc!r}", file=sys.stderr)

    sender_task = asyncio.create_task(sender())
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") in ("action", "turn_ack") and session["human"] is not None:
                session["human"].incoming.put(data)
            elif data.get("type") in ("rebuy", "leave"):
                session["control"].put(data)
    except WebSocketDisconnect:
        pass
    finally:
        session["stop"] = True
        sender_task.cancel()


class NoCacheStaticFiles(StaticFiles):
    """Browsers heuristically cache static files that lack Cache-Control, so
    after a code update a tab can keep running a stale main.js -- which then
    silently ignores any new message type the server starts sending (e.g. the
    rebuy prompt) and just sits there. no-cache forces revalidation on every
    load (cheap: the ETag makes it a 304 when nothing changed)."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/", NoCacheStaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


def main():
    import uvicorn
    # HOST/PORT follow the common PaaS convention (Render/Railway/Fly inject
    # $PORT; HOST defaults to all-interfaces so a container's published port
    # actually reaches it) -- 127.0.0.1 only worked for local-machine-only use.
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8420"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
