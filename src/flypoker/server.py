"""FastAPI + WebSocket bridge: lets a browser play heads-up NLHE against the
fly connectome. Each browser tab gets its own game loop running in a
background thread (the poker engine and brain sim are synchronous); the
thread blocks on a plain queue.Queue while waiting for the human's action,
and pushes JSON messages onto an outgoing queue that an async task drains
and sends over the websocket.

Run with:
    .venv/bin/python -m flypoker.server
"""
from __future__ import annotations

import asyncio
import os
import queue
import random
import threading

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from .agents import N_SIM_STEPS, FlyAgent
from .brain import LIFNetwork
from .decoder import SoftmaxDecoder
from .encoder import encode
from .poker import Action, HeadsUpHand, Observation

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
DEFAULT_DECODER_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "runs", "decoder.npz")

BIG_BLIND = 20
STARTING_STACK = 2000

app = FastAPI()


def card_str(c) -> str:
    return repr(c)


def raise_to_for_pot_mult(obs: Observation, mult: float) -> int:
    """The "raise to" total for a pot-relative preset (0.5x pot, 1x pot,
    ...), using the same math as the engine's own RAISE_SMALL/RAISE_BIG
    sizing, so the UI's quick buttons land on the same amounts those names
    used to mean."""
    raise_amt = int(obs.pot * mult)
    total = obs.to_call + max(raise_amt, BIG_BLIND)
    max_raise_to = obs.my_bet + obs.my_stack
    return min(obs.my_bet + total, max_raise_to)


def obs_to_dict(obs: Observation) -> dict:
    max_raise = obs.my_bet + obs.my_stack
    min_raise = raise_to_for_pot_mult(obs, 0.0)  # smallest legal raise (to_call + BB)
    return {
        "street": obs.street,
        "board": [card_str(c) for c in obs.board],
        "hole": [card_str(c) for c in obs.hole],
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
            "half_pot": raise_to_for_pot_mult(obs, 0.5),
            "pot": raise_to_for_pot_mult(obs, 1.0),
            "two_pot": raise_to_for_pot_mult(obs, 2.0),
        },
    }


class QueueHumanAgent:
    def __init__(self, outgoing: queue.Queue):
        self.outgoing = outgoing
        self.incoming: queue.Queue = queue.Queue()

    def act(self, obs: Observation):
        self.outgoing.put({"type": "your_turn", "obs": obs_to_dict(obs)})
        msg = self.incoming.get()
        action = Action[msg["action"]]
        amount = msg.get("amount")
        if amount is not None and action != Action.ALL_IN:
            return action, int(amount)
        return action


def make_fly_agent(outgoing: queue.Queue, seed: int) -> FlyAgent:
    net = LIFNetwork()
    decoder = SoftmaxDecoder(n_features=len(net.idx_by_role.get("out:action", [])), seed=seed)
    if os.path.exists(DEFAULT_DECODER_PATH):
        decoder.load(DEFAULT_DECODER_PATH)

    def on_decide(obs, action, probs, equity, activity):
        outgoing.put({
            "type": "fly_decision",
            "street": obs.street,
            "action": action.name,
            "probs": [round(float(p), 3) for p in probs],
            "equity": round(float(equity), 3),
            "activity": activity,
        })

    return FlyAgent(net, decoder=decoder, seed=seed, on_decide=on_decide)


def game_loop(outgoing: queue.Queue, session: dict):
    seed = random.randint(0, 1_000_000)
    fly = make_fly_agent(outgoing, seed)
    human = QueueHumanAgent(outgoing)
    session["human"] = human

    rng = random.Random(seed)
    bankroll = {"human": 0, "fly": 0}
    hand_i = 0

    # This runs in a background thread; an uncaught exception here would
    # otherwise just kill the thread silently and leave the browser's
    # websocket sender blocked forever on an empty queue -- the UI would
    # sit on "thinking..." forever with no indication anything went wrong.
    # Surface it to the frontend instead of hanging.
    try:
        while not session["stop"]:
            button = hand_i % 2
            fly.new_hand()
            stacks = [STARTING_STACK, STARTING_STACK]
            agents = [human, fly] if button == 0 else [fly, human]
            human_seat = 0 if button == 0 else 1
            hand = HeadsUpHand(stacks, small_blind=BIG_BLIND // 2, big_blind=BIG_BLIND, rng=rng)

            outgoing.put({
                "type": "new_hand", "hand_number": hand_i + 1,
                "human_seat": human_seat, "human_hole": [card_str(c) for c in hand.hole[human_seat]],
                "small_blind": BIG_BLIND // 2, "big_blind": BIG_BLIND,
            })

            def on_street_dealt(street, board, hand=hand, human_seat=human_seat):
                # Both players are already all-in with no more decisions left
                # this hand -- this is purely cosmetic (lets the brain panel
                # keep flickering through the run-out instead of going dark),
                # so approximate pot/stack rather than threading the real
                # betting state out of HeadsUpHand: total chips in play never
                # changes, so pot = 2*STARTING_STACK - what's left in stacks.
                fly_seat = 1 - human_seat
                pot = 2 * STARTING_STACK - sum(hand.stacks)
                obs = Observation(
                    seat=fly_seat, hole=hand.hole[fly_seat], board=board, street=street,
                    pot=pot, to_call=0, my_stack=hand.stacks[fly_seat],
                    opp_stack=hand.stacks[human_seat], is_button=(fly_seat == 0),
                )
                inputs, equity = encode(obs)
                fly.net.reset()
                fly.net.run(inputs, n_steps=N_SIM_STEPS)
                outgoing.put({
                    "type": "runout_street",
                    "street": street,
                    "board": [card_str(c) for c in board],
                    "activity": fly.net.role_activity(),
                    "equity": round(float(equity), 3),
                })

            result = hand.play(agents, on_street_dealt=on_street_dealt)
            bankroll["human"] += result.payoff[human_seat]
            bankroll["fly"] += result.payoff[1 - human_seat]

            outgoing.put({
                "type": "hand_result",
                "log": result.log,
                "showdown": result.showdown,
                "fly_hole": [card_str(c) for c in hand.hole[1 - human_seat]] if result.showdown else None,
                "human_payoff": result.payoff[human_seat],
                "bankroll": dict(bankroll),
            })

            if session["stop"]:
                break
            hand_i += 1
    except Exception as exc:
        outgoing.put({"type": "error", "message": str(exc)})


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    outgoing: queue.Queue = queue.Queue()
    session = {"human": None, "stop": False}
    thread = threading.Thread(target=game_loop, args=(outgoing, session), daemon=True)
    thread.start()

    async def sender():
        while True:
            msg = await asyncio.to_thread(outgoing.get)
            await websocket.send_json(msg)

    sender_task = asyncio.create_task(sender())
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "action" and session["human"] is not None:
                session["human"].incoming.put(data)
    except WebSocketDisconnect:
        pass
    finally:
        session["stop"] = True
        sender_task.cancel()


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8420)


if __name__ == "__main__":
    main()
