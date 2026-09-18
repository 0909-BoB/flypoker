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

from .agents import FlyAgent
from .brain import LIFNetwork
from .decoder import SoftmaxDecoder
from .poker import Action, HeadsUpHand, Observation

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
DEFAULT_DECODER_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "runs", "decoder.npz")

BIG_BLIND = 20
STARTING_STACK = 2000

app = FastAPI()


def card_str(c) -> str:
    return repr(c)


def obs_to_dict(obs: Observation) -> dict:
    return {
        "street": obs.street,
        "board": [card_str(c) for c in obs.board],
        "hole": [card_str(c) for c in obs.hole],
        "pot": obs.pot,
        "to_call": obs.to_call,
        "my_stack": obs.my_stack,
        "opp_stack": obs.opp_stack,
        "is_button": obs.is_button,
        "legal_actions": [a.name for a in Action if not (a == Action.FOLD and obs.to_call == 0)],
    }


class QueueHumanAgent:
    def __init__(self, outgoing: queue.Queue):
        self.outgoing = outgoing
        self.incoming: queue.Queue = queue.Queue()

    def act(self, obs: Observation) -> Action:
        self.outgoing.put({"type": "your_turn", "obs": obs_to_dict(obs)})
        action_name = self.incoming.get()
        return Action[action_name]


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

        result = hand.play(agents)
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
                session["human"].incoming.put(data["action"])
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
