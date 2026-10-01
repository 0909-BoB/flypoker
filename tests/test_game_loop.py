import json
import os
import queue
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from flypoker.server import game_loop, validate_rebuy


def run_session(n_hands, starting_stack, big_blind, human_action):
    outgoing: queue.Queue = queue.Queue()
    session = {"human": None, "stop": False, "control": queue.Queue()}
    thread = threading.Thread(target=game_loop, args=(outgoing, session, starting_stack, big_blind), daemon=True)
    thread.start()

    log = []
    hands_done = 0
    while hands_done < n_hands:
        msg = outgoing.get(timeout=60)
        json.dumps(msg)  # what the websocket sender does -- must never raise
        log.append(msg)
        if msg["type"] == "your_turn":
            session["human"].incoming.put({"type": "action", "action": human_action})
        elif msg["type"] == "busted":
            session["control"].put({"type": "rebuy", "amount": starting_stack})
        elif msg["type"] == "hand_result":
            hands_done += 1
        elif msg["type"] == "error":
            raise AssertionError(msg["message"])
    session["stop"] = True
    return log


def test_chips_carry_over_and_conserve():
    log = run_session(10, starting_stack=60, big_blind=20, human_action="ALL_IN")
    results = [m for m in log if m["type"] == "hand_result"]
    new_hands = [m for m in log if m["type"] == "new_hand"]

    buy_in_total = 4 * 60
    for m in results:
        assert sum(m["final_stacks"].values()) == buy_in_total + 60 * sum(
            1 for r in log[: log.index(m)] if r["type"] in ("busted", "fly_rebuy")
        )
        assert all(v >= 0 for v in m["final_stacks"].values())

    # next hand starts from the previous hand's final stacks (unless a rebuy happened in between)
    for prev, nxt, start in zip(results, results[1:], new_hands[1:]):
        between = log[log.index(prev): log.index(start)]
        if any(m["type"] in ("busted", "fly_rebuy") for m in between):
            continue
        by_seat = dict(zip(start["seats"], start["stacks"]))
        assert by_seat == prev["final_stacks"]


def test_validate_rebuy():
    assert validate_rebuy("500", 20) == (500, None)
    assert validate_rebuy("5", 20)[0] is None
    assert validate_rebuy("abc", 20)[0] is None


def test_wait_for_rebuy_handles_leave_invalid_and_valid():
    from flypoker.server import wait_for_rebuy

    outgoing: queue.Queue = queue.Queue()
    session = {"stop": False, "control": queue.Queue()}

    session["control"].put({"type": "leave"})
    assert wait_for_rebuy(session, 20, outgoing) is None

    session["control"].put({"type": "rebuy", "amount": 5})      # below the big blind
    session["control"].put({"type": "rebuy", "amount": "abc"})  # not a number
    session["control"].put({"type": "rebuy", "amount": 300})
    assert wait_for_rebuy(session, 20, outgoing) == 300
    errors = [outgoing.get_nowait() for _ in range(2)]
    assert all(e["type"] == "error" for e in errors)

    session["stop"] = True  # a dropped connection must release the waiting thread
    assert wait_for_rebuy(session, 20, outgoing) is None


def _turn_obs(to_call):
    from flypoker.poker import Card, Observation
    return Observation(seat=0, hole=[Card(14, 0), Card(13, 0)], board=[], street="preflop", pot=30,
                       to_call=to_call, my_stack=1000, opp_stack=3000, is_button=False, big_blind=20)


def test_human_turn_times_out_to_fold_or_check(monkeypatch):
    import flypoker.server as srv
    from flypoker.poker import Action

    monkeypatch.setattr(srv, "TURN_TIME_LIMIT", 0.3)
    monkeypatch.setattr(srv, "TURN_ACK_GRACE", 0.2)
    for to_call, expected in ((20, Action.FOLD), (0, Action.CHECK_CALL)):
        out: queue.Queue = queue.Queue()
        agent = srv.QueueHumanAgent(out, 20, {"stop": False})
        assert agent.act(_turn_obs(to_call)) == expected
        types = [out.get_nowait()["type"] for _ in range(2)]
        assert types == ["your_turn", "turn_timeout"]


def test_late_reply_from_an_old_turn_is_ignored_and_ack_starts_the_clock(monkeypatch):
    import flypoker.server as srv
    from flypoker.poker import Action

    monkeypatch.setattr(srv, "TURN_TIME_LIMIT", 5)
    monkeypatch.setattr(srv, "TURN_ACK_GRACE", 5)
    out: queue.Queue = queue.Queue()
    agent = srv.QueueHumanAgent(out, 20, {"stop": False})
    agent.turn_id = 6  # this call becomes turn 7
    agent.incoming.put({"type": "action", "action": "ALL_IN", "turn_id": 6})  # stale: must be skipped
    agent.incoming.put({"type": "turn_ack", "turn_id": 7})
    agent.incoming.put({"type": "action", "action": "CHECK_CALL", "turn_id": 7})
    assert agent.act(_turn_obs(20)) == Action.CHECK_CALL


def test_learning_status_is_json_serializable_with_a_full_window():
    # Regression: once the recent-decision window had >= MIN_SAMPLES entries,
    # status()["healthy"] was a numpy bool, hand_result failed to serialize,
    # the websocket sender task died silently and the browser froze mid-game.
    from flypoker.learning import MIN_SAMPLES, LearningHub

    hub = LearningHub()
    for i in range(MIN_SAMPLES + 40):
        hub.recent.append((0.3 + (i % 7) / 10, i % 2 == 0, i % 5))
    status = hub.status()
    assert type(status["healthy"]) is bool
    json.dumps(status)


def test_sender_json_default_handles_numpy():
    import numpy as np
    from flypoker.server import _json_default

    out = json.dumps({"a": np.bool_(True), "b": np.float64(1.5), "c": np.arange(3)}, default=_json_default)
    assert out == '{"a": true, "b": 1.5, "c": [0, 1, 2]}'


def test_spike_events_keep_first_time_and_count_and_span_the_whole_run():
    from flypoker.server import spike_events_for_ui

    assert spike_events_for_ui([]) == []
    rec = [(5.0, 7), (9.0, 7), (2.0, 3), (140.0, 11)]
    assert spike_events_for_ui(rec) == [[3, 2, 1], [7, 5, 2], [11, 140, 1]]

    many = [(float(i % 150), i) for i in range(6000)]  # more neurons than the cap
    out = spike_events_for_ui(many, max_total=1000)
    assert len(out) == 1000
    assert out[0][1] <= 1 and out[-1][1] >= 148  # strided, not truncated to the earliest


def _san_obs(**kw):
    from flypoker.poker import Card, Observation
    base = dict(seat=0, hole=[Card(14, 0), Card(13, 0)], board=[], street="flop", pot=100, to_call=0,
                my_stack=2000, opp_stack=6000, is_button=False, big_blind=20, street_actions=[])
    base.update(kw)
    return Observation(**base)


def test_san_drops_under_pressure_and_recovers():
    from flypoker import sanity
    from flypoker.poker import Action

    calm, _ = sanity.after_decision(100.0, _san_obs(), equity=0.9)
    assert calm > 90  # nothing to worry about

    shove = _san_obs(to_call=300, my_stack=300, pot=400, street_actions=[Action.ALL_IN])
    san = 100.0
    for _ in range(4):  # a fly facing an all-in it can't comfortably call
        san, reason = sanity.after_decision(san, shove, equity=0.5)
    assert san < 40 and reason

    hurt = sanity.after_hand(san, payoff_bb=-30)
    assert 0 <= hurt < san
    assert sanity.after_hand(50.0, payoff_bb=+10) > 50
    assert sanity.label(95) == "冷靜" and sanity.label(10) == "崩潰"
