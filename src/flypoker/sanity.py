"""A fly's SAN value (0-100, like a sanity meter): how composed it is.

It starts at 100 and is knocked down by the same pressures the fly's brain
receives as inputs, then by chips lost, and recovers slowly. Display-only --
it does not feed back into the fly's decisions.

Per decision, `stress` (0..1) is a weighted sum of:
    opponent aggression  0.25   how hard the last action on this street was
    commitment           0.25   share of my remaining stack the call costs
    short stack          0.15   fewer than ~25 big blinds behind
    pot odds burden      0.15   price of the call relative to the pot
    doubt                0.20   how close to a coin flip my equity is
                                (matters mostly when facing a bet)
SAN eases toward 100 * (1 - stress); after each hand, losses subtract more,
wins add a little, and it drifts back toward 100.
"""
from __future__ import annotations

from .poker import Action, Observation

_AGGRESSION = {
    Action.FOLD: 0.0, Action.CHECK_CALL: 0.2, Action.RAISE_SMALL: 0.6,
    Action.RAISE_BIG: 0.85, Action.ALL_IN: 1.0,
}
REASONS = {
    "pressure": "對手施壓", "commit": "跟注佔籌碼高", "short": "短碼",
    "odds": "底池賠率", "doubt": "勝負難分",
}
SAN_START = 100.0
EASE = 0.45  # how fast SAN moves toward the per-decision target


def decision_stress(obs: Observation, equity: float) -> tuple[float, dict[str, float]]:
    aggression = _AGGRESSION[obs.street_actions[-1]] if obs.street_actions else 0.0
    facing = obs.to_call > 0
    commit = min(obs.to_call / obs.my_stack, 1.0) if obs.my_stack > 0 else 1.0
    depth = obs.my_stack / max(obs.big_blind, 1)
    short = min(max(1.0 - depth / 25.0, 0.0), 1.0)
    pot_after = obs.pot + obs.to_call
    odds = obs.to_call / pot_after if pot_after > 0 else 0.0
    doubt = (1.0 - abs(2.0 * equity - 1.0)) * (1.0 if facing else 0.4)
    parts = {
        "pressure": 0.25 * aggression, "commit": 0.25 * commit, "short": 0.15 * short,
        "odds": 0.15 * odds, "doubt": 0.20 * doubt,
    }
    return min(sum(parts.values()), 1.0), parts


def after_decision(san: float, obs: Observation, equity: float) -> tuple[float, str]:
    """New SAN after one decision, plus the main reason if it's noticeably
    stressed (empty string when calm)."""
    stress, parts = decision_stress(obs, equity)
    target = 100.0 * (1.0 - stress)
    new = san + (target - san) * EASE
    new = max(0.0, min(100.0, new))
    top, weight = max(parts.items(), key=lambda kv: kv[1])
    return new, (REASONS[top] if weight >= 0.06 else "")


def after_hand(san: float, payoff_bb: float) -> float:
    if payoff_bb < 0:
        san -= min(20.0, -payoff_bb * 1.5)
    else:
        san += min(8.0, payoff_bb * 0.6)
    san += (100.0 - san) * 0.12  # slow recovery between hands
    return max(0.0, min(100.0, san))


def label(san: float) -> str:
    return "冷靜" if san >= 80 else "緊張" if san >= 55 else "動搖" if san >= 30 else "崩潰"
