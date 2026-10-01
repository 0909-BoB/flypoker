"""Translates poker game state into current injections for the brain's
named input channels. This mapping is invented (there is no literal "pot
odds neuron" in a real fly) in the same spirit as DOOMFLY mapping pixel
brightness to photoreceptors: an arbitrary but consistent sensory code.
"""
from __future__ import annotations

from .poker import Action, Observation, STREETS, monte_carlo_equity

INJECT_GAIN = 3.0
EQUITY_GAIN = 8.0  # separate knob: equity is the channel the decision leans on most


def encode(obs: Observation, dead_cards: list | None = None) -> dict[str, float]:
    equity = monte_carlo_equity(obs.hole, obs.board, dead_cards or [],
                                 n_opponents=max(obs.active_opponents, 1), trials=150)

    pot_after_call = obs.pot + obs.to_call
    pot_odds = obs.to_call / pot_after_call if pot_after_call > 0 else 0.0

    total_chips = obs.my_stack + obs.opp_stack
    spr = obs.my_stack / obs.pot if obs.pot > 0 else min(obs.my_stack / max(total_chips, 1), 5.0)
    spr_norm = min(spr / 10.0, 1.0)

    # Stack-awareness channels: how much of my remaining chips this call
    # puts at risk (1.0 = the call is a shove), how deep I am in big blinds
    # (saturates at 100BB), and where I sit against the deepest opponent
    # still in the hand.
    commit_ratio = min(obs.to_call / obs.my_stack, 1.0) if obs.my_stack > 0 else 1.0
    depth_norm = min(obs.my_stack / max(obs.big_blind, 1) / 100.0, 1.0)
    deepest_opp = obs.max_opp_stack or obs.opp_stack
    rel_stack = obs.my_stack / max(obs.my_stack + deepest_opp, 1)

    street_norm = STREETS.index(obs.street) / (len(STREETS) - 1)

    position_signal = 1.0 if obs.is_button else 0.0

    # Which kind of action we're facing (fold/check-call/raise/all-in), not
    # a continuous read of the real bet size. An earlier version tried
    # `min(pot_odds * 1.5, 1.0)` so this channel would carry real information
    # once raises could be any amount instead of two fixed presets -- but
    # every trained decoder tested against it collapsed to an
    # input-independent verdict (identical decision regardless of equity).
    # pot_odds already carries the actual bet-size information faithfully
    # (it's untouched by this); this channel is just a coarse "how
    # aggressive was the action type" tag on top of it.
    if obs.street_actions:
        last = obs.street_actions[-1]
        aggression = {
            Action.FOLD: 0.0, Action.CHECK_CALL: 0.2,
            Action.RAISE_SMALL: 0.6, Action.RAISE_BIG: 0.85, Action.ALL_IN: 1.0,
        }[last]
    else:
        aggression = 0.0

    return {
        "in:equity": equity * EQUITY_GAIN,
        "in:pot_odds": pot_odds * INJECT_GAIN,
        "in:spr": spr_norm * INJECT_GAIN,
        "in:commit": commit_ratio * INJECT_GAIN,
        "in:depth": depth_norm * INJECT_GAIN,
        "in:rel_stack": rel_stack * INJECT_GAIN,
        "in:street": street_norm * INJECT_GAIN,
        "in:position": position_signal * INJECT_GAIN,
        "in:opp_aggression": aggression * INJECT_GAIN,
    }, equity
