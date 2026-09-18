"""Translates poker game state into current injections for the brain's
named input channels. This mapping is invented (there is no literal "pot
odds neuron" in a real fly) in the same spirit as DOOMFLY mapping pixel
brightness to photoreceptors: an arbitrary but consistent sensory code.
"""
from __future__ import annotations

from .poker import Action, Observation, STREETS, monte_carlo_equity

INJECT_GAIN = 3.0


def encode(obs: Observation, dead_cards: list | None = None) -> dict[str, float]:
    equity = monte_carlo_equity(obs.hole, obs.board, dead_cards or [], n_opponents=1, trials=150)

    pot_after_call = obs.pot + obs.to_call
    pot_odds = obs.to_call / pot_after_call if pot_after_call > 0 else 0.0

    total_chips = obs.my_stack + obs.opp_stack
    spr = obs.my_stack / obs.pot if obs.pot > 0 else min(obs.my_stack / max(total_chips, 1), 5.0)
    spr_norm = min(spr / 10.0, 1.0)

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
        "in:equity": equity * INJECT_GAIN,
        "in:pot_odds": pot_odds * INJECT_GAIN,
        "in:spr": spr_norm * INJECT_GAIN,
        "in:street": street_norm * INJECT_GAIN,
        "in:position": position_signal * INJECT_GAIN,
        "in:opp_aggression": aggression * INJECT_GAIN,
    }, equity
