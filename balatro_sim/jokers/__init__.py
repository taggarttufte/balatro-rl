"""
jokers/ — Joker effect implementations.

Each joker is registered via @register_joker("key").
The registry maps joker key -> JokerEffect instance.

Trigger points:
  on_score(card, context)      — fires once per scoring card
  on_hand_scored(context)      — fires after all cards scored
  on_discard(cards, context)   — fires when player discards
  on_round_end(context)        — fires at end of round (cash out)
  on_blind_selected(context)   — fires when blind is selected

Status: STUB — implementations in submodules (mult.py, chips.py, etc.)
"""
from .base import JOKER_REGISTRY, JokerInstance, register_joker

# Import submodules to trigger registration
from . import mult, chips, scaling, hand_type, economy  # noqa: F401
