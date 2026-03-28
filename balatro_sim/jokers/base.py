"""
base.py — Joker base class and registry.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..game import ScoreContext

JOKER_REGISTRY: dict[str, "JokerInstance"] = {}


def register_joker(key: str):
    """Decorator to register a joker effect by its game key (e.g. 'j_joker')."""
    def decorator(cls):
        JOKER_REGISTRY[key] = cls
        cls.key = key
        return cls
    return decorator


@dataclass
class ScoreContext:
    """Passed to joker trigger functions. Mutated as jokers fire."""
    chips: float = 0.0
    mult: float = 0.0
    mult_mult: float = 1.0      # Multiplicative mult (xMult jokers)
    hand_type: str = ""
    scoring_cards: list = field(default_factory=list)
    all_cards: list = field(default_factory=list)   # full hand (including non-scoring)
    jokers: list = field(default_factory=list)
    hands_left: int = 0
    discards_left: int = 0
    dollars: int = 0
    ante: int = 1
    deck_remaining: int = 0
    planet_levels: dict = field(default_factory=dict)   # hand_type -> level

    @property
    def n_jokers(self) -> int:
        return len(self.jokers)


class JokerInstance:
    """
    A joker in the player's joker slots.
    Holds the joker key, runtime state (ability.mult, extra, etc.), and edition.
    """
    def __init__(self, key: str, edition: str = "None"):
        self.key = key
        self.edition = edition
        self.state: dict = {}   # runtime state (e.g. {"mult": 0} for scaling jokers)

    def on_score_card(self, card, ctx: ScoreContext):
        """Fires for each scoring card."""
        effect = JOKER_REGISTRY.get(self.key)
        if effect and hasattr(effect, "on_score_card"):
            effect.on_score_card(self, card, ctx)

    def on_hand_scored(self, ctx: ScoreContext):
        """Fires after all scoring cards processed."""
        effect = JOKER_REGISTRY.get(self.key)
        if effect and hasattr(effect, "on_hand_scored"):
            effect.on_hand_scored(self, ctx)

    def on_discard(self, cards, ctx: ScoreContext):
        effect = JOKER_REGISTRY.get(self.key)
        if effect and hasattr(effect, "on_discard"):
            effect.on_discard(self, cards, ctx)

    def on_round_end(self, ctx: ScoreContext):
        effect = JOKER_REGISTRY.get(self.key)
        if effect and hasattr(effect, "on_round_end"):
            effect.on_round_end(self, ctx)

    def __repr__(self):
        return f"Joker({self.key}, state={self.state}, ed={self.edition})"
