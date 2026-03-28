"""
game.py — Top-level Balatro game state machine.

States mirror Balatro's G.STATES:
  MENU -> BLIND_SELECT -> SELECTING_HAND -> (HAND_PLAYED -> DRAW_TO_HAND)* ->
  ROUND_EVAL -> SHOP -> back to BLIND_SELECT, or GAME_OVER

This is the primary interface for env_sim.py.
Phase 1: core loop, no jokers, no shop effects.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from .card import Card, make_standard_deck
from .hand_eval import evaluate_hand
from .scoring import score_hand
from .constants import (
    BLIND_CHIPS, STARTING_HANDS, STARTING_DISCARDS, HAND_SIZE,
    INTEREST_RATE, INTEREST_CAP, HAND_PAYOUT, STARTING_MONEY,
)
from .jokers.base import JokerInstance


class State(Enum):
    BLIND_SELECT  = auto()
    SELECTING_HAND = auto()
    HAND_PLAYED   = auto()
    ROUND_EVAL    = auto()
    SHOP          = auto()
    GAME_OVER     = auto()


@dataclass
class BlindInfo:
    name: str
    kind: str          # "Small" | "Big" | "Boss"
    chips_target: int
    is_boss: bool = False
    boss_key: str = ""  # e.g. "bl_hook", "bl_goad"


@dataclass
class GameState:
    """Full observable game state snapshot."""
    state: State
    ante: int
    blind_kind: str        # "Small" | "Big" | "Boss"
    chips_target: int
    chips_scored: int
    hands_left: int
    discards_left: int
    dollars: int
    hand: list[Card]       # current 8-card hand
    deck_remaining: int
    jokers: list[JokerInstance]
    planet_levels: dict[str, int]
    hand_type: str = ""
    scoring_cards: list[Card] = field(default_factory=list)
    done: bool = False
    won: bool = False


class BalatroGame:
    """
    Stateful Balatro game engine.

    Usage:
        game = BalatroGame(seed=42)
        obs = game.reset()
        while not obs.done:
            action = agent.act(obs)
            obs = game.step(action)
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self._reset_state()

    def _reset_state(self):
        self.ante = 1
        self.blind_idx = 0       # 0=Small, 1=Big, 2=Boss
        self.blind_kinds = ["Small", "Big", "Boss"]
        self.dollars = STARTING_MONEY
        self.jokers: list[JokerInstance] = []
        self.planet_levels: dict[str, int] = {h: 1 for h in [
            "High Card", "Pair", "Two Pair", "Three of a Kind",
            "Straight", "Flush", "Full House", "Four of a Kind",
            "Straight Flush", "Five of a Kind", "Flush House", "Flush Five",
        ]}
        self._start_blind()

    def _start_blind(self):
        kind = self.blind_kinds[self.blind_idx]
        chips = BLIND_CHIPS[self.ante][self.blind_idx]
        self.current_blind = BlindInfo(
            name=f"Ante {self.ante} {kind}",
            kind=kind,
            chips_target=chips,
            is_boss=(kind == "Boss"),
        )
        self.chips_scored = 0
        self.hands_left = STARTING_HANDS
        self.discards_left = STARTING_DISCARDS
        self._init_deck()
        self._draw_to_full()
        self.state = State.SELECTING_HAND

    def _init_deck(self):
        self.deck = make_standard_deck()
        self.rng.shuffle(self.deck)
        self.hand: list[Card] = []

    def _draw_to_full(self):
        while len(self.hand) < HAND_SIZE and self.deck:
            self.hand.append(self.deck.pop())

    def reset(self) -> GameState:
        self._reset_state()
        return self._obs()

    def _obs(self) -> GameState:
        return GameState(
            state=self.state,
            ante=self.ante,
            blind_kind=self.current_blind.kind,
            chips_target=self.current_blind.chips_target,
            chips_scored=self.chips_scored,
            hands_left=self.hands_left,
            discards_left=self.discards_left,
            dollars=self.dollars,
            hand=list(self.hand),
            deck_remaining=len(self.deck),
            jokers=list(self.jokers),
            planet_levels=dict(self.planet_levels),
            done=self.state == State.GAME_OVER,
            won=(self.ante > 8),
        )

    def step(self, action: dict) -> GameState:
        """
        Apply an action and advance game state.

        Action format:
            {"type": "play", "cards": [0, 2, 4]}      # play cards at indices
            {"type": "discard", "cards": [1, 3]}       # discard cards at indices
            {"type": "leave_shop"}                     # leave shop (Phase 1 only)

        Returns updated GameState.
        """
        if self.state == State.SELECTING_HAND:
            atype = action.get("type")
            cards_idx = action.get("cards", [])
            selected = [self.hand[i] for i in cards_idx if i < len(self.hand)]

            if atype == "play" and selected:
                self._play_hand(selected)
            elif atype == "discard" and selected and self.discards_left > 0:
                self._discard(selected)

        elif self.state == State.ROUND_EVAL:
            self._end_round()

        elif self.state == State.SHOP:
            # Phase 1: just leave shop immediately
            self._end_shop()

        return self._obs()

    def _play_hand(self, cards: list[Card]):
        hand_type, scoring_cards = evaluate_hand(cards)
        score = score_hand(
            scoring_cards=scoring_cards,
            all_cards=cards,
            hand_type=hand_type,
            jokers=self.jokers,
            planet_levels=self.planet_levels,
            hands_left=self.hands_left,
            discards_left=self.discards_left,
            dollars=self.dollars,
            ante=self.ante,
            deck_remaining=len(self.deck),
        )
        self.chips_scored += score
        self.hands_left -= 1

        # Remove played cards from hand
        for c in cards:
            if c in self.hand:
                self.hand.remove(c)

        self._draw_to_full()

        if self.chips_scored >= self.current_blind.chips_target:
            self.state = State.ROUND_EVAL
        elif self.hands_left <= 0:
            self.state = State.GAME_OVER
        # else stay in SELECTING_HAND

    def _discard(self, cards: list[Card]):
        for c in cards:
            if c in self.hand:
                self.hand.remove(c)
        self.discards_left -= 1
        self._draw_to_full()

    def _end_round(self):
        # Payout: $1 per hand remaining + interest
        earnings = self.hands_left * HAND_PAYOUT
        interest = min(self.dollars // INTEREST_RATE, INTEREST_CAP)
        self.dollars += earnings + interest
        self.state = State.SHOP

    def _end_shop(self):
        # Advance blind
        self.blind_idx += 1
        if self.blind_idx >= 3:
            self.blind_idx = 0
            self.ante += 1
            if self.ante > 8:
                self.state = State.GAME_OVER
                return
        self._start_blind()
