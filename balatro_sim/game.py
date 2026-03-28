"""
game.py — Top-level Balatro game state machine.

States:
  BLIND_SELECT   -> agent chooses to play or skip a blind
  SELECTING_HAND -> agent plays or discards cards
  ROUND_EVAL     -> end-of-round payout (auto-advances)
  SHOP           -> agent buys, sells, uses consumables, rerolls, then leaves
  BOOSTER_OPEN   -> agent picks from opened booster pack
  GAME_OVER      -> terminal state

Actions (passed as dict to game.step()):
  BLIND_SELECT:
    {"type": "play_blind"}
    {"type": "skip_blind"}

  SELECTING_HAND:
    {"type": "play",    "cards": [0, 2, 4]}
    {"type": "discard", "cards": [1, 3]}
    {"type": "use_consumable", "consumable_idx": 0, "target_cards": [0, 1]}

  SHOP:
    {"type": "buy",          "item_idx": 0}
    {"type": "sell_joker",   "joker_idx": 1}
    {"type": "use_consumable","consumable_idx": 0, "target_cards": [2]}
    {"type": "reroll"}
    {"type": "leave_shop"}

  BOOSTER_OPEN:
    {"type": "pick_booster", "indices": [0, 2]}  # which items to keep
    {"type": "skip_booster"}
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
from .jokers.base import JokerInstance, JOKER_REGISTRY
from .consumables import (
    apply_planet, apply_tarot, apply_spectral,
    PLANET_HAND, ALL_TAROTS, ALL_PLANETS, ALL_SPECTRALS,
    TAROT_NAME, PLANET_NAME, SPECTRAL_NAME,
)
from .shop import ShopItem, generate_shop, buy_item, sell_joker, reroll_shop


class State(Enum):
    BLIND_SELECT   = auto()
    SELECTING_HAND = auto()
    ROUND_EVAL     = auto()
    SHOP           = auto()
    BOOSTER_OPEN   = auto()
    GAME_OVER      = auto()


@dataclass
class BlindInfo:
    name: str
    kind: str           # "Small" | "Big" | "Boss"
    chips_target: int
    is_boss: bool = False
    boss_key: str = ""


@dataclass
class GameState:
    """Full observable game state snapshot."""
    state: State
    ante: int
    blind_kind: str
    chips_target: int
    chips_scored: int
    hands_left: int
    discards_left: int
    dollars: int
    hand: list[Card]
    deck_remaining: int
    jokers: list[JokerInstance]
    consumable_hand: list[str]      # list of consumable keys held
    planet_levels: dict[str, int]
    shop_items: list[ShopItem]
    hand_type: str = ""
    done: bool = False
    won: bool = False
    info: dict = field(default_factory=dict)


# Boss blind chip multipliers (applied on top of base target)
BOSS_BLINDS = [
    "bl_hook",        # discard 2 random cards on play
    "bl_goad",        # all Clubs debuffed
    "bl_window",      # all Diamonds debuffed
    "bl_manacle",     # -1 hand size
    "bl_eye",         # can't play same hand type twice
    "bl_mouth",       # can only play 1 hand type
    "bl_fish",        # draw 1 fewer card after each play
    "bl_plant",       # all face cards debuffed
    "bl_needle",      # only 1 hand
    "bl_head",        # all Hearts debuffed
    "bl_tooth",       # lose $1 per card played
    "bl_wall",        # +100 chips to target
    "bl_house",       # all ranks odd are debuffed
    "bl_mark",        # all face cards are flipped
    "bl_flint",       # base chips + mult halved
    "bl_psychic",     # must play exactly 5 cards
    "bl_grim",        # Ace appears again... (minor)
    "bl_verdant",     # -1 discard until a Spade scored
    "bl_serpent",     # after each play, discard hand and draw new one
    "bl_pillar",      # cards that have been played before score nothing
    "bl_water",       # start with 0 discards
    "bl_ox",          # playing Flush sets money to $0
    "bl_cerulean",    # first hand is decided by Blue Seal
    "bl_amber",       # your hand always = full of Twos
    "bl_violet",      # placeholder for pool size
]


class BalatroGame:
    """
    Full stateful Balatro game engine.

    Usage:
        game = BalatroGame(seed=42)
        obs = game.reset()
        while not obs.done:
            action = agent.act(obs)
            obs = game.step(action)
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self._init_game_vars()

    # ── Initialization ───────────────────────────────────────────────────────

    def _init_game_vars(self):
        """Set all mutable game state to starting values."""
        self.ante = 1
        self.blind_idx = 0                  # 0=Small, 1=Big, 2=Boss
        self.dollars = STARTING_MONEY
        self.jokers: list[JokerInstance] = []
        self.joker_slots = 5
        self.consumable_hand: list[str] = []  # held consumable keys
        self.consumable_slots = 2
        self.planet_levels: dict[str, int] = {h: 1 for h in [
            "High Card", "Pair", "Two Pair", "Three of a Kind",
            "Straight", "Flush", "Full House", "Four of a Kind",
            "Straight Flush", "Five of a Kind", "Flush House", "Flush Five",
        ]}
        self.vouchers: set[str] = set()
        self.planets_used: list[str] = []
        self.tarots_used: list[str] = []

        # Hand / discard / hand-size settings
        self.base_hands = STARTING_HANDS
        self.base_discards = STARTING_DISCARDS
        self.hand_size = HAND_SIZE

        # Shop settings
        self.shop_joker_slots = 2
        self.shop_card_slots = 2
        self.shop_discount = 0.0
        self.reroll_cost = 5
        self.reroll_discount = 0
        self.free_rerolls_per_round = 0
        self.free_rerolls_remaining = 0
        self.current_shop: list[ShopItem] = []

        # Booster state
        self.booster_choices: list = []
        self.booster_picks_remaining: int = 0

        # Blind state
        self.current_blind: BlindInfo = BlindInfo("", "Small", 0)
        self.chips_scored = 0
        self.hands_left = self.base_hands
        self.discards_left = self.base_discards
        self.hand: list[Card] = []
        self.deck: list[Card] = []
        self.played_hand_types_this_round: set[str] = set()

        self.state = State.BLIND_SELECT
        self._prepare_next_blind()

    def reset(self) -> GameState:
        self._init_game_vars()
        return self._obs()

    # ── Blind setup ──────────────────────────────────────────────────────────

    def _prepare_next_blind(self):
        """Set up current_blind without starting play yet."""
        kind = ["Small", "Big", "Boss"][self.blind_idx]
        chips = BLIND_CHIPS[self.ante][self.blind_idx]
        boss_key = ""
        if kind == "Boss":
            boss_key = self.rng.choice(BOSS_BLINDS[:20])  # use first 20 for stability
        self.current_blind = BlindInfo(
            name=f"Ante {self.ante} {kind}",
            kind=kind,
            chips_target=chips,
            is_boss=(kind == "Boss"),
            boss_key=boss_key,
        )
        self.state = State.BLIND_SELECT

    def _start_blind(self):
        """Begin playing the current blind."""
        self.chips_scored = 0
        self.hands_left = self.base_hands
        self.discards_left = self.base_discards
        self.played_hand_types_this_round = set()
        self._init_deck()
        self._draw_to_full()
        # Apply boss debuffs
        if self.current_blind.is_boss:
            self._apply_boss_start(self.current_blind.boss_key)
        # Fire blind_selected joker hooks
        for j in self.jokers:
            effect = JOKER_REGISTRY.get(j.key)
            if effect and hasattr(effect, "on_blind_selected"):
                effect.on_blind_selected(j, None)
        self.state = State.SELECTING_HAND

    def _apply_boss_start(self, boss_key: str):
        """Apply start-of-blind boss effects."""
        if boss_key == "bl_manacle":
            self.hand_size = max(1, self.hand_size - 1)
        elif boss_key == "bl_needle":
            self.hands_left = 1
        elif boss_key == "bl_water":
            self.discards_left = 0
        elif boss_key == "bl_goad":
            for c in self.deck + self.hand:
                if c.suit == "Clubs":
                    c.debuffed = True
        elif boss_key == "bl_window":
            for c in self.deck + self.hand:
                if c.suit == "Diamonds":
                    c.debuffed = True
        elif boss_key == "bl_head":
            for c in self.deck + self.hand:
                if c.suit == "Hearts":
                    c.debuffed = True
        elif boss_key == "bl_plant":
            for c in self.deck + self.hand:
                if c.is_face_card:
                    c.debuffed = True
        elif boss_key == "bl_fish":
            pass  # handled in _draw_to_full
        elif boss_key == "bl_psychic":
            pass  # enforced in _play_hand validation

    def _undo_boss_debuffs(self, boss_key: str):
        """Re-enable cards after boss blind ends."""
        if boss_key in ("bl_goad", "bl_window", "bl_head", "bl_plant"):
            for c in self.deck + self.hand:
                c.debuffed = False

    def _init_deck(self):
        self.deck = make_standard_deck()
        self.rng.shuffle(self.deck)
        self.hand = []

    def _draw_to_full(self):
        target = self.hand_size
        if self.current_blind.boss_key == "bl_fish":
            played = self.base_hands - self.hands_left
            target = max(1, self.hand_size - played)
        while len(self.hand) < target and self.deck:
            self.hand.append(self.deck.pop())

    # ── Main step ────────────────────────────────────────────────────────────

    def step(self, action: dict) -> GameState:
        atype = action.get("type", "")

        if self.state == State.BLIND_SELECT:
            if atype == "play_blind":
                self._start_blind()
            elif atype == "skip_blind":
                self._skip_blind()

        elif self.state == State.SELECTING_HAND:
            if atype == "play":
                self._play_hand(action.get("cards", []))
            elif atype == "discard":
                self._discard(action.get("cards", []))
            elif atype == "use_consumable":
                self._use_consumable(
                    action.get("consumable_idx", 0),
                    action.get("target_cards", [])
                )

        elif self.state == State.ROUND_EVAL:
            self._end_round()

        elif self.state == State.SHOP:
            if atype == "buy":
                idx = action.get("item_idx", 0)
                if idx < len(self.current_shop):
                    buy_item(self, self.current_shop[idx])
            elif atype == "sell_joker":
                sell_joker(self, action.get("joker_idx", 0))
            elif atype == "use_consumable":
                self._use_consumable(
                    action.get("consumable_idx", 0),
                    action.get("target_cards", [])
                )
            elif atype == "reroll":
                reroll_shop(self)
            elif atype == "leave_shop":
                self._end_shop()

        elif self.state == State.BOOSTER_OPEN:
            if atype == "pick_booster":
                self._pick_booster(action.get("indices", []))
            elif atype == "skip_booster":
                self.booster_choices = []
                self.state = State.SHOP

        return self._obs()

    # ── Play ─────────────────────────────────────────────────────────────────

    def _play_hand(self, card_indices: list[int]):
        selected = [self.hand[i] for i in card_indices if i < len(self.hand)]
        if not selected:
            return

        # Boss: psychic — must play exactly 5
        if self.current_blind.boss_key == "bl_psychic" and len(selected) != 5:
            return

        # Boss: hook — discard 2 random scoring cards
        if self.current_blind.boss_key == "bl_hook":
            shuffle = list(selected)
            self.rng.shuffle(shuffle)
            for c in shuffle[:2]:
                selected.remove(c)
                self.hand.remove(c)
            if not selected:
                self._draw_to_full()
                self.hands_left -= 1
                if self.hands_left <= 0:
                    self.state = State.GAME_OVER
                return

        hand_type, scoring_cards = evaluate_hand(selected)

        # Boss: eye — can't play same hand type twice
        if self.current_blind.boss_key == "bl_eye":
            if hand_type in self.played_hand_types_this_round:
                return  # illegal action

        # Boss: mouth — can only play first hand type used
        if self.current_blind.boss_key == "bl_mouth":
            if self.played_hand_types_this_round and \
               hand_type not in self.played_hand_types_this_round:
                return

        self.played_hand_types_this_round.add(hand_type)

        score, ctx = score_hand(
            scoring_cards=scoring_cards,
            all_cards=selected,
            hand_type=hand_type,
            jokers=self.jokers,
            planet_levels=self.planet_levels,
            hands_left=self.hands_left - 1,
            discards_left=self.discards_left,
            dollars=self.dollars,
            ante=self.ante,
            deck_remaining=len(self.deck),
        )

        # Boss: flint — halve chips and mult (approximate: halve score)
        if self.current_blind.boss_key == "bl_flint":
            score = score // 2

        # Boss: tooth — lose $1 per card played
        if self.current_blind.boss_key == "bl_tooth":
            self.dollars = max(0, self.dollars - len(selected))

        self.chips_scored += score
        self.hands_left -= 1

        # Apply pending side-effects from scoring
        self.dollars += ctx.pending_money
        for key in ctx.pending_consumables:
            if len(self.consumable_hand) < self.consumable_slots:
                self.consumable_hand.append(key)

        # Remove played cards from hand
        for c in selected:
            if c in self.hand:
                self.hand.remove(c)

        # Boss: serpent — discard remaining hand after play, redraw
        if self.current_blind.boss_key == "bl_serpent":
            self.hand = []

        self._draw_to_full()

        # Blue seal: add Planet card to consumable hand
        for c in scoring_cards:
            if c.seal == "Blue" and len(self.consumable_hand) < self.consumable_slots:
                hand_to_planet = {v: k for k, v in PLANET_HAND.items()}
                planet_key = hand_to_planet.get(hand_type)
                if planet_key:
                    self.consumable_hand.append(planet_key)

        # Purple seal: add Tarot card
        for c in selected:
            if c.seal == "Purple" and len(self.consumable_hand) < self.consumable_slots:
                self.consumable_hand.append(self.rng.choice(ALL_TAROTS))

        # Check win / loss
        if ctx.prevent_loss and self.chips_scored >= self.current_blind.chips_target * 0.25:
            # Mr. Bones: prevent death if >= 25% reached
            self.chips_scored = self.current_blind.chips_target
            self.state = State.ROUND_EVAL
        elif self.chips_scored >= self.current_blind.chips_target:
            self.state = State.ROUND_EVAL
        elif self.hands_left <= 0:
            self.state = State.GAME_OVER

    def _discard(self, card_indices: list[int]):
        if self.discards_left <= 0:
            return
        selected = [self.hand[i] for i in card_indices if i < len(self.hand)]
        if not selected:
            return

        # Fire on_discard joker hooks
        for j in self.jokers:
            effect = JOKER_REGISTRY.get(j.key)
            if effect and hasattr(effect, "on_discard"):
                effect.on_discard(j, selected, None)
            # Collect pending money from discard jokers
            self.dollars += j.state.pop("pending_money", 0)

        for c in selected:
            self.hand.remove(c)
        self.discards_left -= 1
        self._draw_to_full()

    def _use_consumable(self, consumable_idx: int, target_cards: list[int]):
        if consumable_idx >= len(self.consumable_hand):
            return
        key = self.consumable_hand[consumable_idx]
        success = False

        if key in PLANET_HAND:
            success = apply_planet(self, key)
        elif key in {t for t in ALL_TAROTS}:
            success = apply_tarot(self, key, target_cards)
        elif key in {s for s in ALL_SPECTRALS}:
            success = apply_spectral(self, key, target_cards)

        if success:
            self.consumable_hand.pop(consumable_idx)

    # ── Round end / shop ─────────────────────────────────────────────────────

    def _end_round(self):
        # Payout
        earnings = self.hands_left * HAND_PAYOUT
        interest = min(self.dollars // INTEREST_RATE, INTEREST_CAP)
        self.dollars += earnings + interest

        # Boss blind beaten: fire on_boss_beaten hooks
        if self.current_blind.is_boss:
            for j in self.jokers:
                effect = JOKER_REGISTRY.get(j.key)
                if effect and hasattr(effect, "on_boss_beaten"):
                    effect.on_boss_beaten(j, None)
            self._undo_boss_debuffs(self.current_blind.boss_key)

        # Fire on_round_end hooks; collect pending money
        for j in self.jokers:
            effect = JOKER_REGISTRY.get(j.key)
            if effect and hasattr(effect, "on_round_end"):
                effect.on_round_end(j, None)
            self.dollars += j.state.pop("pending_money", 0)

        # Gold seal: $3 per Gold seal card held in hand
        for c in self.hand:
            if c.seal == "Gold":
                self.dollars += 3

        # Gold enhancement: $3 per Gold card held
        for c in self.hand:
            if c.enhancement == "Gold":
                self.dollars += 3

        # Reset hand size mods from boss (bl_manacle)
        if self.current_blind.boss_key == "bl_manacle":
            self.hand_size = HAND_SIZE  # restore (voucher adjustments persist)

        # Reset reroll cost and free rerolls
        self.reroll_cost = 5
        self.free_rerolls_remaining = self.free_rerolls_per_round

        # Generate shop
        self.current_shop = generate_shop(self)
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
        self._prepare_next_blind()

    def _skip_blind(self):
        """Skip a non-Boss blind (Boss can't be skipped)."""
        if self.current_blind.kind == "Boss":
            return
        # Fire blind_skipped joker hooks
        for j in self.jokers:
            effect = JOKER_REGISTRY.get(j.key)
            if effect and hasattr(effect, "on_blind_skipped"):
                effect.on_blind_skipped(j, None)
        # Give a skip tag reward (approximate: +$5)
        self.dollars += 5
        self._end_blind_and_enter_shop()

    def _end_blind_and_enter_shop(self):
        self.reroll_cost = 5
        self.free_rerolls_remaining = self.free_rerolls_per_round
        self.current_shop = generate_shop(self)
        self.state = State.SHOP

    def _pick_booster(self, indices: list[int]):
        picks = min(self.booster_picks_remaining, len(indices))
        for idx in indices[:picks]:
            if idx < len(self.booster_choices):
                choice = self.booster_choices[idx]
                if isinstance(choice, str):
                    # Planet, tarot, spectral, or joker key
                    if len(self.consumable_hand) < self.consumable_slots:
                        self.consumable_hand.append(choice)
                    elif choice in {k for k in JOKER_REGISTRY}:
                        if len(self.jokers) < self.joker_slots:
                            self.jokers.append(JokerInstance(choice))
                elif isinstance(choice, tuple) and choice[0] == "card":
                    # Playing card from Standard pack
                    self.deck.insert(0, choice[1])
        self.booster_choices = []
        self.booster_picks_remaining = 0
        self.state = State.SHOP

    # ── Observation ──────────────────────────────────────────────────────────

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
            consumable_hand=list(self.consumable_hand),
            planet_levels=dict(self.planet_levels),
            shop_items=list(self.current_shop),
            done=(self.state == State.GAME_OVER),
            won=(self.ante > 8 and self.state == State.GAME_OVER),
            info={
                "boss_key": self.current_blind.boss_key,
                "vouchers": list(self.vouchers),
                "booster_choices": list(self.booster_choices),
            }
        )
