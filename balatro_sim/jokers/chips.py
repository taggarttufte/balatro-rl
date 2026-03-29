"""
chips.py — Additive chip jokers.
"""
from .base import JOKER_REGISTRY, ScoreContext

# ── Hand-type chip jokers ─────────────────────────────────────────────────────

def _has_hand(hand_type, *targets):
    return hand_type in targets

class _HtChips:
    def __init__(self, targets, bonus):
        self.targets = set(targets)
        self.bonus = bonus
    def on_hand_scored(self, inst, ctx):
        if ctx.hand_type in self.targets:
            ctx.chips += self.bonus

# j_sly: +50 chips if Pair
JOKER_REGISTRY["j_sly"] = _HtChips(
    {"Pair", "Two Pair", "Full House", "Four of a Kind", "Five of a Kind", "Flush House"}, 50)

# j_wily: +100 chips if Three of a Kind
JOKER_REGISTRY["j_wily"] = _HtChips(
    {"Three of a Kind", "Full House", "Five of a Kind", "Flush House"}, 100)

# j_clever: +80 chips if Two Pair
JOKER_REGISTRY["j_clever"] = _HtChips(
    {"Two Pair", "Full House"}, 80)

# j_devious: +100 chips if Straight
JOKER_REGISTRY["j_devious"] = _HtChips(
    {"Straight", "Straight Flush"}, 100)

# j_crafty: +80 chips if Flush
JOKER_REGISTRY["j_crafty"] = _HtChips(
    {"Flush", "Straight Flush", "Flush House", "Flush Five"}, 80)


# ── Suit chip jokers ──────────────────────────────────────────────────────────

class _SuitChips:
    def __init__(self, suit, bonus):
        self.suit = suit
        self.bonus = bonus
    def on_score_card(self, inst, card, ctx):
        if card.suit == self.suit and not card.debuffed:
            ctx.chips += self.bonus

# j_arrowhead: +50 chips per Spade scored
JOKER_REGISTRY["j_arrowhead"] = _SuitChips("Spades", 50)

# j_rough_gem: +30 chips per Diamond scored
JOKER_REGISTRY["j_rough_gem"] = _SuitChips("Diamonds", 30)

# j_onyx_agate: +7 chips per Club scored
JOKER_REGISTRY["j_onyx_agate"] = _SuitChips("Clubs", 7)


# ── j_blue_joker: +2 chips per card remaining in deck ───────────────────────
class _BlueJoker:
    def on_hand_scored(self, inst, ctx):
        ctx.chips += 2 * ctx.deck_remaining
JOKER_REGISTRY["j_blue_joker"] = _BlueJoker()


# ── j_stone: +25 chips per Stone card in full deck ───────────────────────────
# TODO: needs full deck access; approximate with a counter in state
class _StoneJoker:
    def on_hand_scored(self, inst, ctx):
        n_stones = inst.state.get("stone_count", 0)
        ctx.chips += 25 * n_stones
JOKER_REGISTRY["j_stone"] = _StoneJoker()


# ── j_wee_joker: already in mult.py but belongs here semantically; kept there ─

# ── j_banner: also chips – defined in mult.py since it was already there ───────

# ── j_superposition: TODO ─────────────────────────────────────────────────────
# +1 Tarot card if played hand has Straight + Ace scored. Requires consumable system.

# ── j_8_ball: handled in mult.py (TODO) ──────────────────────────────────────

# ── j_burglar: +3 hands, -1 discard per round ────────────────────────────────
# TODO: game state modification (not scoring)

# ── j_egg: gains $3 of sell value each round (scaling economy) ───────────────
class _Egg:
    def on_round_end(self, inst, ctx):
        inst.state["sell_bonus"] = inst.state.get("sell_bonus", 0) + 3
JOKER_REGISTRY["j_egg"] = _Egg()


# ── j_erosion: +4 chips for each card below starting deck size ───────────────
class _Erosion:
    STARTING_DECK = 52
    def on_hand_scored(self, inst, ctx):
        current = ctx.deck_remaining + len(ctx.all_cards)  # approx
        below = max(0, self.STARTING_DECK - current)
        ctx.chips += 4 * below
JOKER_REGISTRY["j_erosion"] = _Erosion()


# ── j_flash_card: gains +2 chips each shop reroll ────────────────────────────
class _FlashCard:
    def on_hand_scored(self, inst, ctx):
        ctx.chips += inst.state.get("bonus_chips", 0)
    def on_reroll(self, inst, ctx):
        inst.state["bonus_chips"] = inst.state.get("bonus_chips", 0) + 2
JOKER_REGISTRY["j_flash_card"] = _FlashCard()


# ── j_popcorn: +20 mult, -4 each round (vanishes at 0) ──────────────────────
class _Popcorn:
    def on_hand_scored(self, inst, ctx):
        bonus = inst.state.get("mult", 20)
        if bonus > 0:
            ctx.mult += bonus
    def on_round_end(self, inst, ctx):
        inst.state["mult"] = max(0, inst.state.get("mult", 20) - 4)
JOKER_REGISTRY["j_popcorn"] = _Popcorn()


# ── j_ramen: x2 Mult, loses -0.01 Mult per card discarded ───────────────────
class _Ramen:
    def on_hand_scored(self, inst, ctx):
        factor = inst.state.get("mult_mult", 2.0)
        ctx.mult_mult *= factor
    def on_discard(self, inst, cards, ctx):
        inst.state["mult_mult"] = max(0.0, inst.state.get("mult_mult", 2.0) - 0.01 * len(cards))
JOKER_REGISTRY["j_ramen"] = _Ramen()
