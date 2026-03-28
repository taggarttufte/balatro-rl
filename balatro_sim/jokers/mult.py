"""
mult.py — Pure mult jokers (Phase 2).
TODO: implement each joker below.
"""
from .base import register_joker, JOKER_REGISTRY, ScoreContext

# ------------------------------------------------------------
# j_joker: +4 mult
# ------------------------------------------------------------
class _Joker:
    @staticmethod
    def on_hand_scored(instance, ctx: ScoreContext):
        ctx.mult += 4

JOKER_REGISTRY["j_joker"] = _Joker()

# ------------------------------------------------------------
# j_greedy_mult: +3 mult if played hand contains a Diamond card
# j_lusty_mult:  +3 mult if played hand contains a Heart card
# j_wrathful_mult: +3 mult if played hand contains a Spade card
# j_gluttonous_mult: +3 mult if played hand contains a Club card
# ------------------------------------------------------------
# TODO

# ------------------------------------------------------------
# j_zany: +12 mult if hand contains a Three of a Kind
# j_mad:  +10 mult if hand contains a Two Pair
# j_crazy: +12 mult if hand contains a Straight
# j_droll: +10 mult if hand contains a Flush
# j_sly:   +50 chips if hand contains a Pair
# j_wily:  +100 chips if hand contains Three of a Kind
# j_clever: +80 chips if hand contains Two Pair
# j_devious: +100 chips if hand contains a Straight
# j_crafty: +80 chips if hand contains a Flush
# ------------------------------------------------------------
# TODO

# ------------------------------------------------------------
# j_half: +20 mult if played hand has 3 or fewer cards
# ------------------------------------------------------------
class _HalfJoker:
    @staticmethod
    def on_hand_scored(instance, ctx: ScoreContext):
        if len(ctx.scoring_cards) <= 3:
            ctx.mult += 20

JOKER_REGISTRY["j_half"] = _HalfJoker()

# ------------------------------------------------------------
# j_abstract: +3 mult per joker owned
# ------------------------------------------------------------
class _AbstractJoker:
    @staticmethod
    def on_hand_scored(instance, ctx: ScoreContext):
        ctx.mult += 3 * ctx.n_jokers

JOKER_REGISTRY["j_abstract"] = _AbstractJoker()

# More jokers: TODO
