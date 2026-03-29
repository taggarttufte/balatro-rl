"""
scaling.py — Jokers with persistent runtime state that scales over the run.
"""
from .base import JOKER_REGISTRY, ScoreContext

# ── j_runner: gains +15 chips whenever a Straight is played ─────────────────
class _Runner:
    def on_hand_scored(self, inst, ctx):
        if ctx.hand_type in ("Straight", "Straight Flush"):
            inst.state["chips"] = inst.state.get("chips", 0) + 15
        ctx.chips += inst.state.get("chips", 0)
JOKER_REGISTRY["j_runner"] = _Runner()

# ── j_ice_cream: +100 chips, -5 chips per hand played ───────────────────────
class _IceCream:
    def on_hand_scored(self, inst, ctx):
        bonus = inst.state.get("chips", 100)
        if bonus > 0:
            ctx.chips += bonus
        inst.state["chips"] = max(0, bonus - 5)
JOKER_REGISTRY["j_ice_cream"] = _IceCream()

# ── j_green_joker: +1 mult per hand played, -1 per discard ──────────────────
class _GreenJoker:
    def on_hand_scored(self, inst, ctx):
        inst.state["mult"] = inst.state.get("mult", 0) + 1
        ctx.mult += inst.state["mult"]
    def on_discard(self, inst, cards, ctx):
        inst.state["mult"] = max(0, inst.state.get("mult", 0) - 1)
JOKER_REGISTRY["j_green_joker"] = _GreenJoker()

# ── j_ride_the_bus: +1 mult per consecutive hand without face card ───────────
class _RideTheBus:
    def on_hand_scored(self, inst, ctx):
        has_face = any(c.is_face_card and not c.debuffed for c in ctx.scoring_cards)
        if has_face:
            inst.state["mult"] = 0
        else:
            inst.state["mult"] = inst.state.get("mult", 0) + 1
        ctx.mult += inst.state.get("mult", 0)
JOKER_REGISTRY["j_ride_the_bus"] = _RideTheBus()

# ── j_green_joker dup: j_superball — TODO, different mechanic ────────────────

# ── j_supernova: +mult equal to number of times hand type played this run ────
class _Supernova:
    def on_hand_scored(self, inst, ctx):
        counts = inst.state.setdefault("hand_counts", {})
        counts[ctx.hand_type] = counts.get(ctx.hand_type, 0) + 1
        ctx.mult += counts[ctx.hand_type]
JOKER_REGISTRY["j_supernova"] = _Supernova()

# ── j_spare_trousers: +2 mult if Two Pair, scaling ───────────────────────────
class _SpareTrousers:
    def on_hand_scored(self, inst, ctx):
        if ctx.hand_type in ("Two Pair", "Full House"):
            inst.state["mult"] = inst.state.get("mult", 0) + 2
        ctx.mult += inst.state.get("mult", 0)
JOKER_REGISTRY["j_spare_trousers"] = _SpareTrousers()

# ── j_glass_joker: +2 mult per Glass card destroyed ─────────────────────────
class _GlassJoker:
    def on_hand_scored(self, inst, ctx):
        ctx.mult += inst.state.get("mult", 0)
    def on_glass_destroyed(self, inst, ctx):
        inst.state["mult"] = inst.state.get("mult", 0) + 2
JOKER_REGISTRY["j_glass_joker"] = _GlassJoker()

# ── j_lucky_cat: +0.25x mult each time a Lucky card triggers ────────────────
class _LuckyCat:
    def on_hand_scored(self, inst, ctx):
        ctx.mult_mult *= inst.state.get("mult_mult", 1.0)
    def on_lucky_trigger(self, inst, ctx):
        inst.state["mult_mult"] = inst.state.get("mult_mult", 1.0) + 0.25
JOKER_REGISTRY["j_lucky_cat"] = _LuckyCat()

# ── j_constellation: +0.1x mult per Planet card used ────────────────────────
class _Constellation:
    def on_hand_scored(self, inst, ctx):
        ctx.mult_mult *= inst.state.get("mult_mult", 1.0)
    def on_planet_used(self, inst, ctx):
        inst.state["mult_mult"] = inst.state.get("mult_mult", 1.0) + 0.1
JOKER_REGISTRY["j_constellation"] = _Constellation()

# ── j_hiker: permanent +5 chips per Planet card used ────────────────────────
class _Hiker:
    def on_hand_scored(self, inst, ctx):
        ctx.chips += inst.state.get("chips", 0)
    def on_planet_used(self, inst, ctx):
        inst.state["chips"] = inst.state.get("chips", 0) + 5
JOKER_REGISTRY["j_hiker"] = _Hiker()

# ── j_campfire: +1 mult per card sold, resets after Boss blind ───────────────
class _Campfire:
    def on_hand_scored(self, inst, ctx):
        ctx.mult += inst.state.get("mult", 0)
    def on_card_sold(self, inst, ctx):
        inst.state["mult"] = inst.state.get("mult", 0) + 1
    def on_boss_beaten(self, inst, ctx):
        inst.state["mult"] = 0
JOKER_REGISTRY["j_campfire"] = _Campfire()

# ── j_swashbuckler: +mult equal to sum of joker sell values ──────────────────
class _Swashbuckler:
    def on_hand_scored(self, inst, ctx):
        ctx.mult += inst.state.get("mult", 0)
    def on_joker_added(self, inst, sell_value, ctx):
        inst.state["mult"] = inst.state.get("mult", 0) + sell_value
JOKER_REGISTRY["j_swashbuckler"] = _Swashbuckler()

# ── j_ceremonial_dagger: +2 mult per joker destroyed to its right ─────────────
class _CeremonialDagger:
    def on_hand_scored(self, inst, ctx):
        ctx.mult += inst.state.get("mult", 0)
    def on_joker_destroyed(self, inst, ctx):
        inst.state["mult"] = inst.state.get("mult", 0) + 2
JOKER_REGISTRY["j_ceremonial_dagger"] = _CeremonialDagger()

# ── j_vampire: +0.1 mult per enhanced card played (removes enhancement) ───────
class _Vampire:
    def on_score_card(self, inst, card, ctx):
        if card.enhancement != "None" and not card.debuffed:
            inst.state["mult"] = inst.state.get("mult", 0) + 0.1
            card.enhancement = "None"  # remove enhancement
    def on_hand_scored(self, inst, ctx):
        ctx.mult += inst.state.get("mult", 0)
JOKER_REGISTRY["j_vampire"] = _Vampire()

# ── j_hit_the_road: +0.5 mult per Jack discarded this round, resets ──────────
class _HitTheRoad:
    def on_hand_scored(self, inst, ctx):
        ctx.mult_mult *= inst.state.get("mult_mult", 1.0)
    def on_discard(self, inst, cards, ctx):
        jacks = sum(1 for c in cards if c.rank == 11)
        inst.state["mult_mult"] = inst.state.get("mult_mult", 1.0) + 0.5 * jacks
    def on_round_end(self, inst, ctx):
        inst.state["mult_mult"] = 1.0
JOKER_REGISTRY["j_hit_the_road"] = _HitTheRoad()

# ── j_caino: gains +3 mult first time a 8 is played each round ───────────────
class _Caino:
    def on_score_card(self, inst, card, ctx):
        if card.rank == 8 and not card.debuffed and not inst.state.get("fired"):
            inst.state["fired"] = True
            inst.state["mult"] = inst.state.get("mult", 0) + 3
    def on_hand_scored(self, inst, ctx):
        ctx.mult += inst.state.get("mult", 0)
    def on_round_end(self, inst, ctx):
        inst.state["fired"] = False
JOKER_REGISTRY["j_caino"] = _Caino()

# ── j_yorick: gains +1 mult every 23rd card discarded ────────────────────────
class _Yorick:
    def on_hand_scored(self, inst, ctx):
        ctx.mult += inst.state.get("mult", 0)
    def on_discard(self, inst, cards, ctx):
        total = inst.state.get("total_discarded", 0) + len(cards)
        prev = inst.state.get("total_discarded", 0)
        # +1 mult for each time we cross a multiple of 23
        triggers = (total // 23) - (prev // 23)
        if triggers > 0:
            inst.state["mult"] = inst.state.get("mult", 0) + triggers
        inst.state["total_discarded"] = total
JOKER_REGISTRY["j_yorick"] = _Yorick()

# ── j_castle: +3 chips per discarded card of castle suit ─────────────────────
import random as _random
class _Castle:
    SUITS = ["Spades", "Hearts", "Clubs", "Diamonds"]
    def on_hand_scored(self, inst, ctx):
        ctx.chips += inst.state.get("chips", 0)
    def on_discard(self, inst, cards, ctx):
        suit = inst.state.get("suit") or _random.choice(self.SUITS)
        inst.state["suit"] = suit
        matching = sum(1 for c in cards if c.suit == suit)
        inst.state["chips"] = inst.state.get("chips", 0) + 3 * matching
    def on_round_end(self, inst, ctx):
        # Suit randomizes each round
        inst.state["suit"] = _random.choice(self.SUITS)
JOKER_REGISTRY["j_castle"] = _Castle()

# ── j_ticket: TODO (gains $1 when Lucky card triggers) ───────────────────────
# TODO: requires lucky card trigger hook

# ── j_trading_card: discard first card of hand for $3 each round ─────────────
# TODO: requires special discard hook (not scoring)

# ── j_riff_raff: at start of each round, create 2 common jokers ──────────────
# TODO: requires joker creation system

# ── j_marble_joker: adds Stone card to deck on Blind select ──────────────────
# TODO: requires deck modification on blind select

# ── j_faceless_joker: earn $5 if discarding 3 or more face cards ─────────────
class _FacelessJoker:
    def on_discard(self, inst, cards, ctx):
        face_count = sum(1 for c in cards if c.is_face_card)
        if face_count >= 3:
            inst.state["pending_money"] = inst.state.get("pending_money", 0) + 5
    def on_round_end(self, inst, ctx):
        inst.state["pending_money"] = 0
JOKER_REGISTRY["j_faceless_joker"] = _FacelessJoker()

# ── j_mail_in_rebate: earn $5 per discarded card of specific rank ─────────────
class _MailInRebate:
    def on_discard(self, inst, cards, ctx):
        rank = inst.state.get("rank", 2)
        matching = sum(1 for c in cards if c.rank == rank)
        inst.state["pending_money"] = inst.state.get("pending_money", 0) + 3 * matching
    def on_round_end(self, inst, ctx):
        rank = inst.state.get("rank", 2)
        inst.state["rank"] = (rank % 14) + 1 if rank < 14 else 2
JOKER_REGISTRY["j_mail"] = _MailInRebate()
