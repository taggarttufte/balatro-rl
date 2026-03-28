"""
shop.py — Shop generation, pricing, buy/sell logic.

The Balatro shop has:
  - 2 Joker slots      (main row)
  - 2 Card slots       (consumable row: planets, tarots)
  - 1 Voucher slot
  - 2 Booster pack slots

Prices:
  Common Joker:    $6    Uncommon: $7    Rare: $8    Legendary: $20
  Planet / Tarot:  $3
  Booster (std):   $4
  Voucher:         $10

Selling jokers: ~50% of buy price (rounded down), minimum $1.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .game import BalatroGame

from .consumables import (
    ALL_PLANETS, ALL_TAROTS, ALL_SPECTRALS, ALL_VOUCHERS,
    PLANET_NAME, TAROT_NAME, SPECTRAL_NAME, VOUCHER_NAME,
)

# ════════════════════════════════════════════════════════════════════════════
# JOKER CATALOGUE — all joker keys with rarity and base price
# ════════════════════════════════════════════════════════════════════════════

JOKER_CATALOGUE: dict[str, dict] = {}

def _reg(key, name, rarity, price):
    JOKER_CATALOGUE[key] = {"key": key, "name": name, "rarity": rarity, "price": price}

# Common ($6)
for k, n in [
    ("j_joker","Joker"),("j_greedy_mult","Greedy Joker"),("j_lusty_mult","Lusty Joker"),
    ("j_wrathful_mult","Wrathful Joker"),("j_gluttonous_mult","Gluttonous Joker"),
    ("j_jolly","Jolly Joker"),("j_zany","Zany Joker"),("j_mad","Mad Joker"),
    ("j_crazy","Crazy Joker"),("j_droll","Droll Joker"),("j_sly","Sly Joker"),
    ("j_wily","Wily Joker"),("j_clever","Clever Joker"),("j_devious","Devious Joker"),
    ("j_crafty","Crafty Joker"),("j_half","Half Joker"),("j_stencil","Joker Stencil"),
    ("j_four_fingers","Four Fingers"),("j_mime","Mime"),("j_credit_card","Credit Card"),
    ("j_ceremonial","Ceremonial Dagger"),("j_banner","Banner"),
    ("j_mystic_summit","Mystic Summit"),("j_marble","Marble Joker"),
    ("j_loyalty_card","Loyalty Card"),("j_8_ball","8 Ball"),("j_misprint","Misprint"),
    ("j_dusk","Dusk"),("j_raised_fist","Raised Fist"),("j_chaos","Chaos the Clown"),
    ("j_fibonacci","Fibonacci"),("j_steel_joker","Steel Joker"),
    ("j_scary_face","Scary Face"),("j_abstract","Abstract Joker"),
    ("j_delayed_grat","Delayed Gratification"),("j_hack","Hack"),
    ("j_pareidolia","Pareidolia"),("j_gros_michel","Gros Michel"),
    ("j_even_steven","Even Steven"),("j_odd_todd","Odd Todd"),
    ("j_scholar","Scholar"),("j_business_card","Business Card"),
    ("j_supernova","Supernova"),("j_ride_the_bus","Ride the Bus"),
    ("j_space_joker","Space Joker"),("j_egg","Egg"),("j_burglar","Burglar"),
    ("j_blackboard","Blackboard"),("j_runner","Runner"),("j_ice_cream","Ice Cream"),
    ("j_dna","DNA"),("j_splash","Splash"),("j_blue_joker","Blue Joker"),
    ("j_sixth_sense","Sixth Sense"),("j_constellation","Constellation"),
    ("j_hiker","Hiker"),("j_faceless","Faceless Joker"),
    ("j_green_joker","Green Joker"),("j_superposition","Superposition"),
    ("j_to_do_list","To Do List"),("j_cavendish","Cavendish"),
    ("j_card_sharp","Card Sharp"),("j_red_card","Red Card"),
    ("j_madness","Madness"),("j_square_joker","Square Joker"),
    ("j_seance","Seance"),("j_riff_raff","Riff-Raff"),
    ("j_vampire","Vampire"),("j_shortcut","Shortcut"),
    ("j_hologram","Hologram"),("j_vagabond","Vagabond"),("j_baron","Baron"),
    ("j_cloud_9","Cloud 9"),("j_rocket","Rocket"),("j_obelisk","Obelisk"),
    ("j_midas_mask","Midas Mask"),("j_luchador","Luchador"),
    ("j_gift_card","Gift Card"),("j_turtle_bean","Turtle Bean"),
    ("j_erosion","Erosion"),("j_reserved_parking","Reserved Parking"),
    ("j_flash","Flash Card"),("j_popcorn","Popcorn"),
    ("j_ramen","Ramen"),("j_walkie_talkie","Walkie Talkie"),
    ("j_seltzer","Seltzer"),("j_castle","Castle"),
    ("j_mr_bones","Mr. Bones"),("j_acrobat","Acrobat"),
    ("j_sock_and_buskin","Sock and Buskin"),("j_swashbuckler","Swashbuckler"),
    ("j_troubadour","Troubadour"),("j_certificate","Certificate"),
    ("j_smeared_joker","Smeared Joker"),("j_throwback","Throwback"),
    ("j_hanging_chad","Hanging Chad"),("j_rough_gem","Rough Gem"),
    ("j_bloodstone","Bloodstone"),("j_arrowhead","Arrowhead"),
    ("j_onyx_agate","Onyx Agate"),("j_glass_joker","Glass Joker"),
    ("j_showman","Showman"),("j_flower_pot","Flower Pot"),
    ("j_wee_joker","Wee Joker"),("j_merry_andy","Merry Andy"),
    ("j_oops","Oops! All 6s"),("j_photograph","Photograph"),
    ("j_lucky_cat","Lucky Cat"),("j_baseball","Baseball Card"),
    ("j_bull","Bull"),("j_diet_cola","Diet Cola"),
    ("j_trading_card","Trading Card"),("j_stuntman","Stuntman"),
    ("j_invisible_joker","Invisible Joker"),("j_brainstorm","Brainstorm"),
    ("j_satellite","Satellite"),("j_shoot_the_moon","Shoot the Moon"),
    ("j_drivers_license","Driver's License"),("j_cartomancer","Cartomancer"),
    ("j_astronomer","Astronomer"),("j_burnt_joker","Burnt Joker"),
]:
    _reg(k, n, "Common", 6)

# Uncommon ($7)
for k, n in [
    ("j_abstract","Abstract Joker"),("j_half","Half Joker"),
    ("j_odd_todd","Odd Todd"),("j_ancient","Ancient Joker"),
    ("j_campfire","Campfire"),("j_seeing_double","Seeing Double"),
    ("j_spare_trousers","Spare Trousers"),("j_matador","Matador"),
    ("j_hit_the_road","Hit the Road"),("j_duo","The Duo"),
    ("j_trio","The Trio"),("j_family","The Family"),
    ("j_order","The Order"),("j_tribe","The Tribe"),
]:
    _reg(k, n, "Uncommon", 7)

# Rare ($8)
for k, n in [
    ("j_blueprint","Blueprint"),("j_wee","Wee Joker"),
    ("j_the_duo","The Duo"),("j_the_trio","The Trio"),
    ("j_the_family","The Family"),("j_the_order","The Order"),
    ("j_the_tribe","The Tribe"),("j_stencil","Joker Stencil"),
    ("j_drivers_license","Driver's License"),("j_caino","Caino"),
    ("j_triboulet","Triboulet"),("j_yorick","Yorick"),
    ("j_chicot","Chicot"),("j_perkeo","Perkeo"),
    ("j_flash","Flash Card"),
]:
    _reg(k, n, "Rare", 8)

# Legendary ($20)
for k, n in [
    ("j_caino","Caino"),("j_triboulet","Triboulet"),
    ("j_yorick","Yorick"),("j_chicot","Chicot"),("j_perkeo","Perkeo"),
]:
    _reg(k, n, "Legendary", 20)

RARITY_WEIGHTS = {"Common": 70, "Uncommon": 20, "Rare": 8, "Legendary": 2}

def random_joker_key(rarity: Optional[str] = None) -> str:
    if rarity:
        pool = [k for k, v in JOKER_CATALOGUE.items() if v["rarity"] == rarity]
    else:
        keys = list(JOKER_CATALOGUE.keys())
        weights = [RARITY_WEIGHTS.get(JOKER_CATALOGUE[k]["rarity"], 10) for k in keys]
        return random.choices(keys, weights=weights, k=1)[0]
    return random.choice(pool) if pool else "j_joker"


# ════════════════════════════════════════════════════════════════════════════
# SHOP ITEM
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class ShopItem:
    kind: str          # "joker" | "planet" | "tarot" | "spectral" | "voucher" | "booster"
    key: str
    name: str
    price: int
    edition: str = "None"    # for jokers
    sold: bool = False

    def discounted_price(self, discount_frac: float) -> int:
        return max(1, int(self.price * (1 - discount_frac)))


# ════════════════════════════════════════════════════════════════════════════
# SHOP GENERATION
# ════════════════════════════════════════════════════════════════════════════

BOOSTER_CATALOGUE = {
    "p_arcana":        ("Arcana Pack",     4, "tarot",    3),   # 3 tarots, pick 1
    "p_arcana_jumbo":  ("Jumbo Arcana",    6, "tarot",    5),
    "p_arcana_mega":   ("Mega Arcana",     8, "tarot",    5),   # pick 2
    "p_celestial":     ("Celestial Pack",  4, "planet",   3),
    "p_celestial_jumbo":("Jumbo Celestial",6, "planet",   5),
    "p_celestial_mega":("Mega Celestial",  8, "planet",   5),
    "p_spectral":      ("Spectral Pack",   4, "spectral", 2),
    "p_spectral_jumbo":("Jumbo Spectral",  6, "spectral", 4),
    "p_spectral_mega": ("Mega Spectral",   8, "spectral", 4),
    "p_standard":      ("Standard Pack",   4, "card",     3),
    "p_standard_jumbo":("Jumbo Standard",  6, "card",     5),
    "p_buffoon":       ("Buffoon Pack",    4, "joker",    2),
    "p_buffoon_jumbo": ("Jumbo Buffoon",   6, "joker",    4),
}

def generate_shop(game: "BalatroGame") -> list[ShopItem]:
    """Generate a full shop for the current ante/round."""
    items: list[ShopItem] = []

    # Joker slots (2 by default)
    for _ in range(game.shop_joker_slots):
        key = random_joker_key()
        info = JOKER_CATALOGUE.get(key, {})
        edition = _roll_edition()
        price = info.get("price", 6)
        if edition != "None":
            price += _edition_markup(edition)
        items.append(ShopItem("joker", key, info.get("name", key), price, edition))

    # Card slots (2 by default: planets / tarots / spectrals)
    for _ in range(game.shop_card_slots):
        items.append(_random_consumable_item(game))

    # Voucher slot (1)
    voucher_key = _random_voucher(game)
    if voucher_key:
        items.append(ShopItem(
            "voucher", voucher_key,
            VOUCHER_NAME.get(voucher_key, voucher_key), 10
        ))

    # Booster pack slots (2)
    for _ in range(2):
        bkey = random.choice(list(BOOSTER_CATALOGUE.keys()))
        bname, bprice, _, _ = BOOSTER_CATALOGUE[bkey]
        items.append(ShopItem("booster", bkey, bname, bprice))

    return items


def _random_consumable_item(game: "BalatroGame") -> ShopItem:
    """Pick a random planet, tarot, or spectral for a card slot."""
    # Weight: planets 40%, tarots 50%, spectrals 10%
    kind = random.choices(["planet", "tarot", "spectral"], weights=[40, 50, 10])[0]
    if kind == "planet":
        key = random.choice(ALL_PLANETS)
        return ShopItem("planet", key, PLANET_NAME.get(key, key), 3)
    elif kind == "tarot":
        key = random.choice(ALL_TAROTS)
        return ShopItem("tarot", key, TAROT_NAME.get(key, key), 3)
    else:
        key = random.choice(ALL_SPECTRALS)
        return ShopItem("spectral", key, SPECTRAL_NAME.get(key, key), 4)


def _random_voucher(game: "BalatroGame") -> Optional[str]:
    available = [v for v in ALL_VOUCHERS if v not in game.vouchers]
    return random.choice(available) if available else None


def _roll_edition() -> str:
    r = random.random()
    if r < 0.003:   return "Negative"
    if r < 0.006:   return "Polychrome"
    if r < 0.02:    return "Holographic"
    if r < 0.04:    return "Foil"
    return "None"


def _edition_markup(edition: str) -> int:
    return {"Foil": 2, "Holographic": 3, "Polychrome": 5, "Negative": 5}.get(edition, 0)


# ════════════════════════════════════════════════════════════════════════════
# BUY / SELL LOGIC
# ════════════════════════════════════════════════════════════════════════════

def buy_item(game: "BalatroGame", item: ShopItem) -> bool:
    """
    Attempt to purchase a shop item. Returns True on success.
    Modifies game.dollars, game.jokers, game.consumable_hand as appropriate.
    """
    if item.sold:
        return False

    effective_price = item.discounted_price(game.shop_discount)
    if game.dollars < effective_price:
        return False

    if item.kind == "joker":
        if len(game.jokers) >= game.joker_slots:
            return False
        from .jokers.base import JokerInstance
        j = JokerInstance(item.key, item.edition)
        j.state["sell_value"] = max(1, effective_price // 2)
        game.jokers.append(j)
        game.dollars -= effective_price
        item.sold = True
        return True

    if item.kind in ("planet", "tarot", "spectral"):
        if len(game.consumable_hand) >= game.consumable_slots:
            return False
        game.consumable_hand.append(item.key)
        game.dollars -= effective_price
        item.sold = True
        return True

    if item.kind == "voucher":
        from .consumables import apply_voucher
        if apply_voucher(game, item.key):
            game.dollars -= effective_price
            item.sold = True
            return True
        return False

    if item.kind == "booster":
        game.dollars -= effective_price
        item.sold = True
        _open_booster(game, item.key)
        return True

    return False


def sell_joker(game: "BalatroGame", joker_idx: int) -> int:
    """Sell joker at index. Returns dollars gained (0 if invalid)."""
    if joker_idx < 0 or joker_idx >= len(game.jokers):
        return 0
    j = game.jokers.pop(joker_idx)
    sell_value = j.state.get("sell_value", 2)
    game.dollars += sell_value
    # Fire on_sell hooks
    effect = _get_effect(j.key)
    if effect and hasattr(effect, "on_sell"):
        effect.on_sell(j, None)
    return sell_value


def reroll_shop(game: "BalatroGame") -> bool:
    """Pay for a reroll. Returns True on success."""
    cost = max(0, game.reroll_cost - game.reroll_discount)
    if game.free_rerolls_remaining > 0:
        game.free_rerolls_remaining -= 1
        cost = 0
    if game.dollars < cost:
        return False
    game.dollars -= cost
    game.reroll_cost += 1
    game.current_shop = generate_shop(game)
    return True


# ════════════════════════════════════════════════════════════════════════════
# BOOSTER PACK OPENING
# ════════════════════════════════════════════════════════════════════════════

def _open_booster(game: "BalatroGame", booster_key: str):
    """Open a booster pack — add options to game.booster_choices for the agent to pick."""
    info = BOOSTER_CATALOGUE.get(booster_key)
    if not info:
        return
    _, _, content_kind, n_cards = info
    picks = 2 if "mega" in booster_key else 1

    choices = []
    if content_kind == "tarot":
        choices = [random.choice(ALL_TAROTS) for _ in range(n_cards)]
    elif content_kind == "planet":
        choices = [random.choice(ALL_PLANETS) for _ in range(n_cards)]
    elif content_kind == "spectral":
        choices = [random.choice(ALL_SPECTRALS) for _ in range(n_cards)]
    elif content_kind == "joker":
        choices = [random_joker_key() for _ in range(n_cards)]
    elif content_kind == "card":
        from .card import make_standard_deck
        deck = make_standard_deck()
        random.shuffle(deck)
        choices = [("card", deck[i]) for i in range(min(n_cards, len(deck)))]

    game.booster_choices = choices
    game.booster_picks_remaining = picks


def _get_effect(key: str):
    from .jokers.base import JOKER_REGISTRY
    return JOKER_REGISTRY.get(key)
