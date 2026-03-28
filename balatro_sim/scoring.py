"""
scoring.py — Chip x mult scoring engine.

Scoring order (mirrors Balatro source):
  1. Base chips + base mult from hand type (adjusted for planet level)
  2. For each scoring card (in order played):
     a. Card base chips
     b. Enhancement effects (Bonus +30, Mult +4, Glass x2 mult, etc.)
     c. Edition effects on card (Foil +50 chips, Holo +10 mult, Poly x1.5 mult)
     d. Seal effects (Red seal retrigger, Blue seal planet, etc.)
     e. Each joker fires on_score_card(card, ctx)
  3. After all cards: each joker fires on_hand_scored(ctx)
  4. Final score = (chips + ctx.chips) * (mult + ctx.mult) * ctx.mult_mult
     (clamped to int)
"""
from .card import Card
from .constants import HAND_BASE, HAND_LEVEL_CHIPS, HAND_LEVEL_MULT
from .jokers.base import JokerInstance, ScoreContext


def score_hand(
    scoring_cards: list[Card],
    all_cards: list[Card],
    hand_type: str,
    jokers: list[JokerInstance],
    planet_levels: dict[str, int],
    hands_left: int,
    discards_left: int,
    dollars: int,
    ante: int,
    deck_remaining: int,
) -> int:
    """
    Compute the total score for a played hand.

    Args:
        scoring_cards: Cards that contribute to the hand type.
        all_cards:     All played cards (including non-scoring ones like kickers).
        hand_type:     The evaluated hand type string.
        jokers:        Player's joker slots (ordered).
        planet_levels: Dict of hand_type -> current level (0-indexed, level 1 = +0 bonus).
        ...

    Returns:
        Integer score (chips * mult, floored).
    """
    base_chips, base_mult = HAND_BASE.get(hand_type, (5, 1))

    # Apply planet card level bonuses
    level = planet_levels.get(hand_type, 1)
    if level > 1:
        base_chips += HAND_LEVEL_CHIPS.get(hand_type, 0) * (level - 1)
        base_mult  += HAND_LEVEL_MULT.get(hand_type, 0)  * (level - 1)

    ctx = ScoreContext(
        chips=0.0,
        mult=0.0,
        mult_mult=1.0,
        hand_type=hand_type,
        scoring_cards=scoring_cards,
        all_cards=all_cards,
        jokers=jokers,
        hands_left=hands_left,
        discards_left=discards_left,
        dollars=dollars,
        ante=ante,
        deck_remaining=deck_remaining,
        planet_levels=planet_levels,
    )

    # Score each card
    for card in scoring_cards:
        if card.debuffed:
            continue

        # Card base chips
        ctx.chips += card.base_chips

        # Enhancement effects
        if card.enhancement == "Bonus":
            ctx.chips += 30
        elif card.enhancement == "Mult":
            ctx.mult += 4
        elif card.enhancement == "Glass":
            ctx.mult_mult *= 2.0
        elif card.enhancement == "Lucky":
            # TODO: Lucky card (+20 mult on trigger, $20 on trigger — probabilistic)
            pass
        elif card.enhancement == "Steel":
            # Steel: +0.5x mult while in hand (handled elsewhere)
            pass

        # Edition effects on card
        if card.edition == "Foil":
            ctx.chips += 50
        elif card.edition == "Holographic":
            ctx.mult += 10
        elif card.edition == "Polychrome":
            ctx.mult_mult *= 1.5

        # Jokers: on_score_card
        for joker in jokers:
            joker.on_score_card(card, ctx)

        # Red seal: retrigger (score card again)
        if card.seal == "Red":
            ctx.chips += card.base_chips
            if card.enhancement == "Bonus":
                ctx.chips += 30
            elif card.enhancement == "Mult":
                ctx.mult += 4
            for joker in jokers:
                joker.on_score_card(card, ctx)

    # Jokers: on_hand_scored (fires after all cards)
    for joker in jokers:
        joker.on_hand_scored(ctx)

    # Joker editions (fire after joker effects)
    for joker in jokers:
        if joker.edition == "Foil":
            ctx.chips += 50
        elif joker.edition == "Holographic":
            ctx.mult += 10
        elif joker.edition == "Polychrome":
            ctx.mult_mult *= 1.5

    total_chips = base_chips + ctx.chips
    total_mult  = (base_mult + ctx.mult) * ctx.mult_mult
    return int(total_chips * total_mult)
