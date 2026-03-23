# V3 Design Notes — Shop Agent + Enhanced Communication

*Notes from 2026-03-23 planning session*

## Overview

V3 introduces a **shop agent** that makes strategic purchasing decisions, communicating with the existing play agent to build coherent strategies.

---

## Architecture

### Two-Agent System

```
┌─────────────────┐         message (16-dim latent + interpretable heads)
│   Shop Agent    │ ──────────────────────────────────────────────────────┐
│                 │                                                        │
│ shop_obs → encoder → action + message                                    │
└─────────────────┘                                                        ▼
                                                                    ┌──────────────┐
                                                                    │  Play Agent  │
                                                                    │   (V2 base)  │
                                                                    │ play_obs + message → action
                                                                    └──────────────┘
```

### Communication: Hybrid Approach

**Continuous latent (16-dim):**
- Rich, emergent communication
- Can encode nuanced mixed strategies
- Learns what's useful to communicate

**Interpretable heads (auxiliary outputs):**
- `target_hand_type` — 12-way softmax (flush, pairs, straight, etc.)
- `economy_priority` — 0.0-1.0 (how much to optimize for money)
- `build_confidence` — 0.0-1.0 (how committed to current strategy)
- `aggression` — 0.0-1.0 (risk tolerance)

**Benefits:**
- Latent handles nuance explicit heads can't express
- Explicit heads enable debugging and analysis
- Can train auxiliary losses on explicit predictions

---

## Shop Agent Action Space

### V3a — Simple (recommended starting point)

```
Actions (14):
├── buy_joker_1
├── buy_joker_2
├── buy_booster_1
├── buy_booster_2
├── buy_voucher
├── sell_joker_1..5 (5 actions)
├── reroll
└── leave_shop
```

- Auto-use consumables (heuristic)
- Auto-pick from booster packs (heuristic)

### V3b — Full (future expansion)

```
Additional actions (~30 more):
├── use_consumable_1..2
├── target_card_1..8 (for Tarot effects)
├── target_hand_type_1..12 (for Planet cards)
└── booster_pick_1..5
```

---

## Enhanced Observations (for Play Agent)

### Suit Distribution (8 features)
```python
hearts_in_hand, diamonds_in_hand, clubs_in_hand, spades_in_hand
hearts_in_deck, diamonds_in_deck, clubs_in_deck, spades_in_deck
```

### Hand Potential (6 features)
```python
cards_to_flush       # 5 - max_same_suit
cards_to_straight    # Cards needed to complete
flush_draw           # 1 if 4-flush
straight_draw        # 1 if open-ended
max_of_a_kind        # 2=pair, 3=trips, 4=quads
best_hand_score      # Estimated score of best current hand
```

### Joker Synergy Flags (~15 features)
```python
has_flush_joker, has_pair_joker, has_straight_joker
has_face_card_joker, has_suit_bonus
target_suit          # Which suit is buffed (0-3, -1 if none)
money_per_hand       # Expected $ generation per hand
money_per_discard    # Expected $ generation per discard
```

---

## Transfer Learning from V2

### Recommended Approach: Expand & Transfer

```
V2: obs(206) → [256,256,128] → actions(20)
V3: obs(206+new) → [256,256,128+new] → actions(20+shop_msg)
```

1. Copy V2 weights for overlapping parts
2. Initialize new neurons near zero
3. Train with lower LR initially (0.5x)
4. Gradually unfreeze V2 weights

---

## Advanced: Privileged Learning for Complex Strategies

### The Consumable Blocking Strategy

High-level play: Hold Tarot cards to block duplicates from packs.

**Problem:** Very delayed reward (20+ turns between hold and pack pull)

**Solution:** Privileged training observations

| Phase | Training % | Privileged Obs |
|-------|------------|----------------|
| Phase 1 | 0-70% | Full block info ("you blocked X, got Y") |
| Phase 2 | 70-90% | Annealing (gradually zero out) |
| Phase 3 | 90-100% | Human-equivalent only |

```python
if training_progress < 0.7:
    obs['block_info'] = actual_block_info
elif training_progress < 0.9:
    mask = random() > (training_progress - 0.7) / 0.2
    obs['block_info'] = actual_block_info if mask else zeros
else:
    obs['block_info'] = zeros  # Fair deployment
```

**Deployment:** Remove privileged observations for fair human comparison.

---

## Money/Economy Communication

Shop agent can signal "we need money" through:

1. **Implicit:** Buying money-generating jokers (Golden Joker, To the Moon)
2. **Explicit:** `economy_priority` interpretable head
3. **Latent:** Learned encoding in 16-dim message

Play agent learns: "When I have money jokers + high economy signal → optimize for $ generation"

---

## Training Considerations

### Shared Reward
- Both agents receive same game outcome reward
- Encourages cooperation, not competition

### Auxiliary Losses
- Predict pack outcomes from shop decisions
- Predict hand type distribution from current jokers
- Predict win probability from current state

### Preventing Communication Collapse
- Add small noise to message during training
- Information bottleneck on latent size
- Auxiliary loss: reconstruct message → useful signal

---

## Roadmap

```
V2 (current): Play agent only, auto-shop
    ↓
V3a: Simple shop agent (buy/sell/skip), transfer V2 weights
    ↓
V3b: Add consumable usage, booster selection
    ↓
V4: Enhanced play observations (deck composition, draw odds)
    ↓
V5: Privileged learning for advanced strategies (blocking, etc.)
```

---

## Key Insights

1. **Shop agent is the biggest unlock** — enables coherent build strategies
2. **Implicit communication via jokers works** — but explicit channel is richer
3. **Hybrid communication (latent + interpretable)** — best of both worlds
4. **Transfer learning preserves V2 progress** — don't start from scratch
5. **Privileged learning for complex strategies** — train with extra info, deploy without
