# Balatro RL

A reinforcement learning agent that learns to play [Balatro](https://www.playbalatro.com/) using PPO.

The project went through five major versions: from a simple Lua mod with file-based IPC, through socket-based parallelism, to a complete Python simulation — and now a **dual-agent architecture** where separate shop and play agents cooperate to learn Balatro end-to-end.

> **Disclaimer:** Balatro is a product of LocalThunk/Playstack. This is an unofficial mod for research and educational purposes only.

---

## What is Balatro?

Balatro is a roguelike poker deckbuilder. You play 5-card hands to score chips against escalating point thresholds ("blinds"), buying joker modifiers between rounds to multiply your scoring.

**Why it's a hard RL problem:**

| Challenge | Description |
|-----------|-------------|
| Combinatorial action space | 8 cards → 218 possible play/discard combinations per turn |
| Long-horizon credit assignment | A joker bought in ante 1 might only pay off in ante 6 |
| Partial observability | Upcoming blinds, deck order, future shop contents all unknown |
| Sparse rewards | Only hands that clear blinds produce meaningful reward |
| Stochastic elements | Card draws, joker spawns, boss blind effects are randomized |
| Compounding strategy | Hand upgrades, deck thinning, joker combos interact multiplicatively |

A skilled human wins ~70% of runs. Random play wins <0.01%.

---

## Current Version: V5 — Dual-Agent Architecture

### Why dual agents

V4 proved the Python sim works: 4,000 steps/sec, correct scoring, 0.47% win rate with
a single policy. But a single agent treating play and shop as one flat action space
can't effectively learn economy — shop steps are ~5% of total transitions, so the agent
converges to "leave shop immediately" before getting enough signal to connect purchases
to downstream wins.

**The core insight:** play decisions (which cards to use) and shop decisions (which jokers
to buy) require fundamentally different reasoning. Splitting into two cooperating agents
lets each specialize, with a learned communication channel between them.

### V5 Architecture

```
SHOP PHASE                           BLIND PHASE
+------------------+                +------------------+
|   Shop Agent     |                |   Play Agent     |
|   6-layer resnet |                |   6-layer resnet |
|   ~2.3M params   |                |   ~2.3M params   |
+--------+---------+                +--------+---------+
         |                                   |
         |  32-dim communication vector      |
         +---------> concat to obs ----------+
                     (stop-gradient)
```

- **Play agent:** 374-dim obs (342 game features + 32 comm vector), Discrete(46) actions
- **Shop agent:** 188-dim obs (shop-specific features), Discrete(17) actions with hierarchical
  pack sub-states (PACK_OPEN, PACK_TARGET)
- **Communication:** Shop agent produces a 32-dim vector at end of each shop phase, carried
  as fixed context through all hands of the next blind. Stop-gradient prevents co-adaptation.

### Simulation

The Python sim (`balatro_sim/`) reimplements Balatro's full ruleset:

| Module | Contents |
|--------|----------|
| `game.py` | Full game loop — BLIND_SELECT, SELECTING_HAND, SHOP, ROUND_EVAL states; 15 boss blind effects; ante progression; win condition |
| `scoring.py` | Chip/mult computation, card enhancements, editions, seals, retriggers |
| `hand_eval.py` | All 12 hand types including Flush Five / Flush House |
| `jokers/` | **164 jokers** implemented across 6 modules |
| `consumables.py` | 12 planets, 22 tarots, 18 spectrals, 27 vouchers |
| `shop.py` | Weighted rarity rolls, buy/sell/reroll, full 150+ item catalogue |
| `env_v5.py` | V5 dual-agent env — routes phases to agents, hierarchical shop actions |
| `quality.py` | Loadout quality estimator (rarity + synergies + planet levels) |

### Observation Spaces

**Play agent (374 features):**

| Range | Description |
|-------|-------------|
| `[0:40]` | Hand cards — 8 slots × 5 features (rank, suit, enhancement, edition, seal) |
| `[40:100]` | Play combos — 20 slots × 3 features (hand_type_id, score_estimate, num_cards) |
| `[100:108]` | Discard options — 8 slots |
| `[108:174]` | Jokers — 5 slots × (type_id + state features) |
| `[174:198]` | Consumables, vouchers |
| `[198:210]` | Scalar state — ante, blind_idx, score_progress, hands_left, discards_left, dollars, etc. |
| `[210:342]` | Deck composition, hand type levels |
| `[342:374]` | Communication vector from shop agent |

**Shop agent (188 features):**

| Range | Description |
|-------|-------------|
| `[0:10]` | Game scalars — ante, round, dollars, interest, hands_left, reroll_cost, joker slots |
| `[10:60]` | Joker loadout — 5 slots × 10 features |
| `[60:90]` | Shop items — 6 slots × 5 features |
| `[90:100]` | Booster packs — 2 slots × 5 features |
| `[100:112]` | Planet/hand levels — 12 hand types |
| `[112:120]` | Consumable slots |
| `[120:154]` | Boss blind one-hot, voucher flags, deck composition |
| `[154:188]` | Card enhancement/seal counts, deck size |

### Action Spaces

**Play agent — `Discrete(46)` (same as V4):**
```
[0-19]   play combo (sorted by score_hand() descending — best hand always action 0)
[20-27]  discard card at position i
[28-29]  use consumable
[30]     play blind
[31]     skip blind (non-boss only)
[32-38]  buy shop item
[39-43]  sell joker
[44]     reroll shop
[45]     leave shop
```

**Shop agent — hierarchical:**
```
Normal shop:  Discrete(17)
  0      reroll
  1      leave shop
  2-7    buy shop item 0-5
  8-9    buy booster pack 0-1
  10-14  sell joker 0-4
  15-16  use consumable 0-1

Pack open:    Discrete(N+1)  — pick from revealed cards or skip
Pack target:  Discrete(53)   — apply tarot to deck card or skip
```

### Reward Structure

| Signal | Value |
|--------|-------|
| Blind cleared | +2.0 × (9 - ante) (inverse-ante: early progress worth more) |
| Ante completed | +5.0 |
| Game won (past ante 8) | +50.0 |
| Game lost | -2.0 |
| Score progress (log-scaled) | `0.05 × log(1 + delta) × 100` |
| Leave shop penalty | -0.1 (prevents instant-leave collapse) |
| Loadout quality delta | +0.5 × quality improvement (auxiliary shop reward) |
| Spending reward | +0.05 × dollars spent (encourages purchasing) |

### Network Architecture

Both agents use 6-layer residual networks:
```
input → embed (512) → 4 × ResidualBlock(512) → actor head / critic head

ResidualBlock: x + ReLU(fc2(ReLU(fc1(x)))), with LayerNorm
```

Shop agent adds a 32-dim linear communication head on the shared trunk.

### Asymmetric Data Collection

Shop steps are naturally rare (~5% of transitions). V5 uses asymmetric collection:
workers keep running until both conditions are met:
1. At least `steps_target` total steps collected
2. At least `min_shop_steps` shop steps collected

Force-resets episodes after 200 play steps without a shop visit to prevent shop drought.
Hard cap at 16× target prevents runaway collection.

---

## V4 — Python Simulation

V4 replaced live Balatro instances with a pure Python simulation:
- **~4,000 steps/sec** using 16 multiprocessing workers (vs ~14 sps in V3)
- Zero RAM degradation (no Lua runtime)
- Single-agent PPO with flat Discrete(46) action space

See [V4 design notes](results/V4_DESIGN_NOTES.md) for full details.

---

## Training Runs

### V4 Runs (single agent, `train_sim.py`)

| Run | Batch | Network | Params | Iters | Notes |
|-----|-------|---------|--------|-------|-------|
| Run 1 | 4,096 | 3-layer MLP | 345k | 1,000 | Reward unbounded (linear delta) — spikes to 82M |
| Run 2 | 4,096 | 3-layer MLP | 345k | ~180 | Killed — same reward issue |
| Run 3 | 4,096 | 3-layer MLP | 345k | 1,000 | Log1p reward fix; score-sorted combos |
| Run 4 | 32,768 | 3-layer MLP | 345k | 1,000 | Large batch comparison vs run 3 |
| Run 5 | 8,192 | 6-layer residual | 2.3M | 1,000 | Deeper architecture |
| Run 6 | 8,192 | 6-layer residual | 2.3M | 2,000 | Inverse-ante rewards, best V4 result |

**Best V4 result (run 3):** 0.47% win rate, ante 9 by iter 3, reward 1.7 → 48.

### V5 Runs (dual agent, `train_v5.py`)

| Run | Config | Play Init | Shop Init | Iters | Result |
|-----|--------|-----------|-----------|-------|--------|
| Run 1 | Phase A (frozen play) | Run 6 | Scratch | 500 | Shop starved — 0 shop steps by iter 31 |
| Run 2 | Phase B (joint) | Run 6 | Scratch | — | NaN crash — embed shape mismatch |
| Run 3 | Phase B | Run 6 | Scratch | ~1484 | NaN crash — advantage normalization |
| Run 4 | Phase B | Run 6 | Scratch | 1,000 | Play entropy collapsed to 0.002 |
| Run 5 | Phase B | Run 6 | Scratch | ~1106 | NaN crash with 4000-step truncation |
| Run 6 | Phase B | Run 6 | Scratch | 1,000 | Play entropy collapse despite 0.05 coeff |
| Run 7 | Phase B | Run 6 | Scratch | ~1342 | First healthy run, reward -1.5 → +0.35 |
| Run 8 | Phase B | Scratch | Scratch | 841 | Both from scratch — shop steps healthy early, starved later |
| Run 9 | Phase B | Scratch | Scratch | ongoing | Leave penalty + spend reward, shop exploring |

**Key findings:**
- **Pretrained play agent = worse shop starvation.** Better play → longer episodes → fewer
  shop resets per batch. Both-from-scratch gives better early shop exposure.
- **Shop starvation is structural.** Play/shop step ratio is ~20:1. Solved with asymmetric
  collection (min shop steps per iteration) and force-resets during play droughts.
- **Leave bias is real.** Leaving shop is always safe (zero cost). Without a leave penalty
  (-0.1) and spending reward, shop agent converges to "leave immediately" in ~40 iters.
- **Credit assignment is the core challenge.** A joker bought at step 50 may not pay off
  until step 400+. Quality estimator provides intermediate reward signal.

---

## What We Learned

### V5 Lessons (Dual-Agent Training)

**Pretrained agents cause data starvation for the minority phase.** When the play agent
starts competent (from V4 run 6), it survives longer → fewer episode resets → fewer shop
transitions per batch. Runs 1-6 all hit this: shop steps collapsed to near zero. Starting
both agents from scratch (run 8+) gives much better early shop exposure.

**Shop starvation is structural, not a hyperparameter problem.** The game's natural play/shop
ratio is ~20:1. No amount of tuning batch size or truncation fixes this. The solution is
asymmetric collection: guarantee a minimum number of shop steps per iteration, and force-reset
episodes during long play-only streaks.

**Leave bias requires explicit counter-incentives.** Leaving the shop is always safe (zero
immediate cost). Without intervention, the shop agent learns "leave immediately" within
40 iterations. A leave penalty (-0.1), quality-delta reward, and spending reward together
keep the shop agent exploring.

**NaN hardening is non-negotiable for dual-agent PPO.** With two agents and imbalanced data,
edge cases compound: all-False mask rows, empty rollout normalization, extreme advantages
from sparse shop rewards. Every path through the PPO update needs nan_to_num guards.

### V4 Lessons (Python Sim Training)

**Pre-ranking actions matters enormously.** Switching to actual `score_hand()` ordering
(accounting for jokers, planet levels, enhancements) caused ante 9 by **iteration 2**
instead of iteration 460.

**Unbounded rewards destabilize PPO.** Linear score progress hit 82M reward in a single
episode. Log scaling fixed it immediately.

**Winning joker builds are learnable.** The agent discovered the Green Joker + Burglar +
Space Joker synergy (~28% of wins) without being told it exists.

### V3 Lessons (Live Game Training)

**RAM degradation is fatal for long runs.** 8 Balatro instances ballooned to 23GB after 4-6
hours. The Lua GC couldn't keep up at 128x speed. Python sim eliminated this entirely.

**Socket IPC > file IPC.** 3.31x throughput improvement and much richer episodes (20 → 90
steps) from fewer dropped actions.

---

## Throughput Comparison

| Version | Method | Parallelism | Steps/sec | vs V1 |
|---------|--------|-------------|-----------|-------|
| V1 | File IPC, SB3 | 1 env | ~0.08 | 1× |
| V2 single | File IPC, SB3 | 1 env | ~6 | 75× |
| V2 parallel | File IPC, Ray | 8 envs | ~6 | 75× |
| V3 | Socket IPC, custom PPO | 8 envs | ~14* | 175× |
| **V4** | **Python sim, multiprocessing** | **16 workers** | **~4,000** | **50,000×** |

*V3 degraded from ~39 sps to ~14 sps median over training due to RAM issues.

**V4 bottleneck:** Pure Python game simulation. CPU (Ryzen 9 7950X, 16 cores) is fully
utilized at 16 workers. A C extension or JAX-based sim would yield 50-200× further
improvement.

---

## Repository Layout

```
balatro-rl/
├── train_v5.py               # V5 training — dual-agent PPO (current)
├── train_sim.py              # V4 training — single-agent PPO
├── balatro_sim/              # Python Balatro simulation
│   ├── game.py               # Full game loop (BalatroGame class)
│   ├── env_v5.py             # V5 dual-agent env (play + shop routing)
│   ├── env_sim.py            # V4 single-agent env
│   ├── quality.py            # Loadout quality estimator
│   ├── scoring.py            # Chip/mult scoring engine
│   ├── hand_eval.py          # Hand type evaluation
│   ├── consumables.py        # Planets, tarots, spectrals, vouchers
│   ├── shop.py               # Shop system
│   ├── card.py               # Card representation
│   ├── constants.py          # Game constants
│   └── jokers/               # 164 jokers across 6 modules
│       ├── mult.py, chips.py, scaling.py
│       ├── hand_type.py, economy.py, misc.py
│       └── tests/            # 120 tests, all passing
├── tests/                    # V5 environment tests
│   └── test_shop_v5.py       # Shop action masking, buying/selling, packs, comm vector
├── scripts/                  # Plotting and analysis
│   ├── plot_runs.py          # Multi-run comparison charts
│   ├── plot_v5_run8.py       # V5 run 8 training dashboard
│   ├── plot_v4.py            # V4 training dashboard
│   └── plot_training.py      # V2/V3/V4 comparison
├── results/                  # Design notes and historical results
│   ├── V5_DESIGN_NOTES.md    # Dual-agent architecture spec
│   ├── V5_RUN_LOG.md         # V5 training run journal
│   ├── V4_DESIGN_NOTES.md, V3_DESIGN_NOTES.md
│   └── V1_RESULTS.md, V2_*.md, NOTES.md
├── legacy/
│   ├── training/             # V1-V3 training scripts
│   └── tests/                # V1-V3 test and analysis scripts
└── launch/                   # PowerShell scripts for V3 instance management
```

---

## Setup

```bash
git clone https://github.com/taggarttufte/balatro-rl
cd balatro-rl
python -m venv env && source env/bin/activate  # or env\Scripts\activate on Windows
pip install torch numpy gymnasium
```

```bash
# Run tests
python -m pytest balatro_sim/tests/ tests/ -v

# Train V5 (dual-agent, default: 16 workers, 8k batch, 1000 iters)
python train_v5.py --training-phase B --workers 16 --steps 8192 --iterations 1000

# With asymmetric shop collection (guarantee 200 shop steps/iter)
python train_v5.py --training-phase B --min-shop-steps 200

# Resume from checkpoints
python train_v5.py --resume-play checkpoints_v5/iter_0500_play.pt \
                   --resume-shop checkpoints_v5/iter_0500_shop.pt

# Train V4 (single-agent, for comparison)
python train_sim.py --workers 16 --steps-per-worker 512 --iterations 1000
```

---

## Acknowledgments

Built on [Steamodded](https://github.com/Steamopollys/Steamodded) for Lua mod loading
and [Handy](https://github.com/nicholassam6425/balatro-mods) for speed control (V3).
