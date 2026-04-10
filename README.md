# Balatro RL

A reinforcement learning agent that learns to play [Balatro](https://www.playbalatro.com/) using PPO.

Six versions over four months: from a Lua mod with file-based IPC, through socket parallelism and a failed dual-agent experiment, to a complete Python simulation with a single-agent PPO reaching **1.9% win rate** on random seeds with a fully audited simulator. V7 (strategic card selection) is next.

> **Disclaimer:** Balatro is a product of LocalThunk/Playstack. This is an unofficial project for research and educational purposes only.

---

## What is Balatro?

Balatro is a roguelike poker deckbuilder. You play poker hands to score chips against escalating point thresholds ("blinds"), buying joker modifiers between rounds to multiply your scoring. Clear 8 antes (24 blinds) to win.

**Why it's a hard RL problem:**

| Challenge | Description |
|-----------|-------------|
| Combinatorial action space | 8 cards in hand -> 218 possible play/discard subsets per turn |
| Long-horizon credit assignment | A joker bought in ante 1 might only pay off in ante 6 |
| Partial observability | Deck order, future shop contents, upcoming blinds all unknown |
| Sparse rewards | Only hands that clear blinds produce meaningful reward |
| Stochastic elements | Card draws, joker spawns, boss blind effects randomized each game |
| Compounding strategy | Hand upgrades, deck thinning, joker combos interact multiplicatively |

A skilled human wins ~70% of runs. Random play wins <0.01%.

---

## Version History

| Version | Method | Architecture | Peak Win Rate | Key Learning |
|---------|--------|-------------|:---:|-------------|
| V1 | File IPC, SB3 | 1 Balatro instance | <0.01% | Proof of concept |
| V2 | File IPC, Ray | 8 Balatro instances | <0.01% | Parallelism works but IPC is the bottleneck |
| V3 | Socket IPC, custom PPO | 8 Balatro instances | <0.01% | RAM degradation kills live-game training |
| V4 | Python sim, single agent | 16 workers | inflated* | Sim works, combo ranking helps enormously |
| V5 | Python sim, dual agent | 16 workers | 0% | Shop starvation is structural -- failed after 12 runs |
| **V6** | **Python sim, single agent** | **16 workers** | **1.9%** | **Sim audit + heuristic rewards, combo ranker is the ceiling** |
| V7 | TBD | TBD | Target: 30% | Strategic card selection + long-horizon planning |

*V4 produced 35k+ wins across 6 training runs, but all results were inflated by two issues discovered later in V6: (1) fixed seeds caused memorization (agent won on ~16k unique seeds out of 2^31 possible), and (2) the Burglar joker was incorrectly implemented as a scaling mult engine (+3 mult/hand, accumulating permanently) instead of a hand modifier (+3 hands, -3 discards). The agent's dominant strategy (Burglar + Green Joker + Space Joker producing 612M chip Pairs) was entirely a training artifact. **V6 Run 6 (1.9% win rate) is the first legitimate measurement** -- random seeds, fully audited sim, no broken jokers.

### Throughput Progression

| Version | Method | Steps/sec | vs V1 |
|---------|--------|:---------:|:-----:|
| V1 | File IPC, SB3, 1 env | ~0.08 | 1x |
| V2 | File IPC, Ray, 8 envs | ~6 | 75x |
| V3 | Socket IPC, 8 envs | ~14* | 175x |
| V4-V6 | Python sim, 16 workers | ~4,000 | 50,000x |

*V3 degraded from ~39 sps to ~14 sps over training due to Lua GC / RAM issues.

---

## Current Version: V6 -- Single Agent with Enhanced Shop Awareness

V5's dual-agent architecture (separate play + shop networks with a 32-dim communication vector) failed after 12 training runs due to **shop starvation**: the shop agent received <0.1% of training steps and could never learn. V6 returns to V4's single-agent approach with all lessons applied.

### What Changed from V4

1. **Critical combo scoring bug fix** -- V5's `score_hand()` was called with wrong arguments, causing all combos to score 0. This single bug caused 10 failed V5 runs. V6 uses V4's correct implementation.

2. **+60 observation features** (342 -> 402 dims) -- Reroll cost, voucher ownership flags, boss blind one-hot, draw pile composition (suit ratios, face/ace/number counts), enhancement/edition/seal counts.

3. **Heuristic shop rewards** -- +0.3 for buying jokers, +0.2 for using planets, -0.2 for leaving with buyable jokers and empty slots, -0.02/dollar wasted above interest cap, -0.5 for selling jokers with open slots and no upgrade target.

4. **Random seeds per episode** -- V4 used fixed seeds per worker. Across all V4 runs the agent accumulated 35k+ wins, but on a limited pool of memorized seeds. V6 randomizes every episode for true generalization.

5. **Full simulation audit** -- 32 joker fixes (~30% of implementations were wrong), 3 non-joker fixes. Burglar was implemented as a scaling mult joker instead of a hand modifier, which made the agent's entire winning strategy (Burglar + Green Joker + Space Joker) a training artifact.

6. **Boss blind loop fixes** -- bl_mouth and bl_eye could cause infinite loops when all combos were blocked. Combo ranker now filters restricted combos with a fallback.

7. **Episode truncation** -- 2000-step max prevents degenerate no-op loops.

### Architecture

```
Input (402 dims) -> Embed (512, ReLU) -> 4x ResidualBlock(512) -> Actor (46) / Critic (1)

ResidualBlock: x + ReLU(FC(ReLU(FC(x)))) with LayerNorm
Total params: ~2.34M
```

### Observation Space (402 features)

| Range | Dims | Description |
|-------|:----:|-------------|
| `[0:14]` | 14 | Game scalars: ante, blind, boss flag, score progress, target, hands/discards left, dollars, joker count, phase one-hot |
| `[14:222]` | 208 | Hand cards: 8 slots x 26 features (rank, suit, enhancement, edition, seal, debuff, present) |
| `[222:272]` | 50 | Joker slots: 5 x 10 features (present, type ID, rarity, edition, mult/chips/sell/destroyed/mult_mult/price) |
| `[272:314]` | 42 | Shop items: 7 x 6 features (available, kind, price, affordable, has joker slot, has consumable slot) |
| `[314:326]` | 12 | Planet levels: 12 hand types |
| `[326:342]` | 16 | Consumables: 2 x 8 features (present, type flags, planet target, afford context, hands left, phase) |
| `[342:344]` | 2 | Reroll info: cost, free rerolls remaining |
| `[344:371]` | 27 | Voucher ownership flags |
| `[371:386]` | 15 | Boss blind one-hot (15 boss types) |
| `[386:394]` | 8 | Draw pile composition: 4 suit ratios + face/ace/number fractions + pile size |
| `[394:402]` | 8 | Draw pile modifiers: foil/holo/poly/gold/wild/seal counts |

### Action Space -- Discrete(46)

```
BLIND_SELECT:    30 = play blind, 31 = skip blind (non-boss only)
SELECTING_HAND:  0-19 = play combo i (ranked by score_hand() descending)
                 20-27 = discard card i (single card)
                 28-29 = use consumable 0/1
SHOP:            32-38 = buy shop item 0-6
                 39-43 = sell joker 0-4
                 44 = reroll, 45 = leave shop
```

### Reward Structure

| Signal | Value | Notes |
|--------|-------|-------|
| Blind cleared | +2.0 x (9 - ante) | Inverse-ante: ante 1 = +16, ante 8 = +2 |
| Ante completed | +5.0 | Boss blind beaten |
| Game won | +50.0 | Cleared ante 8 boss |
| Game lost | -2.0 | |
| Score progress | 0.05 x log1p(delta) x 100 | Dense, log-scaled to prevent reward explosion |
| Buy joker | +0.3 | Into an empty slot |
| Use planet | +0.2 | Free hand level upgrade |
| Leave with buyable jokers | -0.2 | Empty slots + affordable jokers in shop |
| Excess money | -0.02/dollar | Above nearest $5 interest boundary (cap $25) |
| Sell blunder | -0.5 | Selling with open slots and no expensive upgrade |

### Training Configuration

```
PPO: LR=3e-4, gamma=0.99, lambda=0.95, clip=0.2
     entropy=0.01, VF=0.5, grad_clip=0.5
     epochs=10, minibatch=128

Workers: 16 (multiprocessing, CPU inference)
Batch: 32,768 steps/iter (2048/worker)
Truncation: 2000 steps/episode
Seeds: randomized per episode
Device: CUDA (RTX 3080 Ti for PPO updates)
```

---

## V6 Training Runs

> **Note on win rates:** Runs 1-4 used the broken Burglar joker (see Run 5). Their win rates
> are inflated and not comparable to Run 6. Runs 1 also used fixed seeds (memorization).
> **Run 6 is the only legitimate measurement.**

### Runs 1-4 -- Pre-Burglar Fix (inflated results)

| Run | Seeds | Config | Result | Notes |
|-----|-------|--------|--------|-------|
| 1 | Fixed | 4k batch | 19k wins, ante 9 | Seed memorization (117 unique seeds) |
| 2 | Random | 32k batch | Ante 9 by iter 4 | First generalization attempt, killed for boss fix |
| 3 | Random | 32k batch | 2.5% win rate | Boss blind loop fix applied, 70% die at ante 1 |
| 4 | Random | 32k batch | Killed iter 117 | Added money/sell penalties, killed for Burglar fix |

All runs used the broken Burglar joker. The agent's dominant strategy (Burglar + Green Joker + Space Joker, producing 612M chip Pairs) was entirely a training artifact.

### Run 5 -- Burglar Bug Fix (hard reset)

**Discovery:** Burglar was implemented as a scaling mult joker (+3 mult/hand played, accumulating permanently) instead of a hand modifier (+3 hands, -3 discards per blind). Combined with Green Joker and Space Joker, this produced 612 million chip Pairs by ante 8.

**Impact:** The agent's entire winning strategy was built on a broken joker. All previous win rates (V4 and V6 runs 1-4) were inflated. Hard reset of learning.

### Run 6 -- Full Sim Audit + Card Counting (first legitimate measurement)

**32 joker fixes** across 6 modules (~30% of implementations wrong): 7 completely wrong effects, 8 additive->multiplicative corrections, 13 other fixes, 4 stubs implemented. Also fixed Lucky enhancement probability (1/5 -> 1/4), Stone card scoring, and passive joker accumulation bugs.

**Result: 1.9% win rate** (9,031 wins / 476,930 episodes), converged by iter ~100, random seeds, fully audited sim. Green Joker (51%) + Space Joker (45%) still dominant. Killed at iter 287.

**Conclusion:** The combo ranker IS the policy ceiling. The agent converges to "always play action 0" (best combo) within 50 iterations. It cannot learn strategic discarding because card selection is automated. 80% of games die at ante 1 boss blind (600 chips, 4 hands, no jokers) -- a human survives this 95%+ of the time by discarding intelligently.

---

## Why 1.9% Is the Ceiling

| Problem | Why It Matters |
|---------|---------------|
| No strategic discards | Agent can't turn a bad hand into a good one. At mercy of initial draw. |
| Greedy combo ranking | Always plays the highest-scoring combo, even when "just enough" would save resources. |
| No multi-round planning | Plays each hand in isolation. Can't plan across 4 hands per blind. |
| No conditional strategy | Same policy regardless of joker loadout. Doesn't adapt play style to synergies. |

80% of games die at ante 1. The agent needs to discard to survive, and it can't.

---

## Simulation

The Python sim (`balatro_sim/`) reimplements Balatro's full ruleset:

| Module | Contents |
|--------|----------|
| `game.py` (661 lines) | Full game loop -- 5 states, 15 boss blind effects, ante progression, win/loss |
| `scoring.py` (142 lines) | Chip/mult computation, enhancements, editions, seals, retriggers |
| `hand_eval.py` (172 lines) | All 12 hand types including Flush Five / Flush House |
| `jokers/` (6 modules) | **164 jokers**, fully audited (504 tests passing) |
| `consumables.py` (549 lines) | 12 planets, 22 tarots, 18 spectrals, 27 vouchers |
| `shop.py` (361 lines) | Weighted rarity rolls, buy/sell/reroll, 150+ item catalogue |
| `card.py` (75 lines) | Card representation with enhancement/edition/seal |
| `constants.py` (109 lines) | Game constants, starting values |
| `quality.py` (94 lines) | Loadout quality estimator (V5, kept for reference) |

Test suite: **504 tests passing** across joker behavior, scoring, boss blinds, game transitions, consumables, and environment integration.

---

## What We Learned

### Architecture

**Don't split what doesn't need splitting.** V5's dual-agent created more problems (starvation, credit assignment, entropy) than the single-agent "shop is 5% of steps" issue it was designed to solve. A single agent with heuristic shop rewards outperformed 12 dual-agent runs in 4 iterations.

**Pre-ranking actions matters enormously.** Sorting combos by actual `score_hand()` output (accounting for jokers, planet levels, enhancements) caused ante 9 by iteration 2 instead of iteration 460.

**Entropy floors fight the gradient when the optimal policy is genuinely low-entropy.** With correct combo ranking, action 0 is overwhelmingly dominant. Forcing exploration degraded performance.

### Simulation Fidelity

**Fix the fundamentals first.** The V5 combo scoring bug caused 10 failed training runs before being discovered. One broken line of code masked every other improvement.

**Audit everything.** ~30% of jokers were implemented incorrectly. The agent's best strategy (Burglar + Green + Space) was entirely built on a broken joker producing 612M chip Pairs.

### Training

**Seed diversity prevents memorization.** V4's fixed seeds produced 35k+ wins but on a limited pool of memorized games. Random seeds per episode forces true generalization across millions of unique configurations.

**Heuristic reward shaping bootstraps learning.** Dense rewards for smart shop actions give signal from step 1. The agent can surpass the heuristic as it trains.

**Inverse-ante rewards help early survival.** Weighting ante 1 blind clears at 8x ante 8 clears focuses learning where 80% of deaths occur.

---

## Repository Layout

```
balatro-rl/
├── train_sim.py              # V6 training -- single-agent PPO (current)
├── train_v5.py               # V5 training -- dual-agent PPO (deprecated)
├── balatro_sim/              # Python Balatro simulation
│   ├── game.py               # Full game loop (BalatroGame)
│   ├── env_sim.py            # V6 environment (402 obs, 46 actions)
│   ├── env_v5.py             # V5 dual-agent env (deprecated)
│   ├── scoring.py            # Chip/mult scoring engine
│   ├── hand_eval.py          # Hand type evaluation (12 types)
│   ├── consumables.py        # Planets, tarots, spectrals, vouchers
│   ├── shop.py               # Shop system, rarity rolls
│   ├── card.py               # Card representation
│   ├── constants.py          # Game constants
│   ├── quality.py            # Loadout quality estimator
│   └── jokers/               # 164 jokers across 6 modules
│       ├── base.py, chips.py, mult.py, scaling.py
│       ├── hand_type.py, economy.py, misc.py
│       └── tests/            # Joker unit tests
├── tests/                    # Integration tests (504 total passing)
├── scripts/                  # Plotting and analysis
│   ├── training_report.py    # Training dashboard
│   ├── plot_runs.py          # Multi-run comparison
│   └── plot_v5_run8.py, plot_v4.py, plot_training.py
├── results/                  # Design notes and run logs
│   ├── V7_PLANNING.md        # V7 architecture design (strategic card selection)
│   ├── V6_DESIGN_NOTES.md    # V6 design and changes from V4
│   ├── V6_RUN_LOG.md         # V6 training run journal (6 runs)
│   ├── V5_DESIGN_NOTES.md    # V5 dual-agent spec (for reference)
│   ├── V5_RUN_LOG.md         # V5 training runs (12 runs, all failed)
│   └── V4_DESIGN_NOTES.md, V3_DESIGN_NOTES.md, V1/V2 results
├── checkpoints_sim/          # V6 model weights + episode logs
├── checkpoints_v5/           # V5 model weights (deprecated)
├── logs_sim/                 # Training logs, highlight episodes
├── legacy/                   # V1-V3 training scripts and tests
├── launch/                   # PowerShell scripts (V3 instance management)
├── mod/, mod_v2/             # Lua mod files (V1-V3)
└── versions/                 # Archived Lua versions
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
# Run tests (504 passing)
python -m pytest balatro_sim/jokers/tests/ tests/ -v

# Train V6 (single agent, default: 16 workers, 32k batch, 1000 iters)
python train_sim.py --workers 16 --steps-per-worker 2048 --iterations 1000

# Smaller batch for testing
python train_sim.py --workers 4 --steps-per-worker 256 --iterations 100

# Resume from checkpoint
python train_sim.py --resume checkpoints_sim/iter_0100.pt

# Benchmark sim throughput
python -m balatro_sim.env_sim
```

---

## Next: V7 -- Strategic Card Selection

The 2% ceiling exists because the combo ranker automates card selection. The agent picks from pre-ranked combos and converges to "always play action 0." It can't discard strategically, can't play "just enough," and can't adapt to joker synergies.

**Planned approach: Hierarchical Intent + Card Selection (Approach E)**

1. **Strategic intent** (discrete action): "play for score", "discard to improve hand", "play just enough to clear blind", "discard to thin deck"
2. **Card selection** (learned scoring head): given the intent, score each card and select the optimal subset

This separates strategy (when to be greedy vs conservative) from tactics (which cards to play). See [V7 Planning](results/V7_PLANNING.md) for full analysis of 5 approaches considered.

**Target: 30% win rate.**

---

## Acknowledgments

Built on [Steamodded](https://github.com/Steamopollys/Steamodded) for Lua mod loading and [Handy](https://github.com/nicholassam6425/balatro-mods) for speed control (V1-V3).
