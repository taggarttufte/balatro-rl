# Balatro RL

A reinforcement learning agent that learns to play [Balatro](https://www.playbalatro.com/) using PPO.

Seven versions over five months: from a Lua mod with file-based IPC, through socket parallelism, a failed dual-agent experiment, a complete Python simulation (V6: 1.9% win rate), and a hierarchical intent + learned card selection architecture (V7: 2.35% win rate after 6 reward tuning runs). Six distinct V7 runs explored reward shaping to push past the 2% plateau — the agent learned strategic card play from scratch but couldn't break the Green+Space joker lock-in.

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
| V6 | Python sim, single agent | 16 workers | 1.9% | Sim audit + heuristic rewards, combo ranker is the ceiling |
| **V7** | **Hierarchical intent + learned card selection** | **16 workers** | **2.35%** | **Agent can learn card play from reward, but shop strategy plateaus at ~2%** |
| V8 | TBD | TBD | Target: 5%+ | Curriculum learning / search-guided imitation |

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
├── train_v7.py               # V7 training -- hierarchical intent + learned card selection (current)
├── train_sim.py              # V6 training -- single-agent PPO with pre-ranked combos
├── train_v5.py               # V5 training -- dual-agent PPO (deprecated)
├── balatro_sim/              # Python Balatro simulation
│   ├── game.py               # Full game loop (BalatroGame)
│   ├── env_v7.py             # V7 environment (434 obs, hierarchical actions)
│   ├── env_sim.py            # V6 environment (402 obs, 46 actions)
│   ├── env_v5.py             # V5 dual-agent env (deprecated)
│   ├── card_selection.py     # V7: subset enumeration, intent-to-action translation
│   ├── synergy.py            # V7: 164 jokers tagged, coherence scoring, auto-positioning
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
├── tests/                    # Integration tests
│   ├── test_env_v7.py        # V7 environment + hierarchical action tests
│   ├── test_card_selection.py # Subset enumeration, boss validation
│   └── test_*.py             # Joker + game flow tests (504 total passing)
├── scripts/                  # Plotting and analysis
│   ├── training_report.py    # Training dashboard
│   ├── plot_runs.py          # Multi-run comparison
│   └── plot_v5_run8.py, plot_v4.py, plot_training.py
├── results/                  # Design notes and run logs
│   ├── V7_PLANNING.md        # V7 architecture design (5 approaches analyzed)
│   ├── V7_RUN_LOG.md         # V7 training runs (6 runs, final 2.35% win rate)
│   ├── V6_DESIGN_NOTES.md    # V6 design and changes from V4
│   ├── V6_RUN_LOG.md         # V6 training run journal (6 runs)
│   ├── V5_DESIGN_NOTES.md    # V5 dual-agent spec (for reference)
│   ├── V5_RUN_LOG.md         # V5 training runs (12 runs, all failed)
│   └── V4_DESIGN_NOTES.md, V3_DESIGN_NOTES.md, V1/V2 results
├── checkpoints_v7/           # V7 active model weights + episode logs
├── checkpoints_v7_run4/      # V7 Run 4 checkpoints (2.35% peak)
├── checkpoints_v7_run5/      # V7 Run 5 checkpoints (slot filling peak)
├── checkpoints_sim/          # V6 model weights + episode logs
├── checkpoints_v5/           # V5 model weights (deprecated)
├── logs_v7/                  # V7 training logs
├── logs_sim/                 # V6 training logs, highlight episodes
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
# Run tests
python -m pytest balatro_sim/jokers/tests/ tests/ -v

# Train V7 (current, hierarchical intent + learned card selection)
python train_v7.py --workers 16 --steps-per-worker 2048 --iterations 1000

# Train V7 with V6 weight migration (warm start)
python train_v7.py --workers 16 --steps-per-worker 2048 --iterations 1000 \
    --migrate-v6 checkpoints_sim/iter_0280.pt

# Train V6 (pre-ranked combos)
python train_sim.py --workers 16 --steps-per-worker 2048 --iterations 1000

# Smaller batch for smoke testing
python train_v7.py --workers 4 --steps-per-worker 256 --iterations 10

# Resume from V7 checkpoint
python train_v7.py --resume checkpoints_v7/iter_0200.pt

# Benchmark sim throughput
python -m balatro_sim.env_sim
```

---

## V7 — Hierarchical Intent + Learned Card Selection

V6's ceiling was structural: a pre-ranked combo action space meant the agent always
picked "action 0" (best combo). V7 replaced this with a two-level decision trained
end-to-end via PPO.

### Architecture

```
Input (434 dims) -> Embed(512) -> 4x ResBlock(512) -> trunk (512)

SELECTING_HAND (new hierarchical):
  trunk -> intent_head(3)                    -> intent logits
  trunk + intent_embed(32) -> card_head(8)   -> card scores (sigmoid)
  card_scores -> enumerate 218 subsets -> softmax -> Categorical -> sample

Other phases:
  trunk -> phase_head(17)                    -> blind/shop actions

Shared:
  trunk -> critic(1)                         -> value

~2.5M params (vs V6's 2.3M)
```

The **intent head** picks PLAY / DISCARD / USE_CONSUMABLE. The **card scoring head**
outputs 8 scores (one per card slot) conditioned on the intent. These scores define a
probability distribution over the 218 possible card subsets (1-5 cards from the 8-card
hand), and a specific subset is sampled. The factored log_prob (intent + subset) feeds
standard PPO.

### Observation Space Changes (V6 -> V7)

Added 4 features per card slot (32 new dims total, 402 -> 434):
- `suit_match_count` — other hand cards sharing this suit
- `rank_match_count` — other hand cards sharing this rank
- `straight_connectivity` — cards within ±4 rank distance
- `card_chip_value` — normalized base chip contribution

### V7 Run History

Six training runs explored reward shaping to push past the 2% win rate plateau:

| Run | Win Rate | Avg Jokers | Coherence | Key Change |
|:-:|:-:|:-:|:-:|---|
| Run 1 | 0.0007% | — | — | Initial V7: no card quality reward — agent couldn't learn card selection |
| Run 2 | 1.98% | 2.3 | 0.50 | Added card quality reward (played_score / best_possible_score) |
| Run 3 | 1.80% | 2.0 | 0.62 | Synergy-based shop rewards — too weak to break Green+Space lock-in |
| Run 4 | **2.35%** | 4.0 | 0.65 | Slot-scaled synergy + ante-scaled empty slot penalty |
| Run 5 | 2.20% | 4.90 | 0.63 | Auto-position Blueprint/Brainstorm/Ceremonial + smart sell rewards |
| Run 6 | 2.23% | 3.33 | 0.58 | Amplified coherence rewards BACKFIRED (agent gamed penalties) |

**V7 peak: 2.35% (Run 4 last 50 iters)**. See [V7 Run Log](results/V7_RUN_LOG.md) for
per-run details.

### V7 Learnings

**What the agent learned:**
- Play good hands (card quality reward → avg ~0.8 hand quality)
- Use discards strategically (28-32% of SELECTING_HAND actions, vs 0% in V6)
- Fill joker slots when pressured (96% full loadouts in Run 5)
- Maintain some coherence in buys (0.58-0.65)

**What the agent didn't learn:**
- Conditional strategy (Flush build vs Pair build vs Straight build)
- Long-horizon joker planning (multi-purchase coordinated strategies)
- When to deviate from Green+Space based on shop availability
- Positional joker use (Blueprint/Brainstorm appear in <4% of wins despite auto-positioning)

**Key insight from Run 6 (coherence paradox):**
Amplifying coherence rewards caused the agent to buy FEWER jokers, not more coherent ones.
Strong penalties for sub-coherent buys created a perverse incentive: doing less to get
penalized less. The agent defaulted to Green+Space only (both universal, no strategy tag,
always neutral 0.5 coherence) and collected baseline rewards safely. This is a fundamental
limitation of reward shaping with PPO: local optima are sticky.

### Architecture Also Built During V7

- [`balatro_sim/card_selection.py`](balatro_sim/card_selection.py) — Subset enumeration
  (cached by hand size), subset logit computation, boss blind validation
- [`balatro_sim/synergy.py`](balatro_sim/synergy.py) — 164 jokers tagged by strategy
  (hand type, suit, mechanic), coherence scoring, ante-aware decay, auto-positioning
- [`balatro_sim/env_v7.py`](balatro_sim/env_v7.py) — V7 environment with hierarchical
  action handling and evolved reward structure
- [`train_v7.py`](train_v7.py) — PPO with factored log_prob (intent + subset) and
  vectorized subset logit reconstruction for the PPO update

### Why V7 Plateaued

After 6 runs of reward shaping, win rate consistently lands at 2.0-2.35%. The ceiling is
structural, not a reward tuning issue:

1. **Green+Space are legitimately S-tier** in Balatro, like Blueprint/Brainstorm.
   Expert players always buy them. The agent picking them up isn't wrong.
2. **The agent has only ONE decision branch** — "if I see Green/Space, buy them."
   It lacks backup branches for "commit to Flush build when flush jokers appear" or
   "reroll when shop has no strategy-coherent options."
3. **PPO can't discover multi-step coordinated strategies** from reward signal alone.
   Flush builds need 3-4 flush jokers across multiple shops; single-step exploration
   never finds the coordinated path.
4. **Local optimum lock-in** — once the agent converges to Green+Space, every reward
   tweak creates a new local optimum around that strategy.

### Next Steps (V8)

Options for breaking the 2% plateau:

1. **Curriculum learning** — Train without Green+Space for first 500k steps to force
   learning alternatives, then re-enable. Most promising given V7 findings.
2. **Search-guided imitation** — Monte Carlo search to find winning strategies per game,
   supervise the policy on those decisions. Proven technique from AlphaGo.
3. **Population-based training** — Multiple agents with different reward weights,
   periodic selection and crossover.
4. **Larger network** — 2.5M params may be too small for conditional strategy. Try 10M+.
5. **Add move_joker action** — Explicit joker reordering so Blueprint/Brainstorm can be
   used strategically by the agent, not just auto-placed.

---

## Acknowledgments

Built on [Steamodded](https://github.com/Steamopollys/Steamodded) for Lua mod loading and [Handy](https://github.com/nicholassam6425/balatro-mods) for speed control (V1-V3).
