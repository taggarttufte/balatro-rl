# Balatro RL

A reinforcement learning agent that learns to play [Balatro](https://www.playbalatro.com/) using PPO.

The project went through four major versions: from a simple Lua mod with file-based IPC, through socket-based parallelism, to a complete Python simulation that trains at **429x** the speed of the original approach.

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

## Current Version: V4 — Python Simulation

### Why the rewrite

V3 trained against 8 live Balatro instances over socket IPC at ~14 steps/sec after RAM
degradation. The Lua runtime leaked memory across long runs, instances needed manual
restarts, and the GIL limited Python-side concurrency. The biggest bottleneck: every
training step required round-tripping through a live game process.

**The core insight:** Balatro's rules are fully deterministic given a seed. We don't need
the game running during training — we just need a correct Python model of it.

V4 replaces the live game with a pure Python simulation. Training is now:
- **~4,000 steps/sec** using 16 multiprocessing workers (bypasses GIL entirely)
- Zero RAM degradation (no Lua runtime)
- No instance management, no watchdog scripts, no socket timeouts
- Real Balatro still used as the deployment/validation target via the V3 socket IPC

### V4 Architecture

```
train_sim.py
  16 multiprocessing workers (one per CPU core)
  each worker: BalatroSimEnv → BalatroGame
  
  Rollout collection: 16 workers × 512 steps = 8,192 steps/iter
  PPO update: CUDA (RTX 3080 Ti), minibatch=256, 10 epochs
  
  IPC: multiprocessing.Pipe (no sockets, no file I/O)
```

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
| `env_sim.py` | Gymnasium-compatible wrapper, obs=342, actions=46 |

### Observation Space (342 features)

| Range | Description |
|-------|-------------|
| `[0:40]` | Hand cards — 8 slots × 5 features (rank, suit, enhancement, edition, seal) |
| `[40:100]` | Play combos — 20 slots × 3 features (hand_type_id, score_estimate, num_cards) |
| `[100:108]` | Discard options — 8 slots |
| `[108:174]` | Jokers — 5 slots × (type_id + state features) |
| `[174:198]` | Consumables, vouchers |
| `[198:210]` | Scalar state — ante, blind_idx, score_progress, hands_left, discards_left, dollars, etc. |
| `[210:342]` | Deck composition, hand type levels |

### Action Space — `Discrete(46)`

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

Play combos are pre-ranked by actual `score_hand()` output, accounting for current
jokers, planet levels, and card enhancements. Action 0 is always the highest-scoring play.

### Reward Structure

| Signal | Value |
|--------|-------|
| Blind cleared | +2.0 |
| Ante completed | +5.0 |
| Game won (past ante 8) | +20.0 |
| Game lost | -2.0 |
| Score progress (log-scaled) | `0.05 × log(1 + delta) × 100` |

Score progress uses log scaling so overshooting the blind target is rewarded but with
diminishing returns — 1.5x target ≈ 1.3x reward, 5x ≈ 2x reward, 1000x ≈ 10x reward.
Linear scaling produced reward values up to 82 million, destabilizing PPO updates.

### Network Architecture

**Current (run 5+):** 6-layer residual network, 2.3M parameters
```
input (342) → embed (512) → 4 × ResidualBlock(512) → actor head (46) / critic head (1)

ResidualBlock: x + ReLU(fc2(ReLU(fc1(x)))), with LayerNorm
```

**Previous (runs 1-4):** 3-layer MLP, 345k parameters
```
input (342) → 512 → 256 → 128 → actor / critic
```

The residual architecture was chosen based on research showing deeper networks are
needed to learn 4+ way conditional interactions (e.g. "Green Joker + Burglar + high
hand level + sufficient hands remaining → play this combo"). A 3-layer MLP can learn
pairwise correlations but struggles with deeper joker synergies.

---

## Training Runs

All runs use 16 workers, `train_sim.py`, same sim code. The key variables are batch size and network architecture.

| Run | Batch | Minibatch | Network | Params | Iters | Notes |
|-----|-------|-----------|---------|--------|-------|-------|
| Run 1 | 4,096 | 128 | 3-layer MLP | 345k | 1,000 | Reward unbounded (linear delta) — spikes to 82M |
| Run 2 | 4,096 | 128 | 3-layer MLP | 345k | ~180 | Killed — same reward issue |
| Run 3 | 4,096 | 128 | 3-layer MLP | 345k | 1,000 | Log1p reward fix; score-sorted combos; highlight logger |
| Run 4 | 32,768 | 512 | 3-layer MLP | 345k | 1,000 | Large batch comparison vs run 3 |
| Run 5 | 8,192 | 256 | 6-layer residual | 2.3M | 1,000 | New architecture |

**Key results (run 3, 1,000 iters / 4.1M steps / 41 min wall time):**
- Best ante: 9 (full win), reached by iter 3
- Full win rate: ~0.47% (965 wins / 79k episodes combined)
- Reward: 1.7 → 48 avg (last 10 iters)
- Entropy stable at 2.5 — policy stayed exploratory throughout
- Top jokers in winning runs: Green Joker (28%), Burglar (27%), Space Joker (27%)

---

## What We Learned

### V3 Lessons (Live Game Training)

**RAM degradation is fatal for long runs.** After ~4-6 hours, 8 Balatro instances ballooned
to 23GB RAM combined. The Lua GC (even with aggressive tuning: `setpause=80, setstepmul=400`)
couldn't keep up with the object churn from 128x game speed. Training degraded from 39 sps
to 3-10 sps. A watchdog script helped but didn't fully solve it — instances needed hard kills
and restarts every few hours.

**Threading vs Ray.** Ray's VectorEnv steps environments sequentially — 8 environments gave
the same throughput as 1. Custom Python threading releases the GIL during `socket.recv()`,
giving true I/O concurrency. But Python threading still hits CPU limits from the game sim.

**Socket IPC vs file IPC.** At 128x game speed, file polling dropped actions when Lua
advanced multiple states between Python polls. Socket IPC is synchronous — Lua waits for
a response before proceeding. 3.31x throughput improvement at identical game speed, and
much richer episodes (20 → 90 steps average) from fewer dropped actions.

**128x speed caused CPU bottleneck.** At 128x, the Balatro instances saturated the CPU
before the Python training loop did. 64x was the sweet spot for 8 parallel instances.

### V4 Lessons (Python Sim Training)

**Pre-ranking actions matters enormously.** Run 1 used hand type priority for combo ordering,
which correctly puts Flush Five above High Card. But within the same hand type, it used
rank sum as a tiebreak, ignoring joker effects. With 20 unranked combos, the agent had
to learn which index corresponded to the best play purely from reward signal. Switching to
actual `score_hand()` tiebreaking (accounting for current jokers, planet levels,
enhancements) caused ante 9 to appear by **iteration 2** instead of iteration 460.

**Unbounded rewards destabilize PPO.** The original score progress reward was linear:
`reward = 0.05 × (chips_scored / chips_target) × 100`. With stacked jokers at late antes,
chips_scored can be 10,000× the target — producing rewards of 82 million in a single
episode. This blew up return normalization and caused positive policy gradient loss. Log
scaling fixed it immediately.

**Larger batches = smoother gradients, same peak performance.** Run 4 (32k batch) reached
the same reward levels as run 3 (4k batch) with significantly less variance in the learning
curve. Both hit best ante 9 within the first 5 iterations. Larger batches don't necessarily
learn faster in terms of steps, but updates are more stable.

**Episode length is the key training health indicator.** Starting episodes: ~15 steps (agent
dies at ante 1). By iter 1000: ~100+ steps per episode. This measures whether the policy
is genuinely improving — a policy that's stuck would maintain short episodes.

**Winning joker builds are learnable.** After 80k+ episodes, the agent consistently finds
the Green Joker + Burglar + Space Joker core in ~28% of winning runs. This makes game-
theoretic sense: Burglar gives extra hands → Green Joker scales mult per hand played →
Space Joker levels up hand types from the extra plays. The agent discovered this synergy
without being told it exists.

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
├── train_sim.py              # V4 training — 16-worker multiprocessing PPO (current)
├── balatro_sim/              # Python Balatro simulation
│   ├── game.py               # Full game loop (BalatroGame class)
│   ├── env_sim.py            # Gymnasium env wrapper
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
├── scripts/                  # Plotting and analysis
│   ├── plot_runs.py          # Multi-run comparison charts
│   ├── plot_v4.py            # V4 training dashboard
│   └── plot_training.py      # V2/V3/V4 comparison
├── results/                  # Design notes and historical results
│   ├── V1_RESULTS.md, V2_SINGLE_RESULTS.md, V2_PARALLEL_RESULTS.md
│   ├── V3_DESIGN_NOTES.md, V4_DESIGN_NOTES.md
│   └── NOTES.md, PARALLEL_SETUP.md
├── legacy/
│   ├── training/             # V1-V3 training scripts
│   └── tests/                # V1-V3 test and analysis scripts
└── launch/                   # PowerShell scripts for V3 instance management
```

---

## Setup (V4 — Python Sim)

```bash
git clone https://github.com/taggarttufte/balatro-rl
cd balatro-rl
python -m venv env && source env/bin/activate  # or env\Scripts\activate on Windows
pip install torch numpy gymnasium
```

```bash
# Run tests
python -m pytest balatro_sim/tests/ -v

# Train (default: 16 workers, 8k batch, 1000 iters)
python train_sim.py --workers 16 --steps-per-worker 512 --iterations 1000

# Resume from checkpoint
python train_sim.py --resume checkpoints_sim/iter_0500.pt --iterations 500
```

---

## Acknowledgments

Built on [Steamodded](https://github.com/Steamopollys/Steamodded) for Lua mod loading
and [Handy](https://github.com/nicholassam6425/balatro-mods) for speed control (V3).
