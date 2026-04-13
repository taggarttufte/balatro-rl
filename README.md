# Balatro RL

A reinforcement learning agent that learns to play [Balatro](https://www.playbalatro.com/) using PPO.

Eight versions over five months: from a Lua mod with file-based IPC, through socket parallelism, a failed dual-agent experiment, a complete Python simulation (V6: 1.9% win rate), a hierarchical intent + learned card selection architecture (V7: 2.35% win rate after 6 reward tuning runs), and a self-play multiplayer implementation (V8: training with multiplayer-aware observations). V7 broke through 2% win rate but hit a ceiling caused by Green+Space joker lock-in. V8 uses self-play on the multiplayer Balatro ruleset to break that lock-in through same-seed strategy comparison.

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

| Version | Method | Peak Win Rate | Key Learning |
|---------|--------|:---:|-------------|
| V1 | File IPC, SB3 | <0.01% | Proof of concept |
| V2 | File IPC, Ray | <0.01% | Parallelism works but IPC is the bottleneck |
| V3 | Socket IPC, custom PPO | <0.01% | RAM degradation kills live-game training |
| V4 | Python sim, single agent | inflated* | Sim works, combo ranking helps enormously |
| V5 | Python sim, dual agent | 0% | Shop starvation is structural — failed after 12 runs |
| V6 | Python sim + full audit | 1.9% | Sim fidelity matters, combo ranker is the ceiling |
| V7 | Hierarchical intent + learned card selection | 2.35% | Agent can learn card play from reward; plateaus at Green+Space lock-in |
| **V8** | **Self-play multiplayer + MP-aware obs** | **In training** | **Same-seed self-play to break lock-in through differential gradient** |

*V4 produced 35k+ wins but on fixed seeds + broken Burglar joker. V6 Run 6 (1.9%) is the first legitimate measurement with random seeds + audited sim.

### Throughput Progression

| Version | Method | Single-env sps | Training sps (16-20 workers) | vs V1 |
|---------|--------|:---:|:---:|:---:|
| V1 | File IPC, SB3, 1 env | ~0.08 | — | 1x |
| V2 | File IPC, Ray, 8 envs | ~6 | — | 75x |
| V3 | Socket IPC, 8 envs | ~14** | — | 175x |
| V4-V6 | Python sim, 16 workers | ~1,000 | ~900-1000 | ~12,500x |
| V7 | +`_best_hand_score` vectorization | **~1,535** | ~985 | ~19,000x |
| **V8** | MP env + extended obs (20 workers) | ~1,400 | **~780** | — |

**V3 degraded from ~39 sps to ~14 sps over training due to Lua GC / RAM issues.

**V7 → V8 throughput tradeoff:**
- Each V8 step runs 2 agents (both players) on same seed → 2x trajectories per step
- Look-ahead win evaluation + life tracking + PvP resolution adds coordination overhead
- Extended obs (434 → 438) slightly larger network forward pass
- Net: V8 at ~780 sps is slower per step, but effective training data throughput is similar since each step produces data for two players

**V7 → V8 optimizations partially offsetting the regression:**
- `_best_hand_score` vectorization: 218 subsets → top 5-8 candidates via priority filtering (lossless, 1.5x speedup)
- Batched inference within workers: per-step batched forward pass
- Hyperthreading: 16 → 20 workers on 16-core Ryzen 9 7950X (+2-5%)
- Banned joker filter for shop generation

---

## Current Version: V8 — Self-Play Multiplayer with MP-Aware Observations

V7 hit a 2.35% win rate ceiling because the agent locked into a single dominant strategy: Green Joker + Space Joker. Six runs of reward shaping couldn't break it. The fundamental issue is that PPO with averaged rewards can't discover multi-step coordinated strategies — the signal is too diluted across episodes.

V8's hypothesis: **same-seed self-play gives PPO direct differential gradient signal.** When two policies face identical game states and one wins, PPO sees exactly which strategy was better. Additionally, the multiplayer PvP boss blind creates burst-scoring pressure that naturally favors different strategies than long-term scaling (which is what Green+Space optimize for).

### Multiplayer Rules (Standard Ranked + House Rule)

```
Each ante: Small Blind → Big Blind → PvP Boss Blind
  Both players play all blinds on the SAME SEED (identical card draws)
  Independent shops (different random offerings per player)
  4 lives per player
  HOUSE RULE: failing ANY blind costs a life (vs official mod: only PvP losses)
  PvP blind: higher score wins, loser loses a life + gets $4/life comeback money
  Game ends at 0 lives; survivor wins, ties broken by ante progression
  Auto-placed positional jokers (Blueprint/Brainstorm/Ceremonial Dagger)
  4 banned jokers (Chicot, Matador, Mr. Bones, Luchador — boss-blind-dependent)
```

### Network Architecture

```
Input (438) → Embed(512, ReLU) → 4× ResidualBlock(512) → trunk (512)
    [V7 obs 434 dims: hand, jokers, shop, planets, consumables, shop context]
    [V8 extension +4 dims: self_lives/4, opp_lives/4, opp_pvp_score, is_pvp_blind]

SELECTING_HAND (hierarchical from V7):
    trunk → intent_head(3)                              → PLAY/DISCARD/USE_CONSUMABLE
    trunk + intent_embed(32) → card_head(8, sigmoid)    → card scores
    card_scores → 218 subsets → softmax → Categorical  → sample subset

Other phases:
    trunk → phase_head(17)                              → blind/shop actions

Shared:
    trunk → critic(1)                                   → value

~2.48M params
```

### Observation Space (438 features)

| Range | Dims | Description |
|-------|:----:|-------------|
| `[0:14]` | 14 | Game scalars (ante, blind, progress, target, hands/discards left, dollars, phase) |
| `[14:254]` | 240 | Hand cards: 8 slots × 30 features (rank, suit, enhancement, edition, seal, debuff, present, suit_match, rank_match, straight_conn, chip_value) |
| `[254:304]` | 50 | Joker slots: 5 × 10 features |
| `[304:346]` | 42 | Shop items: 7 × 6 features |
| `[346:358]` | 12 | Planet levels |
| `[358:374]` | 16 | Consumable slots: 2 × 8 features |
| `[374:434]` | 60 | Shop context (reroll, vouchers, boss, deck composition, enhancements) |
| `[434:438]` | **4** | **V8 multiplayer state: self_lives/4, opp_lives/4, opp_pvp_score_ratio, is_pvp_blind** |

### Action Space

**SELECTING_HAND phase (hierarchical):**
- Intent head: Discrete(3) → PLAY / DISCARD / USE_CONSUMABLE
- Card scoring head: 8-dim continuous (sigmoid) → defines distribution over 218 subsets
- Sample intent from Categorical, sample subset from softmax(card_scores), factored log_prob = log_prob(intent) + log_prob(subset|intent)

**BLIND_SELECT + SHOP phases:** Discrete(17)
```
0-1     play_blind / skip_blind
2-8     buy shop item 0-6
9-13    sell joker 0-4
14      reroll
15      leave_shop
16      use planet in shop
```

### Reward Structure

**V7 base rewards** (inherited, per player per step):

| Signal | Value |
|--------|:-:|
| Blind cleared | +1.0 × (9-ante) |
| Ante complete | +2.5 |
| Score progress | +0.02 × log1p(delta) × 100 |
| Card quality | +2.0 × (played_score / best_possible_score) |
| Synergy buy (slot 1-5) | +1.5 to +3.0 × synergy (coherence-aware) |
| Anti-synergy buy | -1.0 × (0.5 - synergy) when synergy < 0.5 |
| Empty slot penalty | -0.3 × (ante-1) × empty_slots |
| Per-blind coherence | +1.5 × coherence |
| Episode-end coherence | +6.0 × coherence × ante |
| Smart sell rewards | +0.3 to +2.0 (sacrificial, weak, late-scaling swap) |
| Use planet | +0.2 |

**V8 multiplayer additions:**

| Event | Value |
|-------|:-:|
| PvP win | +3.0 |
| PvP loss | -2.0 |
| Life lost (any cause) | -1.5 |
| Game win (opponent at 0 lives) | +20.0 base |
| Game win "strong" adjustment | +5.0 if winner would reach next PvP intact |
| Game win "shaky" adjustment | -10.0 if winner would die to next small/big blind |
| Game loss (you at 0 lives) | -10.0 |

The **look-ahead win evaluation** simulates the winner's next few blinds with a greedy scripted policy to check whether they'd actually survive. Separates "won because opponent collapsed" from "won because my strategy is sustainable."

### Self-Play Design Decisions

**1. Single shared policy, not multiple networks.**
Both players sample from the same policy. Stochastic action sampling creates divergence. No need for separate networks in Phase 1.

**2. Temperature asymmetry (P2 at temp 1.4).**
Without this, Run 1 had 74% draws because both players' sampled identically from sharp distributions. P2 samples logits scaled by 1.4 (flatter = more exploratory). P1 stays at temp 1.0 (base policy).

**3. Multiplayer observation from day 1.**
Without MP state in the observation, the agent couldn't see its lives, opponent lives, opponent PvP score, or whether it was on a PvP blind. Life-loss penalties were pure reward with no observable context. Run 3 adds 4 MP state features so the policy can condition on multiplayer context.

**4. Fresh training from scratch (no V7 migration).**
Runs 1-2 tried V7→V8 migration. Both failed: V7's sharp distributions got disrupted by V8's new reward signal before replacement strategies could emerge, producing a policy that had forgotten basic play. Run 3 trains from random weights.

### Training Configuration

```
PPO: LR=3e-4, gamma=0.99, lambda=0.95, clip=0.2
     intent entropy=0.05, subset entropy=0.005, phase entropy=0.01
     VF coeff=0.5, grad_clip=0.5, epochs=10, minibatch=128

Workers: 20 (hyperthreading sweet spot on Ryzen 9 7950X)
Batch: 40,960 agent-steps/iter (1024 MP steps × 2 agents × 20 workers)
4 lives per player, fresh random seeds per game
Device: CUDA (RTX 3080 Ti for PPO updates)
```

### V8 Run History

| Run | Status | Key Finding |
|:-:|---|---|
| Run 1 | Killed iter 63 | Shop-skip bug: failed players never bought jokers. 84% games ended at ante 2. |
| Run 2 | Killed iter 71 | V7 migration disrupted basic play (solo win rate 2.35% → 0%). Broke Green+Space lock-in successfully (Green 3%, Space not in top 15) but lost competence. |
| **Run 3** | **In training** | Fresh start + MP obs + all prior fixes. At iter 36: reward ~32, best ante 7, phase entropy healthy at 0.44 (vs V7 migration's collapsed 0.22). |

See [V8_RUN_LOG.md](results/V8_RUN_LOG.md) for detailed per-run journal.

### V8 Key Files

- [`balatro_sim/mp_game.py`](balatro_sim/mp_game.py) — `MultiplayerBalatro` coordinator (lives, phase transitions, PvP resolution, comeback money)
- [`balatro_sim/env_mp.py`](balatro_sim/env_mp.py) — RL environment wrapping two V7 envs with MP-extended observation, banned jokers, revive-to-shop logic, look-ahead win evaluation
- [`train_v8.py`](train_v8.py) — Self-play PPO training loop with temperature asymmetry, batched inference across both players
- [`results/V8_DESIGN_NOTES.md`](results/V8_DESIGN_NOTES.md) — full architecture spec
- [`results/MULTIPLAYER_RULESET.md`](results/MULTIPLAYER_RULESET.md) — Standard Ranked spec from balatromp.com

### Next Steps After Run 3

1. **Specialist population training** — 5 policies with differentiated reward shaping (Flush, Pairs, Face cards, etc.) — planned in [V7_RUN_LOG.md](results/V7_RUN_LOG.md)
2. **Multiplayer-specific jokers** — implement 10 new ranked ruleset jokers (Defensive, Skip-Off, Pacifist, etc.)
3. **Move joker action** — explicit joker reordering so Blueprint/Brainstorm aren't just auto-placed
4. **Larger network** — 2.5M params may be too small for conditional strategy

---

## Previous Versions — Brief

Each version below has detailed design notes + run logs in [`results/`](results/).

### V6 — Audited Python Sim + Single Agent ([design](results/V6_DESIGN_NOTES.md) · [runs](results/V6_RUN_LOG.md))

- **Architecture:** Single-agent PPO, 402-dim obs, Discrete(46) action (20 pre-ranked combos, 8 discards, shop/blind actions)
- **Key achievement:** 1.9% win rate (first legitimate measurement on random seeds with audited sim)
- **Major changes:**
  - **Full sim audit:** Fixed 32 jokers (~30% of implementations were wrong). Most critical: Burglar was a scaling mult joker instead of a hand modifier, producing fake 612M chip Pairs.
  - **Random seeds per episode:** V4 had fixed seeds → 35k wins on 16k unique seeds (memorization). V6 randomized for true generalization.
  - **Heuristic shop rewards:** +0.3 buy joker, +0.2 use planet, -0.2 leave with buyable jokers, -0.5 sell blunder
  - **Boss blind loop fixes:** bl_mouth/bl_eye no longer infinite-loop on blocked combos
- **The ceiling:** 80% of games died at ante 1 because the agent couldn't discard strategically. The pre-ranked combo action space meant it always picked action 0 (best combo) and never learned alternatives.

### V7 — Hierarchical Intent + Learned Card Selection ([design](results/V7_PLANNING.md) · [runs](results/V7_RUN_LOG.md))

- **Architecture:** Two-level action space — intent head (3 actions) + card scoring head (8-dim) → distribution over 218 subsets → sampled via factored log_prob. 434-dim obs (added 4 per-card features). ~2.5M params.
- **Peak win rate:** 2.35% (Run 4)
- **Major changes from V6:**
  - **Action space redesign:** Replaced pre-ranked combos with learned card selection via intent + subset sampling
  - **Card quality reward:** +2.0 × (played_score / best_possible_score) — gave PPO direct signal for card selection
  - **Synergy/coherence rewards:** 164 jokers tagged by strategy, slot-scaled synergy bonuses, ante-aware scaling decay, per-blind and episode-end coherence bonuses
  - **Auto-positioning:** Blueprint/Brainstorm/Ceremonial Dagger auto-placed optimally on purchase
  - **Smart sell rewards:** Context-aware (sacrificial jokers +2.0, weak jokers +0.3, accumulated scaling joker -2.0)
- **What the agent learned:** Strategic card play, discard timing (~30% of actions), slot filling (96% full loadouts)
- **What it didn't learn:** Conditional strategy, multi-step joker planning, deviation from Green+Space
- **Why it plateaued:** Green+Space are legitimately S-tier. The agent's only decision branch was "buy them if seen" — no backup for alternative strategies. PPO averaged rewards across games couldn't discover multi-step coordinated builds.

### V5 — Dual-Agent Failure ([design](results/V5_DESIGN_NOTES.md) · [runs](results/V5_RUN_LOG.md))

- **Architecture:** Split play + shop into separate networks with 32-dim communication vector
- **Result:** 0% win rate, 12 failed runs
- **Why it failed:** Shop starvation — the shop agent received <0.1% of training steps. Every fix for shop exposure degraded play. Dual-agent is the wrong abstraction when play/shop ratio is 20:1.

### V1–V4 — Live Game to Python Sim

- **V1-V3:** Trained against live Balatro via Lua mod (file IPC → socket IPC). RAM degradation at 8 parallel instances killed long runs. Peak ~14 sps.
- **V4:** Pivoted to pure Python simulation. ~1000 sps (50,000x vs V1). All subsequent versions build on this sim.

---

## Major Takeaways

### Architecture
- **Don't split what doesn't need splitting.** V5's dual-agent created more problems (starvation, credit assignment, entropy) than the single-agent issues it tried to solve.
- **Pre-ranking actions matters enormously.** V4 used actual `score_hand()` ordering — ante 9 reached by iteration 2 instead of 460. But pre-ranking also became V6's ceiling (see V7).
- **Hierarchical actions can work** when factored log_prob is set up correctly (V7's intent + subset).

### Simulation Fidelity
- **Fix the fundamentals first.** V5's combo scoring bug caused 10 failed runs before being discovered. One broken line masked every other improvement.
- **Audit every joker.** 30% of V6 jokers had implementation bugs. The winning strategy (Burglar+Green+Space producing 612M chips) was entirely a training artifact.
- **Seed diversity prevents memorization.** Fixed seeds = the agent memorizes specific games. Random seeds force generalization.

### Training
- **Reward magnitude matters per-episode.** V7 Run 3's synergy rewards fired too rarely (~3-5 per episode) to shape behavior — needs ~15-25% of mean reward to reliably influence policy.
- **PPO has sticky local optima.** V7's Green+Space lock-in survived 6 different reward restructurings. Once the policy converges, reward shaping alone can't escape it.
- **Sharp distributions from pretrained weights don't transfer well to self-play.** V8 Runs 1-2 showed this — V7's tight distributions produced symmetric play + catastrophic unlearning when hit with new rewards.
- **Observation must contain actionable context.** V8 Runs 1-2 penalized life loss but the agent couldn't see lives → pure reward signal with no observable context → meaningless. Run 3 added MP state to obs for conditioning.

### Self-Play (V8)
- **Same-seed comparison is strong signal.** When both policies face identical game state and one wins, that's cleaner gradient than averaged rewards across different games.
- **Symmetric self-play collapses to draws.** Two copies of a sharp policy on same seed produce identical actions. Temperature asymmetry or diverse initialization required.
- **Multiplayer reward structure shifts optimal strategy.** V7's Green+Space scaling build is less valuable in PvP (which rewards burst scoring on a single hand). Self-play creates pressure for alternative builds naturally.

---

## Simulation

The Python sim (`balatro_sim/`) reimplements Balatro's full ruleset:

| Module | Contents |
|--------|----------|
| `game.py` (661 lines) | Full game loop — 5 states, 15 boss blind effects, ante progression, win/loss |
| `mp_game.py` (new V8) | Multiplayer coordinator: lives, PvP resolution, comeback money |
| `scoring.py` (142 lines) | Chip/mult computation, enhancements, editions, seals, retriggers |
| `hand_eval.py` (172 lines) | All 12 hand types including Flush Five / Flush House |
| `jokers/` (6 modules) | **164 jokers**, fully audited |
| `consumables.py` (549 lines) | 12 planets, 22 tarots, 18 spectrals, 27 vouchers |
| `shop.py` | Weighted rarity rolls, buy/sell/reroll, banned joker filter (V8), 150+ item catalogue |
| `card.py` | Card representation with enhancement/edition/seal |
| `card_selection.py` (V7) | Subset enumeration, intent-to-action translation |
| `synergy.py` (V7) | Joker synergy tags, coherence scoring, auto-positioning |

Test suite: **496 tests passing** across joker behavior, scoring, boss blinds, game transitions, consumables, environment integration, multiplayer mechanics, and banned-joker filtering.

---

## Repository Layout

```
balatro-rl/
├── train_v8.py               # V8 training — self-play multiplayer PPO (current)
├── train_v7.py               # V7 training — hierarchical intent + learned card selection
├── train_sim.py              # V6 training — single-agent PPO with pre-ranked combos
├── train_v5.py               # V5 training — dual-agent PPO (deprecated)
├── balatro_sim/              # Python Balatro simulation
│   ├── env_mp.py             # V8 multiplayer environment (438 obs, self-play)
│   ├── mp_game.py            # V8 multiplayer coordinator
│   ├── env_v7.py             # V7 environment (434 obs, hierarchical actions)
│   ├── env_sim.py            # V6 environment (402 obs, 46 actions)
│   ├── env_v5.py             # V5 dual-agent env (deprecated)
│   ├── card_selection.py     # V7 subset enumeration
│   ├── synergy.py            # V7 joker synergy + auto-positioning
│   ├── game.py               # Core game loop
│   ├── shop.py, scoring.py, hand_eval.py, consumables.py, card.py, constants.py
│   └── jokers/               # 164 jokers across 6 modules
├── tests/                    # Integration tests (496 passing)
│   ├── test_env_mp.py        # V8 MP environment tests
│   ├── test_mp_game.py       # V8 MP coordinator tests
│   ├── test_mp_integration.py # V8 full-game integration
│   ├── test_env_v7.py        # V7 environment tests
│   ├── test_card_selection.py
│   └── test_*.py             # Joker + game flow tests
├── scripts/                  # Plotting and analysis dashboards
├── results/                  # Design notes + run logs (detailed per-version)
│   ├── V8_DESIGN_NOTES.md    # V8 architecture spec
│   ├── V8_RUN_LOG.md         # V8 run journal
│   ├── MULTIPLAYER_RULESET.md # Standard Ranked spec from balatromp.com
│   ├── V7_PLANNING.md        # V7 design (5 approaches analyzed)
│   ├── V7_RUN_LOG.md         # V7 run journal (6 runs, 2.35% peak)
│   ├── V6_DESIGN_NOTES.md, V6_RUN_LOG.md
│   ├── V5_DESIGN_NOTES.md, V5_RUN_LOG.md
│   └── V4_DESIGN_NOTES.md, V3_DESIGN_NOTES.md, V1/V2 results
├── checkpoints_v8/, checkpoints_v7/, checkpoints_sim/, ... # Model weights per version
├── logs_v8/, logs_v7/, logs_sim/    # Training logs per version
├── legacy/                   # V1-V3 training scripts
├── mod/, mod_v2/             # Lua mod files (V1-V3)
└── launch/                   # PowerShell scripts (V3 instance management)
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
# Run tests (496 passing)
python -m pytest balatro_sim/jokers/tests/ tests/ -v

# Train V8 (current — self-play multiplayer, fresh start)
python train_v8.py --workers 20 --steps-per-worker 1024 --iterations 1000

# Resume V8 from checkpoint
python train_v8.py --resume checkpoints_v8/iter_0100.pt

# Train V7 (hierarchical intent + learned card selection)
python train_v7.py --workers 16 --steps-per-worker 2048 --iterations 1000

# Train V7 with V6 weight migration (warm start)
python train_v7.py --workers 16 --steps-per-worker 2048 --iterations 1000 \
    --migrate-v6 checkpoints_sim/iter_0280.pt

# Train V6 (pre-ranked combos)
python train_sim.py --workers 16 --steps-per-worker 2048 --iterations 1000

# Smaller batch for smoke testing
python train_v8.py --workers 4 --steps-per-worker 256 --iterations 10

# Benchmark sim throughput
python -m balatro_sim.env_sim
```

---

## Acknowledgments

Built on [Steamodded](https://github.com/Steamopollys/Steamodded) for Lua mod loading and [Handy](https://github.com/nicholassam6425/balatro-mods) for speed control (V1-V3). The V8 multiplayer ruleset is based on [balatromp.com](https://balatromp.com/)'s Standard Ranked specification.
