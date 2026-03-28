# Balatro RL

A reinforcement learning agent that plays [Balatro](https://www.playbalatro.com/) using PPO trained via a custom Gymnasium environment and live IPC with a Lua game mod.

> **Disclaimer:** Balatro is a product of LocalThunk/Playstack. This is an unofficial mod for research and educational purposes only.

---

## What is Balatro?

[Balatro](https://www.playbalatro.com/) is a roguelike deckbuilder where you play poker hands to score points.

**Gameplay loop:**
1. Play 5-card poker hands to score chips against escalating point thresholds ("blinds")
2. Clear 3 blinds per ante (8 antes total to win)
3. Between blinds, visit the shop to buy "jokers" — modifiers that multiply your scoring
4. Joker synergies are the core strategic layer: the right combo can score millions from a simple pair

**Why it's an interesting RL problem:**

| Challenge | Description |
|-----------|-------------|
| Combinatorial action space | 8 cards → 218 possible play/discard combinations per turn |
| Long-horizon credit assignment | A joker bought in ante 1 might only pay off in ante 6 |
| Partial observability | Agent can't see upcoming blinds, full deck order, or future shop contents |
| Sparse rewards | Only hands that clear blinds produce meaningful reward |
| Stochastic elements | Card draws, joker spawns, boss blind effects are all randomized |
| Compounding strategy | Hand upgrades, deck thinning, and joker combos interact multiplicatively |

A skilled human wins ~70% of runs. Random play wins <0.01%.

---

## Architecture (Current: V3)

### IPC Protocol — Socket-based

Python and Lua communicate over TCP sockets (IPv6, loopback):
- Lua acts as **server** (one port per instance: 5001-5008)
- Python acts as **client**, connects once per env on startup
- Messages are newline-delimited JSON
- State is sent on meaningful events only (G.STATE transitions, discard detection, connect)

```
Lua game loop:
  G.STATE changes → send {"event": "selecting_hand", ...}
  discards_left decreases → send {"event": "selecting_hand", ...}
  on connect → send current state

Python env:
  reset() → send "new_run", wait for "selecting_hand"
  step()  → send action, wait for next actionable state
```

### Observation Space (206 features)

| Range | Description |
|-------|-------------|
| `[0:40]` | Top 10 play options × 4 features (chips, mult, hand_type_id, num_cards) |
| `[40:70]` | Top 10 discard options × 3 features (expected score delta, target hand type, num_cards) |
| `[70:83]` | Hand cards — 8 slots × (rank, suit, exists) |
| `[83:113]` | Jokers — 5 slots × 6 features (type, sell value, rarity, buy price, effect, exists) |
| `[113:122]` | Scalar state — ante, round, score_progress, hands_left, discards_left, money, joker_slots, deck_remaining, discard_count |
| `[122:146]` | Hand levels — 12 types × 2 (level, chips) |
| `[146:163]` | Deck composition — 13 rank counts + 4 suit counts |

### Action Space — `Discrete(20)`

```
[0-9]   → play one of top 10 pre-ranked hands (sorted by chip×mult, computed by Lua)
[10-19] → discard one of top 10 pre-ranked card subsets (sorted by draw potential)
```

Lua enumerates all ~218 card combinations each step, ranks them, and returns the top 10 of each type in the state JSON. Agent picks one index. Eliminates the need to learn card combination evaluation from scratch.

### Reward Structure

| Signal | Value |
|--------|-------|
| Blind cleared | +1.0 |
| Ante completed (all 3 blinds) | +3.0 |
| Game won (ante 8) | +10.0 |
| Lost (hands exhausted) | -2.0 |
| Score progress (per 1% of target) | +0.05 |
| Discard quality bonus | delta best_play_score |

### Training — Custom Threaded PPO (V3)

8 threads, one per Balatro instance. Python's GIL releases during socket `recv()`, so
threads genuinely overlap while waiting for game responses:

```
Thread 1: Env 1 (port 5001) ─┐
Thread 2: Env 2 (port 5002) ─┤
...                            ├→ shared experience queue → PPO update (PyTorch)
Thread 8: Env 8 (port 5008) ─┘
```

Expected throughput: ~90 steps/sec (8 × 11.7 steps/sec per instance at 128x game speed).

---

## Results Summary

| Version | Episodes | Timesteps | Ante 2+ Rate | Best Ante |
|---------|----------|-----------|--------------|-----------|
| V1 (MultiBinary, single) | 26,412 | 376k | 1.89% | 5 |
| V2 single (Discrete20, single) | 9,073 | ~210k | **60.4%** at 8.7k eps | 9 |
| V2 parallel (Ray, 8-env) | ~41k steps | ~41k | — (paused early) | 5 by ep 279 |
| V3 (socket, custom loop) | in progress | — | — | — |

See `V1_RESULTS.md`, `V2_SINGLE_RESULTS.md`, `V2_PARALLEL_RESULTS.md` for full details.

---

## Setup

### Requirements

- Balatro (Steam)
- [Steamodded](https://github.com/Steamopollys/Steamodded) mod loader
- [Handy](https://github.com/nicholassam6425/balatro-mods) mod (speed control + animation skip)
- Python 3.10+

```bash
pip install torch numpy gymnasium
```

### Install the Lua Mod

Copy `Mods/BalatroRL_v3/` to:
```
%AppData%\Balatro\Mods\BalatroRL_v3\
```

### Launch Instances

```powershell
# Launch all 8 instances (instances 2-8 from C:\BalatroParallel\)
.\launch_parallel.ps1
```

### Train (V3)

```bash
# Start training (8 envs, 500 iterations)
python train_v3.py --envs 8 --iterations 500
```

### Monitor

```bash
python check_progress.py    # Episode summary + ante distribution
python analyze_recent.py    # Rolling window analysis
```

---

## Project Layout

```
balatro-rl/
├── balatro_rl/
│   ├── env_socket.py       # V3 Gymnasium env (socket IPC)
│   ├── env_parallel.py     # V2 env (file IPC, 8 instances) — reference
│   ├── env_v2.py           # V2 env (file IPC, single) — reference
│   ├── env.py              # V1 env — obsolete
│   ├── state_v2.py         # State parser (206-feature obs)
│   ├── action_v2.py        # Action space (Discrete 20)
│   └── ray/                # Ray wrappers (abandoned)
├── train_v3.py             # V3 training script (custom threaded PPO)
├── train_ray_socket.py     # Ray attempt (abandoned, kept for reference)
├── train_ray.py            # V2 parallel Ray training
├── train_v2.py             # V2 single SB3 training
├── launch_parallel.ps1     # Launch all 8 Balatro instances
├── logs_v2/                # V2 single training logs + checkpoints
├── logs_ray/               # V2 parallel Ray logs
├── logs_ray_socket/        # V3 socket logs
├── checkpoints_v2/         # V2 single model weights (165MB)
├── checkpoints_ray/        # V2 parallel Ray checkpoints
├── V1_RESULTS.md           # V1 final results
├── V2_SINGLE_RESULTS.md    # V2 single training results
├── V2_PARALLEL_RESULTS.md  # V2 parallel (Ray) training results
├── V3_DESIGN_NOTES.md      # V3 architecture: socket IPC + custom loop
├── V4_DESIGN_NOTES.md      # V4 plan: dual shop agent
├── NOTES.md                # Full dev log
└── PARALLEL_SETUP.md       # Parallel instance setup guide

Mods (installed in Balatro):
  %AppData%\Balatro\Mods\BalatroRL_v3\BalatroRL_v3.lua   ← active (all instances)
  %AppData%\Balatro\Mods\BalatroRL\BalatroRL.lua         ← V2 mod (inactive)
```

---

## Acknowledgments

Built on [Steamodded](https://github.com/Steamopollys/Steamodded) for Lua mod loading
and [Handy](https://github.com/nicholassam6425/balatro-mods) for speed control.
