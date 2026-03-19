# Balatro RL

A reinforcement learning agent that plays [Balatro](https://www.playbalatro.com/) using PPO (Proximal Policy Optimization) via a custom Gymnasium environment and live file-based IPC with a Lua game mod.

> **Disclaimer:** Balatro is a product of LocalThunk/Playstack. This is an unofficial mod for research and educational purposes only.

---

## Overview

The agent controls card play and discard decisions in real time. All game navigation (blind selection, cash out, shop, new run) is handled headlessly by the Lua mod — no mouse or keyboard automation required.

**Training runs at 32× game speed** using the [Handy](https://github.com/nicholassam6425/balatro-mods) mod, with animation skip enabled. The agent plays continuously without human input.

---

## Architecture

### IPC Protocol

Communication between Python and the game uses two JSON files polled at 100ms:

```
Balatro/balatro_rl/state.json   ← Lua writes game state every tick
Balatro/balatro_rl/action.json  ← Python writes actions; Lua consumes and deletes
```

The Lua mod hooks into `Game.update` (survives run resets, unlike `love.update`) and drives all navigation via `G.E_MANAGER` event queues — the same mechanism Balatro uses internally.

### Observation Space (119 features)

| Range   | Description |
|---------|-------------|
| `[0:56]`   | Hand cards — 8 slots × 7 features (rank, suit, enhanced, edition, seal, highlighted, exists) |
| `[56:86]`  | Jokers — 5 slots × 6 features (type, sell value, rarity, buy price, effect, exists) |
| `[86:95]`  | Scalar state — ante, round, score progress, hands left, discards left, money, joker slots, deck remaining, discard count |
| `[95:119]` | Hand levels — 12 hand types × 2 (level, chips) |

### Action Space

`MultiBinary(9)`:
- `action[0:8]` — which of the 8 hand slots to highlight
- `action[8]` — 1 = play, 0 = discard
- Caps at 5 cards selected; forces play if discards = 0

### Reward Structure

| Signal | Value |
|--------|-------|
| Blind cleared | +1.0 |
| Ante completed (all 3 blinds) | +3.0 |
| Game won (ante 8 cleared) | +10.0 |
| Lost (hands exhausted) | −2.0 |
| Score progress (per 1% of target) | +0.05 |
| Hand quality bonus on play | +0.10 to +1.10 |

Hand quality bonuses range from +0.10 (pair) to +1.10 (flush five), providing dense early signal before the agent learns to clear blinds.

### PPO Hyperparameters

- Policy: `MlpPolicy`, net arch `[256, 256, 128]`, ~150k params
- `n_steps=512`, `batch_size=64`, `n_epochs=10`
- Trained with Stable-Baselines3

---

## Project Structure

```
balatro-rl/
├── balatro_rl/
│   ├── env.py          # Gymnasium environment
│   ├── state.py        # State parsing from state.json
│   ├── action.py       # Action writing to action.json
│   └── nav.py          # Nav utilities (legacy, now handled by Lua)
├── train.py            # PPO training script with checkpoint + best-run callbacks
├── analyze.py          # Episode log analysis (reward trends, ante distribution)
├── check_progress.py   # Training progress summary
├── plot_training.py    # Reward/episode plots
├── test_loop.py        # Sanity check: observe → act → observe loop
├── logs/               # Episode logs, monitor CSV, training plots
├── checkpoints/        # PPO checkpoint zips
├── versions/           # Archived Lua mod versions
└── NOTES.md            # Detailed architecture and dev notes

Mods/BalatroRL/
├── BalatroRL.lua       # Game mod: state writer + nav controller
└── metadata.json       # Mod config (lua_nav, buy_jokers, debug_log flags)
```

---

## Setup

### Requirements

- Balatro (Steam)
- [Steamodded](https://github.com/Steamopollys/Steamodded) mod loader
- [Handy](https://github.com/nicholassam6425/balatro-mods) mod (for speed + animation skip)
- Python 3.10+

```bash
pip install stable-baselines3 gymnasium torch numpy
```

### Install the Lua Mod

Copy `Mods/BalatroRL/` to your Balatro mods directory:
- Windows: `%AppData%\Balatro\Mods\BalatroRL\`

### Train

```bash
# Fresh run
python train.py

# Resume from latest checkpoint
python train.py --resume
```

### Monitor

```bash
python check_progress.py   # Episode summary + ante distribution
python plot_training.py    # Save training curve to logs/training_progress.png
python analyze.py          # Reward trends by bin
```

---

## Training Progress

As of ~700 episodes / 110k timesteps:

- **Best run:** Ante 4 (cleared 9 blinds), reward 6.08
- **Reward trend:** −1.48 → −0.75 over training
- Agent has discovered discard-heavy play (52 discards vs 11 plays in best run)
- Has not yet learned flush/straight detection or joker synergies

**Planned improvements:**
- Add deck composition to observation (remaining rank/suit counts, +17 features)
- Score simulation: Lua calculates predicted chip×mult for candidate hands
- Joker categorization (scoring / deck-fixing / economy) as obs features

---

## How It Works

1. Balatro launches with the BalatroRL Lua mod active
2. The mod writes game state to `state.json` every 100ms
3. Python reads the state, runs the PPO policy, and writes `action.json`
4. Lua consumes the action, highlights cards, and calls `G.FUNCS.play_cards_from_highlighted` or `G.FUNCS.discard_cards_from_highlighted`
5. After each blind, Lua automatically navigates: cash out → buy jokers → leave shop → select next blind → new run on game over
6. Python's `BalatroEnv` wraps this loop as a standard Gymnasium `step/reset` interface

The key challenge is that Balatro's game engine is event-driven and runs at variable speed. The IPC protocol handles timing edge cases (state written before action consumed, run reset races, boss blind initialization order) that required careful engineering to make robust.

---

## Limitations

- **Single game instance:** No parallel environments; one Balatro window
- **No game state rollback:** Env resets require a full new run
- **Partial observability:** Agent doesn't see upcoming blinds, full deck composition (yet), or joker synergy details
- **Speed cap:** Currently 32× via Handy; further speedup possible via `love.draw` override

---

## Acknowledgments

Built on top of [Steamodded](https://github.com/Steamopollys/Steamodded) for Lua mod loading and [Handy](https://github.com/nicholassam6425/balatro-mods) for speed control.
