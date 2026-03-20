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

## Future Plans

### V2 — Richer Obs + Pre-ranked Action Space (next major version)

The current model selects cards by setting 9 binary bits, with no knowledge of what score each combination produces or what cards remain in the deck. V2 addresses both.

**Action space: `Discrete(20)`** — replaces `MultiBinary(9)`
- `[0–9]` — play one of the top 10 pre-ranked hands (sorted by chip×mult, computed by Lua)
- `[10–19]` — discard one of the top 10 pre-ranked card subsets (sorted by draw potential)
- Lua enumerates all ~218 card combinations each step and returns the best options; <1ms overhead

**Obs additions (~206 total features, up from 119):**
- 10 play options × (chips, mult, hand_type, num_cards) = +40
- 10 discard options × (expected score delta, target hand type) = +30
- Deck composition: 13 rank counts + 4 suit counts = +17

**Why this matters:** The agent goes from guessing which 9 bits produce a good hand to choosing from a ranked menu of options. This directly encodes the 3-step human decision loop:
1. What is the best hand I can play right now?
2. Does playing it sacrifice future value given what's left in the deck?
3. If not worth playing — what to discard to maximise my next draw?

**Curriculum learning via action masking:** Start training with all 20 options unmasked. As the policy converges, progressively mask lower-ranked options to focus fine-tuning on near-optimal decisions. Same network weights throughout — no architecture change needed.

---

### V3 — Dedicated Shop Agent with Bidirectional Communication

V2 keeps a hand-coded Lua shop heuristic. V3 replaces it with a learned shop policy that communicates with the playing agent.

**Two specialised agents:**

*Playing agent* — same architecture as V2, handles SELECTING_HAND phase.

*Shop agent* — new, handles SHOP phase. Observes:
- Money + interest threshold (how close to next $1 bracket)
- Current joker loadout (rarity, edition, category)
- Ante, round, and upcoming boss blind type
- Available skip tags (e.g. free joker tag = don't spend, save for reward)
- Hand level distribution and scoring history
- Shop contents: joker type, cost, rarity, edition

**Bidirectional communication channel:**

The two agents pass latent vectors to each other across phases:

```
Playing agent hidden state → strategy_out (16d) → Shop agent obs
Shop agent hidden state    → loadout_out  (16d) → Playing agent obs
```

`strategy_out` encodes what the playing agent has learned this run — which hands are landing, what the deck is trending toward. The shop agent uses this to decide what to buy.

`loadout_out` encodes what the shop agent just acquired — which joker categories and synergies are now available. The playing agent uses this to adjust its hand selection and discard strategy immediately after the shop.

**The callout analogy:**
Think of the two agents as teammates developing a shared vocabulary. The 16-float vectors are the callouts being made in real time — different content every game. The encoder weights are the shared language that gives those callouts meaning — learned across millions of games and fixed once trained.

Early in training the vectors are noise and agents ignore them, acting only on local observations. Over time the reward signal forces a shared vocabulary to emerge: coordinated plays produce wins, ignored communication doesn't. The language that develops is whatever encoding happened to be most useful for winning — you don't design it, you create conditions for it to develop.

*Simple version (recommended first):* replace learned latent vectors with human-designed structured signals — hand level distribution passed to the shop agent, joker category counts passed back to the playing agent. Same information, trains much faster.

*Complex version:* fully learned bidirectional latent vectors. Requires joint gradient updates through both networks, 4–8 parallel Balatro instances (WSL2/Xvfb), and millions of episodes to converge. Genuinely research-level.

---

### Near-term improvements (V1.x, no architecture change)
- **Deck composition in obs** — remaining rank/suit counts (+17 features); enables flush/straight planning without a full architecture restart
- **Joker-aware obs** — tag each joker as scoring / deck-fixing / economy; rarity and edition already partially encoded
- **Shop sell-upgrade** — sell weakest held joker for a strictly higher-rarity shop joker; edition scoring (Poly > Holo > Foil) and Negative joker protection already implemented

### Training speedup
- **Speed mode** — override `love.draw = function() end`, zero all event delays, remove FPS cap; estimated 20–50× additional speedup on top of the current 32×
- **Parallel training via WSL2/Xvfb** — headless Balatro instances in virtual displays; most practical path to vectorised environments; mandatory for V3 learned communication
- **Lua scoring engine reimplementation** — reimplement Balatro's ~1,000-line scoring engine outside the game loop; thousands of episodes per minute with no game window

### Learning approaches
- **Imitation learning** — behavioural cloning from recorded high-level gameplay; strong starting policy before RL fine-tuning; bottleneck is video-to-state extraction
- **Recurrent policy (LSTM)** — RecurrentPPO (SB3-contrib); agent memory across rounds for implicit deck tracking without explicit deck features

## Limitations

- **Single game instance:** No parallel environments; one Balatro window
- **No game state rollback:** Env resets require a full new run
- **Partial observability:** Agent doesn't see upcoming blinds, full deck composition (yet), or joker synergy details
- **Speed cap:** Currently 32× via Handy; further speedup possible via `love.draw` override

---

## Acknowledgments

Built on top of [Steamodded](https://github.com/Steamopollys/Steamodded) for Lua mod loading and [Handy](https://github.com/nicholassam6425/balatro-mods) for speed control.
