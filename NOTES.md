# Balatro RL — Project Notes

## Overview
Reinforcement learning agent (PPO via Stable-Baselines3) trained to play Balatro via live file-based IPC.
- Lua mod writes `state.json` after every game tick; Python reads it and writes `action.json`
- Agent controls card play/discard only (119-dim obs, MultiBinary(9) action)
- All navigation (blind select, cash out, shop, new run) handled headlessly by Lua mod

---

## Architecture

### IPC Protocol
- Lua writes: `AppData/Roaming/Balatro/balatro_rl/state.json` (every 100ms poll)
- Python writes: `AppData/Roaming/Balatro/balatro_rl/action.json` (consumed by Lua)
- Log: `AppData/Roaming/Balatro/balatro_rl/log.txt`

### Observation Space (119 features)
- `[0:56]`  Hand cards — 8 slots × 7 features (rank, suit, enhanced, edition, seal, highlighted, exists)
- `[56:86]` Jokers — 5 slots × 6 features
- `[86:95]` Scalar state — 9 features (ante, round, score_progress, hands_left, discards_left, money, joker_slots, deck_remaining, discard_count)
- `[95:119]` Hand levels — 12 types × 2 (level, chips)

### Action Space (MultiBinary(9))
- `action[0:8]` — which hand slots to highlight
- `action[8]`   — 1=play, 0=discard
- Cap at 5 cards; force play if discards=0

### PPO Hyperparameters
- `n_steps=512`, `batch_size=64`, `n_epochs=10`
- Net arch: [256, 256, 128], actor-critic MlpPolicy
- ~150k params

### Reward Structure
- `R_BLIND_COMPLETE = 1.0`  — cleared a blind
- `R_ANTE_COMPLETE  = 3.0`  — cleared all 3 blinds in an ante
- `R_WIN            = 10.0` — completed ante 8
- `R_LOSE           = -2.0` — ran out of hands
- `R_SCORE_PROGRESS = 0.05` — per 1% of score target gained
- Hand quality bonus on play: pair=0.10 → flush_five=1.10 (teaches hand selection early)

### Lua Nav (headless)
All nav is handled by `BalatroRL.lua` via `G.E_MANAGER` event queue:
- **Blind select**: 0.5s delay → queues `ease_round` + `new_round` events; uses `G.P_BLINDS[blind_choices[bod]]` for correct blind object
- **Cash out**: 1.0s delay → calls `G.FUNCS.cash_out({config={button="cash_out"}})`
- **Shop**: 0.8s delay → buy affordable jokers (tracks remaining $ and slots locally), then 0.5s → `G.FUNCS.toggle_shop`
- **New run**: 0.5s delay → calls `G.FUNCS.start_run(nil, {stake=1})`

### Shop Joker Buying
- Iterates `G.shop_jokers.cards`, filters to `card.ability.set == 'Joker'` only
- Tracks `remaining_dollars` and `slots_used` locally (game state updates are queued, not instant)
- Buys as many affordable jokers as fit in slots
- Controlled by `buy_jokers` config flag in `metadata.json`

---

## Training Progress (as of 2026-03-19)

### Current results (~700 episodes, ~110k timesteps)
| Metric | Value |
|--------|-------|
| Best reward | 5.72 (ep 444, ante 2) |
| Ante 2+ reached | 1 time |
| Avg reward trend | -1.48 → -0.75 (improving) |
| PPO updates | ~114 |
| Gradient steps | ~9,120 |

### What the agent has learned
- **Discarding heavily**: best run had 52 discards vs 11 plays — correctly learned that fishing for better hands before playing is worth it
- **Playing more cards per hand**: avg 4.6–6.0 cards per play in best runs (approaching 5-card hands)
- **Not yet**: flush/straight detection, joker synergies, deck building awareness

### Reward trend (current run)
```
ep   0- 99: avg=-1.477
ep 100-199: avg=-1.374
ep 200-299: avg=-1.020
ep 300-399: avg=-1.009
ep 400-499: avg=-0.788   ← ante 2 hit here
ep 500-599: avg=-0.799
ep 600-699: avg=-0.772
ep 700-799: avg=-0.752   ← current
```

---

## Known Limitations

### What's hard about Balatro for RL
1. **Non-Markovian deck** — deck composition matters enormously; agent doesn't see what cards remain
2. **Joker synergies** — 200+ jokers with combinatorial interactions; current 6-feature joker encoding barely captures what a joker does
3. **Long-horizon credit assignment** — joker bought in ante 1 shop pays off in ante 6; hundreds of steps of delay
4. **No memory across rounds** — agent has no recurrent state; can't track deck depletion or plan ahead
5. **Partial information** — doesn't see opponent blinds' chip targets ahead of time

### Realistic ceiling with current setup
- Reliable ante 3-4 play with enough training (~1M+ timesteps)
- Learning hand hierarchy (pairs → straights → flushes)
- Not expected to develop joker synergy strategies without architectural changes

---

## Future Development Ideas

### Near-term (tractable)
0. **Deck composition in obs — do this before next long training run (requires fresh start)**
   - Add remaining deck counts to obs: rank counts (13) + suit counts (4) = **17 new features**
   - OBS_SIZE: 119 → 136; PPO input layer auto-resizes, existing checkpoints incompatible (start fresh)
   - **Lua side**: iterate `G.deck.cards` each poll tick, count rank/suit, write as two arrays to state.json
   - **Python side**: parse in `state.py`, append 17 features after existing 119, update OBS_SIZE constant
   - Why useful: lets agent reason about flush/straight odds, avoid fishing for already-seen ranks
   - Option A (52 features, full bitmap) is lossless but larger; Option C (17 features) is sweet spot

1. **Score simulation in obs (Option 2)**: Have Lua calculate the predicted chip×mult score for each candidate hand subset and include it in state.json. Agent sees "playing these 5 cards scores 847 chips" directly — collapses most of card-selection into a lookup. Removes need to learn hand hierarchy from scratch.

2. **Joker categorization**: Tag each joker as one of:
   - Scoring jokers (multiply chips/mult directly)
   - Deck-fixing jokers (improve card quality, e.g. add enhancements)
   - Economy jokers (generate money per hand/discard)
   Add category + rarity (Common/Uncommon/Rare/Legendary) as obs features. Rarity weights already in game data.

3. **Temporal joker value**: Economy/deck-fix jokers have higher value early (ante 1-2); scoring jokers dominate late (ante 6+). This is almost impossible to learn from sparse rewards alone — **reward shaping** could bootstrap it: small bonus for buying economy joker in early antes.

4. **Richer deck obs**: Add remaining deck composition (13 ranks × 4 suits = 52 binary features) so agent can reason about flush/straight probability.

### Medium-term
5. **Recurrent policy (LSTM)**: Replace MlpPolicy with RecurrentPPO (SB3-contrib). Gives agent memory across rounds — can track deck state, plan joker purchases.

6. **Parallel training**: SB3 supports `SubprocVecEnv` for N parallel environments. Would need N Balatro instances. Real unlock is **headless simulation** — strip Balatro to pure Lua state machine (no LÖVE rendering) and run thousands of episodes/minute.

7. **Curriculum learning**: Start on easy blinds only (Small Blind ante 1), gradually unlock harder content as agent achieves milestones. Prevents agent from spending 90% of training on losing runs.

### Long-term / research-level
8. **Imitation learning (behavioral cloning)**: Pre-train policy on high-level gameplay. YouTube has good Balatro content, but video → game state extraction is a major sub-project. Would need either a replay system or a computer vision pipeline to extract card states from video frames. AlphaStar used this approach with StarCraft replays.

9. **World model / score predictor**: Train a separate small net to predict score given hand + jokers, use embeddings as policy input. Effectively learns the Balatro scoring engine from data.

10. **Self-play / MCTS**: Monte Carlo tree search over possible card plays with rollouts. More compute-intensive but principled.

---

## Technical Notes

### Key Balatro API facts
- `G.STATES`: SELECTING_HAND=1, HAND_PLAYED=2, DRAW_TO_HAND=3, GAME_OVER=4, SHOP=5, BLIND_SELECT=7, ROUND_EVAL=8, NEW_ROUND=19
- `G.blind_on_deck`: "Small"/"Big"/"Boss" — current blind position
- `G.GAME.round_resets.blind_choices`: maps "Small"→"bl_small" (string key into G.P_BLINDS)
- `G.shop_jokers.cards`: list of joker cards currently in shop
- `card.ability.set == 'Joker'`: correct filter for jokers (vs consumables, vouchers)
- `ease_dollars(-cost)` is called inside `buy_from_shop` — don't call it separately
- `G.CONTROLLER.locks.toggle_shop` must be cleared after toggle_shop or subsequent calls may be blocked

### SB3 / training gotchas
- Always `python train.py --resume` — never start fresh mid-training (discards weights)
- `hand_levels` in state.json is a list, not dict — `_parse_hand_levels()` converts to keyed dict
- `MultiBinary(9)` actions stored as `.tolist()` in best_runs.jsonl
- Monitor CSV only tracks current session; episode_log.jsonl accumulates across restarts

### Files
| File | Purpose |
|------|---------|
| `BalatroRL.lua` | Lua mod — state writer + headless nav + action executor |
| `balatro_rl/state.py` | State parsing, observation encoding |
| `balatro_rl/action.py` | Action writing, agent output → card indices |
| `balatro_rl/env.py` | Gymnasium environment |
| `train.py` | PPO training loop with BestRunCallback |
| `analyze.py` | Episode log analysis |
| `test_shop.py` | Watch shop nav in real-time |
| `plot_training.py` | Reward curve visualization |
| `versions/` | Archived mod versions (lua_nav, no_nav) |
| `logs/episode_log.jsonl` | Per-episode stats (all runs) |
| `logs/best_runs.jsonl` | Full action sequences of best episodes |
| `checkpoints/` | PPO model checkpoints |
