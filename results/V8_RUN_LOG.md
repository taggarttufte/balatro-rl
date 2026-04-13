# V8 Training Run Log

V8 = self-play multiplayer Balatro. Two players (sharing one policy) face off
in head-to-head games on the same seed via `MultiplayerBalatroEnv`. See
[V8 Design Notes](V8_DESIGN_NOTES.md) for full architecture.

---

## Architecture Quick Reference

- **Network**: ActorCriticV7 (2.5M params, hierarchical intent + card scoring)
- **Env**: `MultiplayerBalatroEnv` with HOUSE RULE (regular blind failures cost lives)
- **4 lives** per player, **4 banned jokers** (Chicot, Matador, Mr. Bones, Luchador)
- **20 workers** × 1 MP env × 1024 steps × 2 agents = 40,960 agent-steps/iter
- **Self-play with shared policy** — stochastic sampling provides divergence
- **V7 weight migration** from Run 4 (best V7 model: 2.35% solo win rate)

---

## Run 1 — Phase 1 Baseline (V7 Migration) — **KILLED at iter 63**

**Config:** 20 workers, 1024 steps/worker, V7 Run 4 migration

**Observed problem: game termination bug + collapsing distributions**

At iter 63 (~55 min in, 2.6M steps):
- **84% of games ended at ante 2** (V7 regularly reached ante 9)
- **74% of games were draws** (both players identical strategies)
- **Avg game length: 16 MP steps** (normal Balatro is 150-300)
- Reward plateau: ~19-20 per player, flat
- Phase entropy collapsed to 0.22 (V7's was ~0.45)

**Root cause #1: Shop skip bug in revive logic**

`_revive_if_needed()` advanced failed players directly to the next BLIND_SELECT,
**skipping the shop visit entirely**. Without shop visits, failed players never
got to buy jokers. They'd fail → skip shop → fail next blind → skip shop → etc.
Both players burned 4 lives in ~4-5 blinds with 0 jokers, ending at ante 2.

**Root cause #2: Sharp policy distributions from V7 migration**

V7 had already sharpened intent/phase distributions to near-deterministic
(entropy 0.22 on phase actions). Both players sampling from same sharp
distributions produced near-identical actions → same outcomes → 74% draws.

Killed at iter 63 to fix both issues for Run 2.

---

## Run 2 — Shop Fix + Temperature Asymmetry + Look-Ahead (planned)

**Three changes from Run 1:**

1. **Shop fix**: `_revive_if_needed()` now sends failed players to SHOP (with
   fresh shop generation) instead of skipping to the next blind. Failed
   players can now buy jokers to recover.

2. **Temperature asymmetry**: P2 samples all actions (intent + subset + phase)
   with temperature `P2_TEMPERATURE = 1.4` — flatter distributions, more
   exploration. P1 samples at normal temp 1.0. Same policy weights, different
   sampling behavior → natural divergence without requiring separate networks.

3. **Look-ahead win evaluation**: When game ends with a winner, simulate the
   winner's next few blinds with greedy scripted policy:
   - If winner dies to small/big blind in next ante → `R_WIN_SHAKY = -10.0`
     applied to game-win reward (so effective win reward = +10.0)
   - If winner reaches next PvP blind intact → `R_WIN_STRONG = +5.0`
     (effective win reward = +25.0)
   - Separates "won because opponent collapsed" from "my strategy is solid"

**Also fixed in Run 2:**
- Double-resolution bug (`_is_blind_done` now tracks `_last_resolved` position
  to avoid re-counting SHOP state until player moves past it)

**Smoke test (2 workers, 128 steps, 2 iters):**
- Zero draws across 19 games (vs 74% in Run 1)
- Reached ante 4 in iter 2 (vs ante 2 ceiling in Run 1)
- Both players distinctly differentiated in rewards (P1=25.4, P2=21.1)
- 491 tests passing

**Config:** 20 workers, 1024 steps/worker, 1000 iters, V7 Run 4 migration

**Status:** Ready to launch

---

## Metrics To Track

Standard from V7:
- Avg reward per player per iteration
- Intent distribution (Play/Discard/Consumable)
- Best ante reached
- Episode count per iteration
- Joker count in winning loadouts
- Loadout coherence

NEW for V8:
- P1 wins / P2 wins / Draws per iteration (should hover near 50/50/X for
  pure self-play)
- Avg game length (in MP steps and antes)
- Lives remaining at game end (per winner)
- PvP blind win rate (per player)
- Strategy divergence between P1 and P2 in same-seed games

---

## Decision Points

- **Iter 100**: Early signal check. If reward and win patterns look identical
  to V7, we may need explicit policy diversification.
- **Iter 500**: Mid-run evaluation. Compare joker diversity and coherence
  against V7 baselines.
- **Iter 1000**: Full evaluation. Decide on Phase 2 (specialist population,
  new multiplayer jokers, Hall of Fame).

---

## Comparison Baseline

For reference, V7 final results (last 50 iters):

| Run | Win Rate | Avg Jokers | Coherence |
|:-:|:-:|:-:|:-:|
| V7 Run 4 | 2.35% | 4.0 | 0.65 |
| V7 Run 5 | 2.20% | 4.90 | 0.63 |
| V7 Run 6 | 2.23% | 3.33 | 0.58 |

V8 success = significantly exceeding these on a comparable solo win rate
metric (we'll need to evaluate V8's policy on single-player V7 env to
compare apples-to-apples).
