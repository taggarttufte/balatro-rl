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

## Run 1 — Phase 1 Baseline (V7 Migration)

**Config:** 20 workers, 1024 steps/worker, 1000 iters, V7 Run 4 migration

**Reward changes from V7:**
- All V7 rewards preserved (card quality, synergy, coherence, sell logic)
- NEW: PvP win +3.0, PvP loss -2.0
- NEW: Life lost (any cause) -1.5
- NEW: Game win +20.0, Game loss -10.0
- House rule: regular blind failure now costs a life (vs V7's "lose game on
  failure")

**Goals:**
1. Validate self-play mechanics work end-to-end at scale
2. See if same-seed comparison signal pulls policy away from Green+Space
3. Measure baseline metrics for Phase 2 comparison

**Status:** Started 2026-04-13, in progress

**Early metrics (to be filled in):**

| Iter | Reward (P1/P2) | Wins (P1/P2/Draw) | Best Ante | Eps |
|------|---------------|---------------------|-----------|-----|

**Observations during run:** *(to be filled in)*

**Final results:** *(to be filled in after completion)*

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
