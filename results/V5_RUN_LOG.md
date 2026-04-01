# V5 Training Run Log

Tracks each training run, what changed, what we learned, and what to try next.

V5 = dual-agent PPO: Play agent (374-dim obs) + Shop agent (188-dim obs).
Shop agent produces a 32-dim comm vector fed into play agent obs.

---

## Run 1 (Phase A — frozen play, train shop)
**Config:** Phase A, 500 iters, 16 workers, 4096 steps/iter, play frozen (run 6 weights)
**Result:** Win rate 0%, shop loss 0.0 by iter 31, best ante 1 throughout
**What happened:** Shop steps collapsed to 0 within 30 iters. Frozen play agent (run 6) survives ante 1 long enough that each episode takes hundreds of steps — the 512-step truncation cap means episodes almost never reach the shop naturally.
**Lesson:** Phase A doesn't work. Frozen play + truncation = shop starvation. The better the play agent, the worse the starvation.

---

## Run 2 (Phase B — joint, pretrained play)
**Config:** Phase B, 1000 iters, resume-play=iter_1000.pt (run 6), both unfrozen
**Crashed:** NaN logits in shop agent PPO update
**What happened:** Embed layer shape mismatch (342 v4 vs 374 v5). Also all-False mask rows slipping into PPO minibatch → all-inf logits → NaN.
**Fix:** Resize embed weight (copy 342 cols, zero-pad 32 new comm dims), strict=False load, nan_to_num on logits, safety fallback on masks.

---

## Run 3 (Phase B — joint, pretrained play, NaN fixed)
**Config:** Phase B, 1000 iters, resume-play=iter_1000.pt, 512-step truncation
**Crashed at iter ~1484:** Same NaN crash — guards in forward() not catching it during PPO backward pass.
**What happened:** Advantage/return normalization can produce inf when rollout is near-empty. NaN propagates through backward.
**Fix:** nan_to_num on adv_b and ret_b before PPO update.

---

## Run 4 (Phase B — joint, pretrained play, all NaN fixed)
**Config:** Phase B, 1000 iters, resume-play=iter_1000.pt, 512-step truncation, ENTROPY_COEFF=0.01
**Result:** Play entropy collapsed to 0.002 by iter ~40. Shop still starved (~5 steps/iter). No wins.
**What happened:** Pretrained play weights + low entropy coeff → play agent converges to deterministic policy almost immediately. Shop learns nothing because (a) no data and (b) play dying too fast to see joker benefits.
**Fix needed:** Higher play entropy coeff.

---

## Run 5 (Phase B — joint, pretrained play, entropy fixed)
**Config:** Phase B, 1000 iters, resume-play=iter_1000.pt, 4000-step truncation, ENTROPY_COEFF_PLAY=0.05
**Crashed at iter ~1106:** NaN again, different source. s_loss going negative then exploding.
**What happened:** With 4000-step truncation, episodes occasionally produce extreme advantage values when combined with sparse shop rewards.
**Lesson:** 4000-step truncation helps shop exposure slightly but doesn't solve starvation.

---

## Run 6 (Phase B — joint, pretrained play, full NaN hardening)
**Config:** Phase B, 1000 iters, resume-play=iter_1000.pt, 4000-step truncation, ENTROPY_COEFF_PLAY=0.05, nan_to_num everywhere
**Result:** Play entropy collapsed to ~0.002 again by iter 40 despite 0.05 coeff. Shop entropy 0.01. ~5 shop steps/iter.
**What happened:** Pretrained play agent dominates. Even with high entropy coeff it collapses because the comm vector input (32 dims, all zero) confuses it early. Shop starved throughout.
**Key insight:** Pretrained play agent = worse starvation, not better. Longer survival = fewer shop resets per batch.

---

## Run 7 (Phase B — joint, pretrained play, entropy 0.05)
**Config:** Phase B, 1000 iters, resume-play=iter_1000.pt, MAX_EP_STEPS=4000, ENTROPY_COEFF_PLAY=0.05
**Status:** Survived to iter ~1342 before system reboot. Play entropy healthy (1.1-1.4). Shop entropy low (~0.3-0.5). Shop steps ~5-100/iter.
**Result:** Reward -1.5 → +0.35. No wins. Shop still mostly "leave immediately."
**Lesson:** Pretrained weights confirmed to cause starvation. Starting fresh is better.

---

## Run 8 (Phase B — both from scratch)
**Config:** Phase B, 1000 iters, NO resume, both agents init from scratch, MAX_EP_STEPS=4000
**Result:** 841 iters completed. Reward -1.18 → +0.08. Play entropy 1.6 (healthy). Shop entropy 0.25 (collapsed). Shop steps ~77/iter by iter 841.
**What happened:** Starting fresh gave much better early shop exposure (600-1300 steps/iter early on) confirming the pretrained-agent starvation hypothesis. But as the play agent learned, shop steps dried up to same starvation pattern.
**Shop strategy converged to:** ~80% leave immediately. Credit assignment gap too large to connect joker purchases to downstream outcomes.
**Key insight:** Both-from-scratch is better for early training. Starvation is structural — it's the game's ratio of play-to-shop steps, not a hyperparameter problem.

---

## Run 9 (Phase B — scratch, leave penalty, higher shop entropy) ← CURRENT
**Config:** Phase B, 1000 iters, scratch, MAX_EP_STEPS=4000, ENTROPY_COEFF=0.03 (shop), ENTROPY_COEFF_PLAY=0.05, R_LEAVE_SHOP=-0.1
**Status:** ~230 iters in. Reward -1.18 → -0.23. Play entropy 1.6. Shop entropy 0.35-0.55 (better than run 8).
**Shop action dist at iter 220:** leave=0.43, use0=0.35, reroll=0.07 — leave penalty working, shop not immediately leaving but pivoted to consumable use instead of buying.
**Observations:** Leave probability staying ~0.43 instead of climbing to 0.8+. Leave penalty is working. But shop still not buying jokers meaningfully.
**Next changes:** Bump R_QUALITY_SCALE (0.2 → 0.5), add R_SPEND reward for dollar spending.

---

## Structural Problems (carry forward to all runs)

1. **Shop starvation:** Play/shop step ratio ~20:1 or worse as play improves. Shop agent gets insufficient training data. Options: replay buffer, dedicated shop workers, guaranteed min shop steps per iter.

2. **Credit assignment gap:** Joker purchased at step 50 pays off at step 400+. PPO's GAE can't trace this. Options: reward shaping (quality delta), intrinsic reward for spending.

3. **Entropy collapse:** Shop agent collapses to deterministic policy before getting meaningful signal. Fixed partially with ENTROPY_COEFF=0.03 and leave penalty.

4. **Shop "leave bias":** Leaving is always safe (zero cost), buying has immediate cost and delayed reward. Agent learns leave=safe by default. Fixed partially with R_LEAVE_SHOP=-0.1.

---

## Planned Next Changes
- [ ] Bump R_QUALITY_SCALE: 0.2 → 0.5
- [ ] Add R_SPEND: small reward proportional to dollars spent (0.05 * dollars_spent)
- [ ] Implement shop replay buffer (keep last N shop experiences, sample for PPO updates)
- [ ] Add more shop epochs (N_SHOP_EPOCHS=30 vs play's 10)
