# Rollout bottleneck: scaling finding + root-cause diagnosis

> **CORRECTION (supersedes the analysis below).** The rollout collapse was an
> **evaluation-harness bug**, not a model/data/training limitation. The rollout
> seeded a *flat* velocity history (`reset()` repeated `v0`); the model infers
> acceleration from the velocity *slope*, so a flat history made it predict ≈0
> (even upward) acceleration → the cloth "didn't fall." With the true C-frame
> history seeded (`eval_rollout.py`, `HybridRollout.reset(v_history=...)`), the
> **same** model predicts gravity to 2 decimals and rollout drift drops ~21× at
> 200 steps (0.190 → 0.009 m). So the "scaling/noise don't help rollout" and
> "gravity-normalization" conclusions below were all measured through the bug and
> do **not** hold. The model has been fine throughout. The genuine remaining
> issue is a much smaller *residual* late-rollout drift near contact (~0.19 m by
> ~395 steps) — see the "Corrected results" section. The diagnostic chain below
> is retained as the record of how we got here.

Fixed across everything below: GNN architecture (847k params, message-passing
L=5), `x, v → a` (no F), optimizer, seeds, and the evaluation clips (held-out
drape seeds 2000/2500/3000, never in any training set). Rollout = pure-neural
autoregressive (`scripts/eval_rollout.py`), 400 steps, semi-implicit Euler.

## Corrected results (the seeding bug)

Model queried on real reference states — true history vs the flat history the
buggy `reset()` produced:

| frame | true a_y | pred a_y (true history) | pred a_y (flat history = bug) |
|---|---|---|---|
| 5 | −9.81 | −9.79 | +0.98 |
| 30 | −9.81 | −9.80 | +1.48 |
| 60 | −9.81 | −9.81 | +2.34 |
| 100 | −9.81 | −9.81 | +3.35 |

Same model + held-out clips, rollout drift by seeding:

| seeding | L2 @ 200 steps | L2 @ ~395 steps |
|---|---|---|
| flat (bug) | 0.190 m | 0.591 m |
| true history (fixed) | **0.009 m** | **0.186 m** |

**And with correct seeding, dataset scaling DOES help rollout (the opposite of
the buggy conclusion below).** Held-out drape, corrected harness:

| model | horizon (≤5 cm) | L2 @ 200 | L2_final (400) | energy drift |
|---|---|---|---|---|
| pilot (100 clips) | 120 ms | 0.242 | 2.30 | 2785 (unstable) |
| scaled (1,500 clips, 30×) | **236 ms** | **0.023** | **0.249** | **45** |

The seeding bug had *flattened* this comparison (both models "hovered," so both
read ~0.62 regardless of quality). Fixed, the 30× model is ~10× lower drift,
~2× longer horizon, ~60× less energy blow-up — so **more data substantially
improves rollout, and the balanced 10k is justified**. Residual issues remain
(scaled still drifts ~0.25 m and energy-drifts 45× by ~400 steps, concentrated
near contact) — that is the real, smaller open problem, addressed next.

### Pushforward fixes the residual contact instability

Same scaled setup + short-unroll (pushforward) training, curriculum ramping the
unroll horizon K to 5 (`--window 5`), corrected harness / same held-out clips:

| scaled GNN | L2_final (400) | energy drift |
|---|---|---|
| no pushforward | 0.249 | 45.1 |
| + pushforward (K→5) | **0.219** | **0.92** |

Energy drift collapses **45× → ~0.9** (near-perfect conservation), late drift
−12%. Early rollout (horizon 236 ms, L2@200 0.023) unchanged — the seeding fix
already made that accurate; pushforward stabilizes the *late/contact* phase.
Conservative result (K→5, half the frames), so more unroll + full data should do
more. Implemented in `src/train.py` (`--window`, `_unroll`); verified by
`scripts/verify_pushforward.py` (perturbations reach inputs; gradients flow
through every unroll step).

Locked in: `HybridRollout.reset(v_history=...)`, `eval_rollout.py` seeds the true
history and starts at frame C-1, and `tests/test_hybrid.py::
test_reset_seeds_velocity_history` asserts the slope-dependence.

## TL;DR (as originally written — now known to be measured through the bug)

1. **Scaling the dataset 30× (100 → 1,500 drape clips) improved one-step
   accuracy ~3× but did not change rollout at all** — same horizon, positional
   drift, and energy, **including on held-out clips**. So the bottleneck is not
   dataset size and not overfitting.
2. **Training-noise injection (GNS-style) did not help rollout either** (and
   hurt one-step). Rollout is **invariant to one-step accuracy** across all
   models → the drift is **systematic, not stochastic error compounding**.
3. **Diagnosis (root cause found):** the rolled-out cloth **does not fall under
   gravity** — the drift is 100% vertical, spatially uniform, and equals
   ½·g·t². Cause: the acceleration target's vertical std is **contact-dominated
   (28.7 m/s²)**, so gravity (−9.81) is a **weak normalized signal (−0.31)** and
   the mean is only **−0.86 m/s²**. In rollout the model regresses toward that
   near-zero mean → effective gravity ≈ **9%** → the cloth hovers.
4. This is a **target/normalization pathology**, i.e. *acceleration-only
   supervision is insufficient* — consistent with exposure bias but more
   specific. It predicts the ablation outcome and suggests an extra fix
   (gravity-residual target).

## 1. Scaling experiment (100 vs 1,500 clips)

| model | clips | one-step val accel-L2 | rollout l2\@200 | rollout l2\_final (400) | horizon (≤5cm) | energy drift |
|---|---|---|---|---|---|---|
| pilot | 100 | 0.288 | 0.19 | **0.63** | 100 ms | 0.99 |
| scaled | 1,500 (**30×**) | **0.091** | 0.19 | **0.62** | 100 ms | 1.00 |

Evaluated on held-out clips (seeds 2000/2500/3000). In-sample vs held-out was
also identical for both models (l2\_final ≈ 0.62 everywhere), so it is **not
overfitting** and **not a generalization gap that data closes**.

## 2. Training-noise arm (already run)

Same setup + GNS random-walk velocity noise (σ≈1e-2, ~4% velocity perturbation;
input-only — the exact `a -= n/dt` correction blows up ~1000× with accel targets
at dt=1e-3, so we widen the input distribution instead).

| model | one-step accel-L2 | held-out rollout l2\_final |
|---|---|---|
| scaled, no noise | 0.091 | 0.625 |
| scaled + noise | 0.60 | **0.627** |

Noise made one-step 6× worse and left rollout unchanged.

## 3. Why it's systematic, not compounding

| model | one-step accel-L2 | rollout l2\_final |
|---|---|---|
| pilot | 0.288 | 0.63 |
| scaled | 0.091 | 0.62 |
| scaled + noise | 0.60 | 0.63 |

Rollout is **the same regardless of one-step accuracy** (0.09 → 0.60). Random
error compounding would predict *better* rollout from *better* one-step. It
doesn't happen → the drift is a deterministic, systematic error every model
shares.

## 4. Root cause — the model doesn't free-fall

Per-axis drift of the scaled model on held-out seed 2000 (RMS over particles):

| frame | RMS (x, y, z) | mean-signed (x, y, z) | pred mean-y | ref mean-y |
|---|---|---|---|---|
| 21 | (0.000, 0.002, 0.000) | (+0.000, +0.002, +0.000) | +0.772 | +0.770 |
| 51 | (0.000, 0.013, 0.000) | (+0.000, +0.013, +0.000) | +0.772 | +0.760 |
| 101 | (0.000, 0.050, 0.000) | (+0.000, +0.050, +0.000) | +0.772 | +0.722 |
| 141 | (0.000, 0.098, 0.000) | (+0.000, +0.098, +0.000) | +0.772 | +0.675 |

- Drift is **entirely vertical** (x, z ≈ 0); RMS = mean-signed → a **uniform**
  offset (every particle, same amount).
- **pred-y is constant (+0.772)** while ref-y falls. The predicted cloth hovers;
  the true cloth free-falls. Drift at 0.14 s = 0.098 m ≈ ½·g·t² = 0.096 m.

**Not a cold-start artifact.** Warm-starting the rollout from mid-fall frames
(nonzero velocity) does not fix it — the cloth brakes to a near-stop regardless:

| start frame | v0-y | drift after 120 steps | pred-y (end) |
|---|---|---|---|
| 0 (rest) | −0.001 | 0.071 | +0.772 |
| 15 | −0.148 | 0.085 | +0.768 |
| 30 | −0.295 | 0.099 | +0.761 |
| 60 (falling fast) | −0.590 | 0.125 | +0.742 |

## 5. The mechanism (quantitative)

Normalization stats of the acceleration target:

```
target_mean = [0.0, -0.86, 0.0]  m/s^2      target_std = [~0, 28.66, ~0]  m/s^2
```

- `std_y = 28.7` is dominated by the **rare, huge accelerations at contact**
  (bounce). Free-fall (−9.81) normalizes to only **−0.31** — a weak signal.
- `mean_y = −0.86` (free-fall's −9.81 averaged with contact's large positives
  and post-contact settling ≈ 0).
- In rollout, on its own (slightly off) states, the model **regresses toward the
  mean** (≈0 normalized) → `a_y ≈ −0.86` = **9% of gravity** → falls ~11× too
  slowly → the observed vertical drift.

This is why data, noise, and warm-starting all did nothing: none of them
addresses a normalization that makes gravity a weak signal.

## 6. Proposed controlled ablation (per mentor)

Fixed architecture / dataset / seeds / optimizer / eval-clips. Report one-step
accel error **and** rollout position-vs-time, velocity error, energy drift,
stable-horizon at a fixed threshold, and failure mode — **separately for drape
and collision** (train on drape1500; evaluate held-out drape + the 141 collision
clips we have).

1. **Baseline** (one-step) — done.
2. **Noise injection only** — done (negative).
3. **Pushforward / short-unroll loss only** — 2→5 steps, ramp up (not ~100).
4. **Noise + pushforward.**
5. **Gravity-residual** (predict `a − g`, add `g` in integration) — direct fix
   for the diagnosed free-fall failure; complements pushforward.

Objective: does exposure to model-generated states (pushforward) shift the
~100 ms ceiling? If yes → scale the corrected setup to the balanced 10k. If no →
investigate deeper (state observability after removing F, integration scheme,
contact representation, message-passing depth, sufficiency of accel-only
supervision — the gravity-residual arm tests the last one directly).

Implementation is verified with diagnostics that (a) perturbations genuinely
enter the inputs and (b) gradients pass through the pushforward steps.
