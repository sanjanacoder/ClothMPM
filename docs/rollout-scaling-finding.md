# Rollout bottleneck: scaling finding + root-cause diagnosis

Fixed across everything below: GNN architecture (847k params, message-passing
L=5), `x, v → a` (no F), optimizer, seeds, and the evaluation clips (held-out
drape seeds 2000/2500/3000, never in any training set). Rollout = pure-neural
autoregressive (`scripts/eval_rollout.py`), 400 steps, semi-implicit Euler.

## TL;DR

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
