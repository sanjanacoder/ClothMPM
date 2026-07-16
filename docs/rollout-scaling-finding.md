# Rollout is the bottleneck, not data — finding + proposed experiment

## TL;DR

Scaling the dataset **30×** made the GNN's **single-step** prediction ~3× more
accurate but did **not** improve **rollout** at all — the same result on
held-out clips (so it is not overfitting and not a generalization gap). The
~100 ms rollout horizon is a **structural** limit of how we train, not a data
limit. Root cause, now pinned in the code: **the trainer does pure one-step
teacher forcing** — the two standard rollout-stabilizers (training-noise
injection and pushforward/unrolled training) are specified in the config but
**never wired into the training loop**. Proposed next step: implement them and
re-measure rollout, on the data we already have (cheap). This applies to
**collision as well** (same compounding mechanism, likely worse due to contact).

## One-step vs rollout (the distinction that matters)

- **One-step**: given the *true* current frame, predict the next frame. The
  model always starts from a perfect snapshot. Easy.
- **Rollout**: feed the model its *own* predictions, frame after frame (here,
  400 steps). No ground truth to correct it. Small per-step errors **compound**
  — the model drifts into states it never saw, gets less sure, errs more.

A learned simulator is only useful if *rollout* is stable; one-step accuracy is
necessary but not sufficient.

## The experiment

Trained the same GNN (x, v → a, no F) at two dataset sizes and evaluated rollout
on drape clips, **including held-out clips neither model was trained on**
(seeds 2000/2500/3000).

| model | clips | one-step val accel-L2 | rollout l2\@200 | rollout l2\_final (400) | horizon | energy |
|---|---|---|---|---|---|---|
| pilot | 100 | 0.288 | 0.19 | 0.63 (held-out) | 100 ms | 0.99 |
| scaled | 1500 (**30×**) | **0.091** | 0.19 | **0.62** (held-out) | 100 ms | 1.00 |

In-sample vs held-out, both models: rollout l2\_final ≈ **0.62 everywhere**.

**Reading:**
- One-step accuracy scales well (3× better with 30× data).
- Rollout does **not** move — same horizon, drift, and energy.
- Held-out = in-sample → **not overfitting**; pilot held-out = scaled held-out →
  **not a generalization gap data closes**. The ceiling is structural.

## Root cause (pinned in the code)

`src/train.py` is **one-step teacher forcing only**:
- The loss is `mse_loss(pred, target)` on single steps; the `horizon`
  ("rollout curriculum") is computed and logged but **never used**.
- The `random_walk` **training noise** in the config is **never applied**.

So the model only ever practices predicting from *perfect* frames and never
practices recovering from its *own* drift. This is textbook exposure bias — and
it is exactly what makes rollout fail and what makes more data irrelevant to
rollout. The two standard fixes (below) were planned (config + a docstring note
"wires up in M3/W6") but not implemented.

## Proposed experiment: teach drift-recovery

Wire in the two established techniques (GNS / MeshGraphNets) and re-measure
rollout on the existing data — no new dataset needed:

1. **Training-noise injection** (cheap, ~10 lines): perturb the input state with
   small random-walk noise during training so the model learns to correct
   states that are slightly off. Sweep the noise scale (config already has
   `sigma_position_m = 3e-3` as a starting point).
2. **Pushforward / short unrolled training** (moderate): unroll the model a few
   steps, feed its own predictions back, and penalize the accumulated drift so
   it practices multi-step stability. Ramp the horizon toward the ~100-step
   range where rollout currently fails (the existing curriculum schedule can
   drive this once it's actually applied).

Success metric: rollout horizon / l2\_final on the same held-out clips. Cost:
~$13 per training run on the data we already have (no 10k needed to test this).

## Implications for collision

The compounding mechanism is **universal** — it applies to collision too, so
scaling data will not fix collision rollout either. Collision is likely
**harder**: contact is a sharp, abrupt event (velocities flip almost
instantly), which is the hardest moment to predict and injects more error per
step. The pilot already hinted at this (collision drifted more than drape:
l2\_final ~0.70–0.78 vs ~0.58–0.64). So collision **needs** drift-recovery
training even more than drape. (Caveat: collision's data-scaling behavior is
untested; the core rollout ceiling should still hold, but contact dynamics are
different enough to warrant a check once the training fix is in.)

## Recommendation on the 10k

The 10k's original purpose — "does scale extend the rollout horizon?" — is now
answered **no** for drape (in-sample and held-out). Its residual value is only
collision-at-scale + dataset completeness, not the scaling question. Better to
**spend the next ~$13 implementing drift-recovery training** and re-measuring
rollout than ~$81 confirming a scaling result we can now predict. If
drift-recovery extends the horizon, generate the balanced 10k *then*, to train
the improved model at scale.
