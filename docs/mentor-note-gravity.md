# Mentor note — root cause of the rollout failure + one added ablation arm

*(Draft to send. Context: written after your ablation plan; adds the diagnosis
we found while implementing it.)*

---

Thanks — I've adopted the controlled ablation exactly as you laid out (fixed
architecture/dataset/seeds/optimizer/eval-clips; report one-step accel error plus
rollout position-vs-time, velocity error, energy drift, stable-horizon, and
failure mode, separately for drape and collision; pushforward starting at 2–5
steps and ramping; with implementation diagnostics for both the input
perturbations and the pushforward gradient flow). Documented the 100-vs-1,500
comparison first, before touching the trainer.

While implementing it I ran the noise-only arm and did a deeper diagnostic, and
I think we've found the specific mechanism — which refines (rather than
contradicts) the exposure-bias picture and suggests one extra arm.

**Noise-only arm:** no rollout improvement, and one-step got worse. More
importantly, rollout is **invariant to one-step accuracy** across all three
models so far (one-step 0.09 → 0.60, rollout l2\_final ≈ 0.62 for all). Random
error compounding would predict better rollout from better one-step, so the
drift looks **systematic**, not stochastic.

**What the systematic error is:** the rolled-out cloth **does not fall under
gravity**. The drift is 100% vertical, spatially uniform, and equals ½·g·t² — the
predicted cloth hovers at its start height while the reference free-falls. It is
**not** a cold-start artifact (warm-starting from mid-fall frames does not fix
it; the cloth brakes to a near-stop regardless of initial velocity).

**Mechanism (quantitative):** the acceleration target's vertical std is
**28.7 m/s²**, dominated by the rare, large accelerations at contact. Against
that scale, gravity (−9.81) is only **−0.31 in normalized units**, and the mean
is **−0.86 m/s²**. In rollout, on its own slightly-off states, the model
regresses toward that near-zero mean → effective gravity ≈ **9%** → it falls
~11× too slowly. This is exactly one item on your "deeper issues" list —
**whether acceleration-only supervision is sufficient** — and it cleanly explains
why dataset size, noise, and warm-starting all changed nothing.

**Proposed adjustment:** keep your 2×2 ablation as the backbone (it's the right
way to attribute the effect of exposing the model to its own states), and add a
**fifth arm — a gravity-residual target**: the network predicts `a − g`
(internal forces) and we add `g` explicitly in the integrator. That makes
free-fall exact by construction, so it directly tests whether the diagnosed
normalization pathology is the binding constraint. It's cheap (~$13) and
complements pushforward — if pushforward alone also removes the ceiling, we learn
exposure was sufficient; if only the residual does, we learn the target
parameterization was the issue; if both are needed, we learn both.

Everything (the four evals, per-axis drift, warm-start, and normalization
numbers) is in `docs/rollout-scaling-finding.md`. Happy to proceed with the
five-arm ablation on the fixed drape1500 setup and report the full metric suite
for drape and collision. Flagging one data limit: we have 3,720 drape but only
141 collision clips, so I'd train fixed on drape1500 and evaluate rollout on
held-out drape **and** the 141 collision clips (collision-out-of-training) for
the per-scenario report, and defer the balanced set until the training fix lands.
