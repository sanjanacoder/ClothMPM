# Mentor note — rollout resolved: a seeding bug + pushforward, and scaling helps

*(Draft to send. Combines the seeding-bug discovery, the scaling reversal, and
the pushforward result.)*

---

Three connected results that resolve the rollout story.

**1. The "~100 ms rollout ceiling" was an evaluation-harness bug, not a model or
training limitation.** The rollout seeded a *flat* velocity history (`reset()`
repeated the initial velocity). The model infers acceleration from the velocity
*slope*, so a flat history made it predict ≈0 (even upward) acceleration — the
cloth "didn't fall." Querying the trained model on real states, it predicts
gravity to two decimals (−9.79…−9.81); on the flat history it predicts +0.98…
+3.35. Seeding the true C-frame history drops rollout drift ~21× at 200 steps
(0.190 → 0.009 m, same model, held-out clips). So the earlier "scaling/noise
don't help rollout" conclusions were all measured through this bug.

**2. With the corrected harness, dataset scaling clearly helps rollout — the
opposite of the buggy conclusion — which re-justifies the 10k.** Held-out drape:

| model | horizon | L2 @ 200 | L2_final | energy drift |
|---|---|---|---|---|
| pilot (100 clips) | 120 ms | 0.242 | 2.30 | 2785 (unstable) |
| scaled (1,500 clips) | 236 ms | 0.023 | 0.249 | 45 |

The bug had flattened both models to ~0.62 (both "hovering"), hiding the gap.
30× data → ~10× less drift, ~2× longer horizon, ~60× less energy blow-up.

**3. Your pushforward arm works — it fixes the residual contact instability.**
Same scaled setup with short-unroll training (curriculum ramped K to 5, per your
"start 2–5 and increase"), evaluated on the corrected harness / same held-out
clips:

| scaled GNN | L2_final (400) | energy drift |
|---|---|---|
| no pushforward | 0.249 | 45.1 |
| + pushforward (K→5) | 0.219 | **0.92** |

Energy drift collapses **45× → ~0.9** (near-perfect conservation); late positional
drift improves ~12%. The early rollout (horizon, L2@200) is unchanged because the
seeding fix already made it accurate — pushforward specifically stabilizes the
*late/contact* phase, which is exactly the residual. And this is conservative:
only K→5 and *half* the training frames (100k vs 200k), so larger unroll and full
data should do more. (Implementation verified: the diagnostic confirms
perturbations reach the inputs and gradients flow through every unroll step.)

**Net.** Two independent fixes, each for a different part of the rollout: the
seeding fix for the early free-fall phase, pushforward for the late/contact
stability. Together the model now tracks accurately *and* conserves energy, and
scaling helps. I'd propose: (1) generate the balanced 10k (now justified),
(2) train the GNN with the corrected setup (real-history eval, pushforward),
ramping the unroll horizon a bit further, and (3) proceed to the complexity
detector / hybrid fallback (M4) — the residual contact frames are exactly what
that stage is designed to catch.

Full details, evals, per-axis drift, and the regression test are in
`docs/rollout-scaling-finding.md`.
