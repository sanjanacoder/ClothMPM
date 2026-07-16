# Mentor note — the rollout "ceiling" was an evaluation bug, not a model limit

*(Draft to send. Supersedes the earlier exposure-bias framing.)*

---

Important update before we run the ablation — I found the actual cause of the
rollout failure, and it changes the conclusion.

**The rollout collapse was an evaluation-harness bug: the model's velocity
history was being seeded incorrectly.** Our model infers acceleration largely
from the *slope* of the recent velocities (the C-frame history). The rollout's
`reset()` seeded a **flat** history (it repeated the initial velocity), which has
zero slope — so the model saw a physically meaningless input and predicted
roughly zero (even slightly upward) acceleration. That is why the "cloth doesn't
fall" and why every model plateaued at the same ~100 ms horizon.

**Direct evidence.** Querying the trained model on real reference states:

| frame | true a_y | pred a_y (true history) | pred a_y (flat history = the bug) |
|---|---|---|---|
| 5 | −9.81 | **−9.79** | +0.98 |
| 30 | −9.81 | **−9.80** | +1.48 |
| 60 | −9.81 | **−9.81** | +2.34 |
| 100 | −9.81 | **−9.81** | +3.35 |

With the true history the model predicts gravity to two decimals; with the flat
(seeded) history it predicts *upward*.

**Effect of the fix.** Seeding the true C-frame history (same model, same
held-out clips):

| seeding | positional L2 @ 200 steps | L2 @ ~395 steps |
|---|---|---|
| flat (bug) | 0.190 m | 0.591 m |
| true history (fixed) | **0.009 m** | **0.186 m** |

**~21× less drift at 200 steps** — the cloth now tracks the reference to ~9 mm
out to 200 ms.

**And the key reversal: with correct seeding, dataset scaling *does* help
rollout — the opposite of what the broken harness showed.** Same held-out clips,
corrected eval:

| model | horizon | L2 @ 200 | L2_final | energy drift |
|---|---|---|---|---|
| pilot (100 clips) | 120 ms | 0.242 | 2.30 | 2785 (unstable) |
| scaled (1,500 clips) | 236 ms | 0.023 | 0.249 | 45 |

The seeding bug flattened this (both models "hovered," so both read ~0.62
regardless of quality). Fixed, the 30× model has ~10× lower drift, ~2× longer
horizon, ~60× less energy blow-up. So **more data substantially improves
rollout, and the balanced 10k is re-justified.** The earlier "scaling doesn't
help rollout" was an artifact of the bug.

**What's actually left.** The scaled model still drifts ~0.25 m and energy-drifts
~45× by ~400 steps, concentrated near contact (~300 ms). That is the real —
and much smaller — remaining challenge, where more data and the
pushforward/short-unroll training should help. I've kept pushforward implemented
and verified (the diagnostic confirming noise reaches the inputs and gradients
flow through the unroll still passes).

**Proposed next step.** Rather than the full 5-arm noise/pushforward ablation
against a bug, I'd:
1. lock in the corrected evaluation (done; regression test added), re-report the
   full metric suite (position/velocity/energy/horizon vs time) for the pilot and
   scaled models on the fixed harness, and
2. run a *targeted* pushforward experiment on the residual late-horizon/contact
   drift only — which is now the real open question — and, if it helps, proceed
   to the balanced 10k with the corrected setup.

I'm sorry for the earlier over-confident diagnosis; the free model-query
diagnostic you'd implicitly asked for (verify perturbations/gradients) is what
surfaced this. Full details, evals, and the fix are in
`docs/rollout-scaling-finding.md`.
