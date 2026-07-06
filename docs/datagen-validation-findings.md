# Dataset-generation validation findings (CPU pre-GPU pass)

Goal: shake out drift/crashes on CPU before spending money on the full 10k-clip
GPU run. Three issues were found; two are fixed, one is a platform limitation.

## ① `dx_m: null` crash in the full path — FIXED

`configs/mpm.yaml` ships `mpm.dx_m`/`inv_dx` as `null`. `load_mpm_config()` fills
them, but only when a config *path* is passed; the generators pass a *dict*
(`run_one_clip → cfg_for_clip → MPMClothSim(cfg)`), so the full (non-smoke) path
hit `float(None)` and crashed on clip #1. The `--smoke` branch happened to set
them, which is why smoke worked and masked the bug. Shared by both
`generate_dataset.py` and `generate_dataset_modal.py`, so the GPU run would have
died immediately.

Fix: `cfg_for_clip()` now derives `dx_m`/`inv_dx` from `grid_resolution` for
both smoke and full paths.

## ② Wind scenario blows up at full resolution — NOT FIXED (deep issue)

A sheet pinned at only its top corners under any lateral wind eventually drives a
membrane element into inversion; the explicit co-rotational force spikes, velocity
diverges, a particle is ejected out of the grid → hard SIGBUS/SIGSEGV (uncatchable
by Python). Two fix approaches were tried and **both failed**; the honest state is
that this needs a substantial solver change.

What was ruled out (data, so nobody re-derives it):
- **Wind magnitude is irrelevant.** Undamped onset is at a fixed *physical* time
  (~86 ms); 1 N and 4 N both crash ~step 850. Capping the sampler does nothing.
- **Pin layout is irrelevant.** 2 corners, full top edge, and corner patches all
  crash (full edge even earlier).
- **`dt` only delays.** `dt=5e-5` moves the crash from 86 ms to ~250 ms.
- **Extra material damping** does nothing.
- **Air drag only delays** (this was the big one). Linear drag `v*=exp(-k·dt)`
  merely postpones inversion, delay ∝ k: blow-up at step 860 (k=0) → 957 (k=8) →
  1145 (k=20) → 1796 (k=50). At k=50, `vmax` reached a *perfect plateau* (0.218
  flat, steps 900–1700) and **still inverted at 1796** — because terminal velocity
  is nonzero, so the sheet keeps drifting/rotating about its pins and accumulating
  deformation until an element inverts. No k stabilizes a full 10000-step clip.
- **It is not the autodiff-through-`ti.svd` gradient.** Replacing the SVD rotation
  extraction with a regularized closed-form 2×2 polar decomposition produced a
  byte-identical blow-up. Regularizing the rotation *magnitude* does not help; the
  model has no inversion *handling* (sign), and the explicit integration diverges
  once an element inverts.

Also surfaced by instrumenting the failure: the per-particle **3×3 `F` label is
already garbage** (det ≈ ±10²⁹) ~40 steps *before* the velocity diverges, while
positions/forces still look normal. This is a codimensional-MPM artifact — for a
thin 2D sheet the APIC update `F=∏(I+dt·C)·F` is unconstrained in the out-of-plane
normal direction and drifts exponentially. So even a "stabilized" wind clip would
ship a corrupt `F` label.

A real fix is multi-day solver work: proper codimensional/membrane MPM, inversion
handling (invertible-elasticity with sign flip, likely analytical forces not
autodiff), and probably implicit integration. Pragmatic alternatives: generate
wind clips with the existing `ImplicitClothSim` (whole-scenario, not mid-clip, so
labels stay self-consistent), or drop the wind scenario. **Decision deferred.**

## ③ Full-res crashes on ALL backends — contact-induced, not a platform bug

**Corrected twice.** First mislabelled "arm64-only" (disproven: a Modal GPU run
crashed 6/6 with `CUDA_ERROR_ILLEGAL_ADDRESS`), then provisionally called "element
inversion" (also disproven — see below). Full-resolution (64×64 cloth / 128³ grid)
crashes on **both** CPU (SIGBUS) and CUDA (illegal address) for **every** scenario.
16×16 is stable on both; 32×32 and 64×64 are not.

**Actual mechanism: contact-induced explicit-integration instability.** Full-res
drape is stable in free-fall (`vmax` grows smoothly as g·t) and blows up **exactly
at sphere contact** (~t=0.3 s). At the blow-up the culprit element is *at the sphere
surface*, stretched σ_max≈3.2 with **σ_min≈0.98 — no inversion, no degeneracy**;
`vmax` then grows exponentially (~1.24×/step) until a particle leaves the grid → OOB
access. It is **not** inversion, **not** the autodiff-`ti.svd` reverse-grad (a
closed-form polar gave an identical crash), **not** global CFL/`dt` (halving `dt`
only delays to the same physical contact time), and **not** fixed by coarsening to
32×32. Best-supported cause: the hard grid-velocity contact projection creating a
sharp velocity discontinuity a fine explicit mesh can't absorb.

The earlier "drape/collision are stable at full res" claim was under-tested (a
probe stopped at 1500 steps, before the ~3000-step contact crash). They are NOT
stable at 64×64.

Consequence: **the production 64×64 config cannot generate any scenario with the
current explicit solver, on any backend.** Full diagnostic journey + candidate
fixes: `docs/full-res-instability-investigation.md`. **Paused for advisor input.**

## Tooling

`scripts/validate_edge_cases.py` — subprocess-isolated edge-case harness (its
isolation is why it survived the hard crashes and logged them as failures rather
than dying). Checks NaN/Inf, the solver's `is_unstable` flag, pinned-corner drift,
drape descent, energy blow-up, and cross-process determinism. Run `--quick` for a
fast 16×16 self-test; the full-res mode is limited by ③.

## Current tree state

- **① `dx_m` fix**: kept (in `scripts/generate_dataset.py`; all 42 tests pass;
  smoke regenerated clean). Unblocks the full path from crashing on clip #1.
- **② wind + ③ full-res inversion**: no solver code change kept. Drag, closed-form
  polar, and `dt` reduction were all tried and reverted; `src/mpm_cloth.py` and
  `configs/mpm.yaml` are at baseline. These are the SAME instability and are being
  addressed by the solver fix (`docs/full-res-instability-investigation.md`).
- **Modal path**: ported to Modal 1.x + system libs (works; validated the 16×16
  smoke and surfaced the full-res crash on CUDA).
- **Tooling**: `scripts/validate_edge_cases.py` harness kept.

## Recommended next step

**Paused for advisor input on the solver.** The blocker is the contact-induced
full-res instability (③); the leading fix is contact robustness (soft/penalty +
damping instead of hard velocity projection), with implicit integration and
resolution/`dt` changes as alternatives. See `docs/full-res-instability-investigation.md`
for the evidence, the four disproven hypotheses, and the candidate directions.

Note: the committed generator default (drape + collision, wind deferred) is not yet
viable at 64×64 — drape/collision also hit ③. Full-dataset generation is blocked on
the solver decision. The 16×16 smoke path and all 42 tests remain green.
