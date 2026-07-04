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

## ③ Full-res clips crash on macOS/arm64 CPU — platform limitation, NOT the sim

Even a physically stable full-res drape clip (no wind, healthy state) SIGSEGVs
nondeterministically (step ~1k–4k) inside Taichi's `launch_kernel`. It scales with
grid size — the 16×16 smoke path runs 10000 steps fine; 64×64/128³ does not. This
is a Taichi CPU/arm64 runtime issue, not the simulation.

Consequence: **full-resolution validation is not reliable on this Mac.** The
16×16 CPU path works and caught ① and ②. The definitive full-res, full-duration
check must run on the real target (CUDA) — i.e. the Modal GPU smoke test
(6 clips, ~$0.10), which also confirms whether ③ is arm64-only.

## Tooling

`scripts/validate_edge_cases.py` — subprocess-isolated edge-case harness (its
isolation is why it survived the hard crashes and logged them as failures rather
than dying). Checks NaN/Inf, the solver's `is_unstable` flag, pinned-corner drift,
drape descent, energy blow-up, and cross-process determinism. Run `--quick` for a
fast 16×16 self-test; the full-res mode is limited by ③.

## Current tree state

- **① `dx_m` fix**: kept (in `scripts/generate_dataset.py`; all 42 tests pass;
  smoke regenerated clean). This alone unblocks the full path from crashing on
  clip #1 on any backend.
- **② wind**: no code change kept. The drag and closed-form-polar attempts were
  reverted; `src/mpm_cloth.py` and `configs/mpm.yaml` are back to baseline.
- **Tooling**: `scripts/validate_edge_cases.py` harness kept.

## Recommended next step

1. **Decide the wind scenario** (see ②): whole-scenario `ImplicitClothSim`, a real
   codimensional-MPM/inversion-handling solver effort, or drop wind. This is a
   design call, not a quick fix.
2. Meanwhile, drape + collision are stable and generateable. A Modal GPU smoke
   test (6 clips, ~$0.10) validates full res on CUDA and confirms whether ③ is
   arm64-only — worth running for drape/collision regardless of the wind decision.
