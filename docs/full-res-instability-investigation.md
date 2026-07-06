# Full-resolution instability — investigation writeup (for review)

**Status: unresolved, paused for advisor input.** This documents what is known so
the solver decision can be made without re-deriving it. Companion to
`datagen-validation-findings.md`.

## TL;DR

The production cloth config (64×64 particles, 128³ background grid, explicit
MLS-MPM, `dt=1e-4`) **cannot generate trajectories** — clips crash partway through
on **every** backend. The crash is an **out-of-bounds memory access** (SIGBUS on
CPU, `CUDA_ERROR_ILLEGAL_ADDRESS` on GPU) caused by a numerical blow-up that ejects
a particle out of the grid. The trigger is **the hard contact projection at fine
mesh resolution — now confirmed**: removing the sphere makes full-res free-fall
stable, and with the sphere the crash fires the instant the cloth touches, even at
low impact speed. It is **not** element inversion (four other root-causes were also
tested and disproven). A damping probe shows **dissipation at contact stabilises
it** (a fix exists in that family), but global damping strong enough also damps the
free dynamics — so the refinement is contact-localised dissipation / soft contact,
or implicit contact. No fix is committed.

## What works vs. what fails

| Cloth res | Particles | Result |
|---|---|---|
| 16×16 | 256 | **Stable** — full 1.0 s, all scenarios, CPU + CUDA (this is the smoke config) |
| 32×32 | 1024 | Crashes at contact (~t=0.35 s) |
| 64×64 (production) | 4096 | Crashes at contact (~t=0.30 s), CPU + CUDA; Modal ran drape+collision 6/6 FAIL |

So there is a resolution threshold between 16 and 32; but coarsening to a usable
resolution does not fix it, and 16×16 is too coarse for the roadmap's ~4k particles.

## The mechanism (best current understanding)

Instrumented a 64×64 drape clip (no wind, no pins) at `dt=5e-5`:

- Cloth falls freely; `vmax` grows smoothly as g·t (0 → ~3.5 m/s by t=0.3 s). Stable.
- At **t≈0.3 s the falling cloth reaches the sphere.** The most-stretched element is
  **right at the sphere surface** (distance ≈ 0), stretched to principal stretch
  `σ_max≈3.2`, with `σ_min≈0.98` — **healthy, not inverted, not collapsed.**
- `vmax` then grows **exponentially, ~1.24×/step** (3.5 → 225 in ~13 steps) → a
  particle leaves the [0, 2]³ grid → OOB write → hard crash.

Signature = a stiff, contact-constrained element hitting the explicit integrator's
stability limit at impact. Contact is enforced by projecting inward normal velocity
to zero at grid nodes inside the sphere (and at the box/ground); at a fast impact on
a fine mesh this creates a sharp velocity discontinuity the explicit step amplifies.

## Hypotheses tested and DISPROVEN (so they aren't repeated)

1. **Autodiff through `ti.svd` reads uninitialised memory** (the reverse-grad emits
   "Loading variable before stored" warnings). → Replacing the SVD rotation with a
   regularised closed-form 2×2 polar gave a **byte-identical crash**. Not the cause.
2. **Global CFL / timestep too large.** → Halving `dt` (1e-4→5e-5) moved the crash
   step 3000→6000 but at the **same physical time** (~0.3 s, contact). A delay tied
   to the event, not a global CFL fix.
3. **The 2×2 xz-projected `F` collapses when triangles fold toward vertical.** →
   `min|det(xz-projection)|` dropped only ~27% (not toward 0) at the crash, and the
   culprit triangle was *stretched* (3D area 2–3× rest), not folded flat. Not it.
4. **Resolution-dependent CFL, fixable by coarsening.** → 32×32 also crashes at
   contact. (16×16 survives, so a threshold exists, but not at a usable resolution.)

Separately, for the **wind** scenario (a *distinct* problem — unbounded drift of a
corner-pinned sheet → true element inversion), these were also disproven as fixes:
air drag (only delays, ∝ k), wind magnitude, and pin layout. See
`docs/wind-deferral.md`.

## CPU follow-up analysis (contact trigger CONFIRMED + damping probe)

Three additional CPU-only experiments (no Modal), all at 64×64 / dt=1e-4:

1. **Contact is definitively the trigger.** Full-res drape with the **sphere moved
   out of the domain** (pure free-fall) ran **clean through t=380 ms** (`vmax`=3.73
   = g·t) — well past the t=300 ms where the *same* config with the sphere crashes.
   No contact ⇒ no crash.
2. **Not impact-velocity dependent.** A **gentle** drop (height 0.62, ~0.9 m/s at
   contact) also crashes — right when the cloth touches the sphere (~step 1000). So
   it is the hard projection itself at fine mesh, not an impact-shock magnitude.
3. **Global grid velocity damping stabilises it — but at a fidelity cost.** Adding
   `grid_v *= (1−d)` per step in the grid update:
   - `d = 2%` → completes through contact, but over-damps to near-static
     (`vmax` frozen at 0.048 = the gravity/damping terminal velocity); the cloth
     never really drapes.
   - `d = 0.2%` → **completes through contact AND the cloth drapes** (`vmax`≈0.49,
     bounded). But even 0.2% compounds over the ~3000-step fall and caps the
     terminal/impact speed at ~0.5 m/s (vs. ~3.5 m/s free-fall), so it noticeably
     damps the impact dynamics.

Takeaway: the instability is squarely the **hard contact projection vs. fine
explicit mesh**, and *dissipation* can stabilise it — but a global damping strong
enough to hold at contact also damps the free dynamics. That argues for a
**contact-localised** dissipation / soft contact (only where/when contact is
active) rather than global damping, or for implicit contact. Good news for the
mentor: a fix exists in the "add dissipation at contact" family; the open question
is doing it without polluting the free-flight dynamics that become training labels.

## Candidate fix directions (for the decision)

- **A. Contact robustness (leading, now evidence-backed).** Replace the hard
  normal-velocity projection with a penalty/soft contact + **contact-localised**
  damping (dissipate only at nodes in/near contact, so free-flight dynamics are
  untouched). The global-damping probe above shows dissipation stabilises it; the
  refinement is to localise it. Contact trigger is confirmed (experiment 1).
- **B. Implicit integration for the reference.** Generate reference clips with the
  project's `ImplicitClothSim` (unconditionally stable, already implemented/tested).
  Downside: labels are no longer MPM dynamics — see the "neural MPM surrogate"
  concern in `wind-deferral.md`.
- **C. Sub-stepping / adaptive `dt` at contact.** Smaller `dt` only when contact is
  active. May control it; adds cost and complexity; unproven here.
- **D. Reduce cloth resolution to ≤16×16.** Works today, but 256 particles is below
  the roadmap's ~4k target.
- **E. Proper thin-shell / codimensional MPM with robust contact.** The principled
  long-term route; largest effort.

## Repo state

Committed on branch `datagen-validation` (all validated, tests green):
`dx_m` crash fix, edge-case harness, Modal 1.x port, wind deferral (docs + default
`--n-wind 0`), eval prune, and the findings docs. **All solver experiments (drag,
closed-form polar, `dt` changes) were reverted** — `src/mpm_cloth.py` and
`configs/mpm.yaml` are at baseline.

**Important:** the committed generator default (drape + collision, wind deferred)
implies drape+collision are generateable — but at 64×64 they are **not** (this
investigation). Full-dataset generation is **blocked** on the solver decision above.
The 16×16 smoke path and all 42 unit tests remain green.
