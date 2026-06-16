# M2 Stack Note — Backend re-decision (E3, 2026-05-18)

## What changed

The W1 stack lock-in chose **NVIDIA Warp MPM** as the physics backend. During Episode 3 (M2/W3) we ran a probe of the actual installed surface and found:

- `warp-lang==1.13.0` ships no MPM example.
- The `warp.sim` module (which previously provided particle/cloth/soft-body primitives) has been **removed** in current Warp releases.
- Warp's bundled `examples/benchmarks/benchmark_cloth_warp.py` is mass-spring cloth implemented manually with `@wp.kernel` decorators — useful as a code template for our F2 fallback (`src/cloth_implicit.py`), but not for an MPM reference.

Building an MLS-MPM reference from scratch on Warp would mean writing P2G / grid-update / G2P kernels by hand. Estimated cost: 5–7 focused days. The roadmap explicitly warned against this ("use a tested MPM implementation rather than rolling your own from scratch").

## What we found in Taichi

Taichi 1.7 ships **5 MPM examples** out of the box:

- `mpm88.py`, `mpm99.py`, `mpm128.py`, `mpm3d.py` — fluid/granular MLS-MPM at increasing resolution.
- `mpm_lagrangian_forces.py` (186 lines) — **MLS-MPM where Lagrangian triangle forces drive the MPM particles via autodiff**. Total energy is computed per triangle from a NeoHookean energy density, `ti.ad.Tape` produces `x.grad`, and that gradient is fed into P2G as a force. This is exactly the cloth-MPM pattern (a Lagrangian mesh deforms, accumulates strain energy, exerts force via the deformation gradient) — and it uses MLS-MPM (paper2's formulation) verbatim.

To go from this 2D demo to our 3D 64×64 cloth:
- `dim = 2 → 3`, `n_grid = 64 → 128`, particles laid out as a 64×64 quad-grid.
- Replace NeoHookean energy with **linear co-rotational anisotropic** stress (warp / weft) per triangle. Three constants instead of two; same energy framework.
- Replace the static circle collider with a configurable scenario (sphere drape / wind force / moving sphere).
- Add per-frame state dump (`x, v, F, a, contact_flag`) for the dataset.

Estimated cost: **1.5–2 days** on top of a tested example. ~3× saving vs the from-scratch Warp path.

## What does not change

- ML framework remains **PyTorch + PyTorch Geometric** (M3 onward).
- Compute target remains **Google Colab T4**. Taichi has a CUDA backend (Colab T4), Metal (Apple Silicon dev), and CPU fallback; install footprint is comparable to Warp.
- Locked thresholds (τ = 0.05 m, ε = 0.10 m, etc., from [m1-paper-notes.md](m1-paper-notes.md)) are unchanged — they were derived from MeshGraphNets cloth numbers, not from the backend.
- The roadmap explicitly listed both Warp and Taichi as acceptable backends ("Taichi (taichi_mpm / MLS-MPM sample) or NVIDIA Warp MPM. Both are GPU-friendly, open source. Pick one in W3 and stop shopping."). Picking Taichi in W3 — exactly when scoped — does not violate the plan.

## Risks introduced by the swap

| Risk | Mitigation |
|---|---|
| Taichi's `ti.ad.Tape` autodiff is slower than hand-written stress kernels. | Cloth has only 4 096 particles and ~8 000 triangles; this is well below Taichi's autodiff sweet spot. If it bottlenecks, swap to a hand-written stress kernel later. |
| Determinism on GPU varies across Taichi versions. | Pin `taichi>=1.7,<2.0`. Determinism test (`tests/test_determinism.py`) catches regressions. |
| Apple Silicon (Metal) and CUDA may yield different float ordering. | All eval/training is on Colab T4 (CUDA). Mac is dev-only — its numbers are illustrative. |

## What we keep from the Warp probe

The Warp `benchmark_cloth_warp.py` is still relevant — it's a reference C-style implementation of implicit Euler mass-spring cloth. We will **read it (not run it)** as a sanity check when we implement `src/cloth_implicit.py` in W4 (Baraff-Witkin fallback).
