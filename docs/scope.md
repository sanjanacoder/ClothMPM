# Scope (M1)

> **Status:** **LOCKED** as of W2 (2026-05-18). Values were derived from the W2 deep reads — see [m1-paper-notes.md](m1-paper-notes.md) for the full derivation chain, and [paper_notes/](paper_notes/) for per-paper extraction. From here on, this file is frozen for the remainder of the project.

## 1. Project in one sentence

Adapt the hybrid Neural-MPM framework from arXiv:2505.18926 (originally for fluids) to interactive cloth simulation: a small neural network replaces the MPM grid-update step by default; a lightweight complexity detector routes hard frames to a physics fallback (full MPM step or implicit mass–spring).

## 2. Hypotheses

- **H1 (Neural grid-update accuracy).** A per-particle MLP (or small GNN) trained on MPM cloth trajectories predicts next-step accelerations such that per-particle position L2 vs. the full-MPM reference stays below **τ = 0.05 m** (5% of cloth diameter) for ≥1 s of rollout on held-out *easy* scenarios (drape, light wind). Rationale: MeshGraphNets reports ~7 cm rollout-50 RMSE on FlagDynamic with a fully-tuned L=15 GNN; we set our budget at 5 cm understanding we run a smaller model on a smaller training budget — failing this is informative, not catastrophic.
- **H2 (Detector separability).** A cosine-similarity detector on per-particle acceleration vectors aggregated over a **10-step window** (per paper7's `δt = 10`), optionally augmented with strain-rate and ΔF channels, separates frames where `L2(t) > ε = 0.10 m` from the rest at AUROC ≥ 0.8 on the held-out evaluation split.
- **H3 (Hybrid speedup).** A hybrid rollout running neural by default and falling back to physics on detector firings is at least 3–5× faster per step than full MPM, while keeping per-particle L2 within 2× of pure-neural on non-fallback frames and inside the H1 budget overall.

## 3. Frozen scientific scope

| Knob | Value |
|---|---|
| Cloth | 64×64 quad sheet, ~4096 particles, 1 m × 1 m, 0.2 kg total mass |
| Material model | Linear co-rotational, anisotropic stiffness along warp/weft (`configs/mpm.yaml`) |
| MPM reference | Taichi MLS-MPM (Apache 2.0; switched from Warp during E3 — see [m2-stack-note.md](m2-stack-note.md)), explicit MLS-MPM step at `dt = 1 ms` |
| Implicit fallback | Mass–spring implicit Euler (Baraff–Witkin 1998 style) on the same particles as mesh nodes |
| Scenarios | `drape`, `wind`, `collision` only — no plasticity, no self-contact, no tearing |
| Compute target | Google Colab T4 (~15 FPS demo); local CUDA optional |
| Particle-count ablation (A4) | 1k vs. 4k particles, only if time permits |

## 4. Critical scientific decisions (with options + chosen default)

Each decision is configurable via YAML so we can run ablations.

### 4.1 Per-particle state representation (input to the neural solver)

| Option | Pros | Cons |
|---|---|---|
| **A. `(x, v, F, f_ext)`** **(default)** | Direct MPM state; fully describes particle dynamics; matches what the seed paper uses for fluids. | `F` is a 3×3 tensor; nine extra channels per particle. |
| B. `(x, v, F)` only | Smaller input. | External forces (wind, contact) become invisible. |
| C. `(v, F, f_ext)` (no `x`) | Translation-invariant (good for generalization). | Loses positional context for self-contact and pinning. |

**Choice:** A. Translation invariance is recovered cheaply by mean-centering positions per clip, which we will do at the data-loader level.

### 4.2 Neighborhood for the per-particle MLP (Model A)

| Option | Pros | Cons |
|---|---|---|
| 1-ring (8 neighbors) **(default)** | Tiny input; matches the regular grid; fast. | May under-resolve fast-moving wrinkles. |
| 2-ring (24 neighbors) | More context; closer to grid-update receptive field. | 3× input size; risk of overfitting at our data scale. |
| Adaptive radius | Topology-aware. | Implementation complexity; not needed at fixed 64×64 grid. |

**Choice:** 1-ring; revisit in M3 if the rollout error is unacceptable on `wind` or `collision`.

### 4.3 Loss

| Option | Pros | Cons |
|---|---|---|
| **L2 on accelerations only** **(default)** | Simple; stable training; matches GNS / MeshGraphNets. | No physical conservation guarantees. |
| L2 + total-energy regularizer | Encourages stable rollouts. | Needs careful weighting; risks underfitting. |
| L2 + physics-residual (PDE residual) | Strong inductive bias. | Complicated to compute on Warp's MPM internals; high risk of bugs. |

**Choice:** L2 on accelerations; energy regularizer kept as a 0-weighted YAML knob for an ablation.

### 4.4 Detector model

| Option | Pros | Cons |
|---|---|---|
| **D2 (logistic regression)** **(default)** | Interpretable; trains in seconds; easy to ship. | Linear in features; may miss interactions. |
| D1 (cosine threshold only, paper7's `rc=0.8` over a 10-step window) | Simplest possible baseline; matches the seed paper exactly. | Likely under-separates `wind` from `collision`. |
| D3 (tiny MLP) | Captures nonlinear interactions. | Risk of overfitting per scenario. |

**Choice:** D2 as default. D1 (with `rc = 0.8`, `δt = 10` from paper7) and D3 are run in parallel for ablation A2.

### 4.5 Fallback mode

| Option | Pros | Cons |
|---|---|---|
| **F2 (implicit mass–spring)** **(default)** | Cheapest fallback; well-understood; fast on the same particle layout. | Stiffness mismatch with MPM may show as small artifacts at switch points. |
| F1 (full MPM step) | Highest fidelity. | Most expensive; partially defeats the speed win. |
| F3 (residual correction) | Smoothest transitions. | Implementation complexity; harder to ablate cleanly. |

**Choice:** F2 default; F1 and F3 are ablation arms.

## 5. Evaluation budget (locked)

- 5 evaluation seeds (200–204), 3 scenarios (drape, wind, collision) → 15 frozen eval clips. These are **never** used for training or detector calibration.
- Headline number is computed on these 15 clips only.

## 6. Out of scope

Plasticity, fracture, self-contact, multi-layer cloth, GPU-shader rendering. These are explicitly stretch goals from the roadmap and are **not** required to claim success.

## 7. What "success by Week 12" means

A reader of `report/manuscript.pdf` and the README quickstart can:
1. Reproduce Figure 1 (rollout error vs. time, MPM/neural/hybrid) and Table 2 (wall-clock per step).
2. See that the hybrid achieves ≥15 FPS on Colab T4 on the demo MP4.
3. See an honest discussion of failure modes — which scenarios, which particles, which frames.
