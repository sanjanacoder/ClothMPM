# Literature Matrix (M1)

> Source of truth: [`lit-matrix.csv`](lit-matrix.csv). This Markdown is generated for readability and is regenerated whenever the CSV changes. Sorted by relevance descending.

## Top picks (relevance 5)

- **paper7 — Hybrid Neural-MPM for Interactive Fluid Simulations in Real-Time (Xu et al. 2025, arXiv:2505.18926).** This is our **seed paper**. Defines the architecture pattern we adapt to cloth: GNN-based neural physics by default, classical-MPM fallback when needed. Section 3 (the hybrid system) and Section 4 (the GNN at low spatio-temporal resolution) are the core read.
- **paper2 — MLS-MPM (Hu et al. 2018).** The MPM variant most modern GPU codes (Taichi, Warp) implement. Reading this fixes vocabulary and discretization choices for `src/mpm_cloth.py`.
- **paper10 — MeshGraphNets (Pfaff et al. 2021).** Direct reference for our M3 GNN backbone (Model B). Cloth (FlagDynamic) is one of the headline domains; their input-noise trick is what we use to stabilize rollouts.
- **paper13 — Large Steps in Cloth Simulation (Baraff & Witkin 1998).** Direct reference for `src/cloth_implicit.py`. The implicit-Euler + modified-CG approach is exactly what F2 (mass-spring fallback) needs.
- **paper9 — GNS (Sanchez-Gonzalez et al. 2020).** Foundation for particle-based learned simulators; their k-NN graph construction and noise-injection regularizer are reused in MeshGraphNets and our setup.

## Full matrix

| paper_id | Title | Year | Venue | Topic | Rel |
|---|---|---|---|---|---|
| paper7 | Hybrid Neural-MPM for Interactive Fluid Simulations in Real-Time | 2025 | arXiv:2505.18926 | Neural physics / learned simulation | 5 |
| paper2 | A Moving Least Squares Material Point Method (MLS-MPM) | 2018 | SIGGRAPH 2018 | MPM foundations | 5 |
| paper10 | Learning Mesh-Based Simulation with Graph Networks (MeshGraphNets) | 2021 | ICLR 2021 | Neural physics / learned simulation | 5 |
| paper13 | Large Steps in Cloth Simulation | 1998 | SIGGRAPH 1998 | Implicit cloth | 5 |
| paper9 | Learning to Simulate Complex Physics with Graph Networks (GNS) | 2020 | ICML 2020 | Neural physics / learned simulation | 5 |
| paper1 | The Material Point Method for Simulating Continuum Materials (Course Notes) | 2016 | SIGGRAPH 2016 Courses | MPM foundations | 4 |
| paper4 | AnisoMPM: Animating Anisotropic Damage Mechanics | 2020 | SIGGRAPH 2020 | Cloth MPM | 4 |
| paper11 | Solver-in-the-Loop | 2020 | NeurIPS 2020 | Neural physics / learned simulation | 4 |
| paper5 | A Material Point Method for Thin Shells with Frictional Contact | — | SIGGRAPH (TOG) | Cloth MPM | 4 |
| paper12 | DiffTaichi: Differentiable Programming for Physical Simulation | 2020 | ICLR 2020 | Differentiable physics | 3 |
| paper3 | A Material Point Method for Snow Simulation | 2013 | SIGGRAPH 2013 | MPM foundations | 3 |
| paper6 | Physics-inspired Estimation of Optimal Cloth Mesh Resolution | — | SIGGRAPH | Implicit cloth | 2 |
| paper8 | A Physics-embedded Deep Learning Framework for Cloth Simulation | — | Conference | Neural physics / learned simulation | 2 |

## Coverage of the M1 reading list

The roadmap's Core Reading List groups papers into four buckets. Our 13 papers cover them as follows:

| Roadmap bucket | Papers we have | Gap? |
|---|---|---|
| MPM foundations | paper1 (course notes), paper2 (MLS-MPM), paper3 (snow MPM) | None — all three roadmap items present. |
| Cloth with MPM | paper4 (AnisoMPM), paper5 (thin shells), paper6 (cloth resolution) | Three solid items; the optional 2022–2024 cloth-MPM paper is not strictly needed. |
| Neural MPM / learned physics | paper7 (seed), paper9 (GNS), paper10 (MeshGraphNets), paper11 (Solver-in-the-Loop), paper12 (DiffTaichi), paper8 (Zhao CNN cloth) | Strongest cluster; the roadmap's "second core paper" arXiv:2403.12820 is not in the set, but paper11 plays a similar role. |
| Implicit-cloth fallback | paper13 (Baraff–Witkin), paper6 (mesh resolution) | The roadmap's Stuyck 2018 / Erleben items are not present; Baraff–Witkin alone is sufficient for our F2 fallback. |

## Deep-read order (suggested for W2)

1. **paper7** — read in full; it is the seed paper.
2. **paper2** — read Sections 3–5 (the MLS-MPM transfers and stress divergence).
3. **paper10** — read Sections 3–4 (model + cloth experiments) for the GNN baseline.
4. **paper13** — read Sections 4–6 (implicit Euler, modified CG, constraints).
5. **paper9** — skim; mainly for the noise-injection trick.

A 5-paper deep read in W2 is achievable; the rest can be skimmed for citations only.
