# Hybrid Neural-MPM for Cloth Roadmap

12-Week Plan

Hybrid Neural–MPM for Interactive Cloth Simulations

A Learned Grid-Update inside MPM for Cloth, with an Implicit-Cloth Safety Net

12-Week Research Roadmap (Five Milestones; Research-Intern Friendly)

Project One-Line: Adapt the hybrid Neural-MPM framework from arXiv:2505.18926, originally developed for fluid simulation, to cloth.

The Material Point Method (MPM) represents cloth as a cloud of particles that exchange information with a background grid every timestep (Particle-to-Grid, grid update, Grid-to-Particle). You will train a small neural network to replace the expensive part of this cycle (the grid-update step that resolves forces and internal stresses), run it as the default solver, and use a cosine-similarity complexity detector to route “hard” frames (folds, fast contacts, high strain-change) back to a standard implicit cloth step (Stuyck/Erleben, or simplified mass–spring implicit Euler) as a safety net. The goal: real-time (≥30 FPS) interactive cloth with stability close to a full MPM solve, evaluated on a single, controlled mesh family.

Why MPM (and not a pure mesh solver)? MPM is a hybrid Lagrangian–Eulerian method: particles carry physical state (position, velocity, deformation gradient F, stress), and a temporary background grid makes forces, contact, and plasticity easy to handle without tangled meshes. This is exactly why the seed paper works: the grid update is the bottleneck, and neural networks are well-suited to approximate it. For cloth, MPM has seen strong recent work (AnisoMPM, MLS-MPM-based cloth), and it unifies contact/self-contact and material nonlinearity in a way mesh-based solvers struggle with.

Primary Research Questions:

- **RQ1 (Neural grid-update):** Can a small neural solver, trained on MPM cloth trajectories, accurately predict the next-step per-particle accelerations (or equivalently, post-grid-update velocities/displacements) so that we can skip the expensive full P2G→grid-solve→G2P cycle for most timesteps?
- **RQ2 (Complexity detector):** Can a simple detector, cosine similarity between consecutive per-particle acceleration vectors, possibly augmented with strain-rate or deformation-gradient-change signals, reliably flag frames where the neural solver will fail, so we fall back to a physics step before visible artifacts appear?
- **RQ3 (Real-time hybrid):** On a cloth of ∼4k particles, does the hybrid pipeline (mostly neural, occasional physics fallback) run at ≥15 FPS on a Google Colab T4 GPU (equivalent to ∼30 FPS on a modern consumer GPU like RTX 4070 / Apple Silicon M3), while keeping per-particle error (L2 / Chamfer vs. a full-MPM reference) inside a pre-declared tolerance band?

Feasibility Note (3 months on Google Colab T4): Achievable in 12 weeks if you (1) fix the cloth particle count, material family, and scenario set early, (2) use a tested MPM implementation (Taichi / taichi_mpm or NVIDIA Warp) as the reference rather than rolling your own from scratch, (3) generate training data yourself from that reference (labels are exact and cheap), and (4) keep the real-time target to a single Colab T4 GPU with a fixed scene, reported as wall-clock per step plus a pre-rendered demo video (Colab does not support a live interactive window). Hardware note: the T4 is a 2018 datacenter card; expect roughly half the per-step throughput of a modern consumer GPU. Checkpoint every 15–30 min so Colab session drops don’t lose work. Stretch goals like rich self-contact or plastic yielding are left for a follow-on project.

---

## Contents

1. Project Overview
   - 1.1 Background: MPM in One Page (Beginner-Friendly)
   - 1.2 High-Level Goals (Pattern-Aligned)
   - 1.3 Timeline at a Glance
   - 1.4 Recommended Stack (Keep It Simple)
   - 1.5 Ethics, Licensing, and Responsible Use
2. Milestone 1: Foundations & Scope (Weeks 1–2)
3. Milestone 2: MPM Reference, Fallback & Dataset (Weeks 3–4)
4. Milestone 3: Neural Grid-Update (Weeks 5–6)
5. Milestone 4: Complexity Detector & Hybrid Rollout (Weeks 7–9)
6. Milestone 5: Real-time Integration, Ablations & Manuscript (Weeks 10–12)
7. Metrics, Targets, and Reporting (What Results to Obtain)
   - 7.1 Primary Metrics
   - 7.2 Concrete Target Bands (Educational, Not SOTA)
   - 7.3 Required Figures and Tables
8. Milestone Details (Week-by-Week)
9. Deliverables Summary (Per Milestone)
10. Appendix A: Excel Columns (copy/paste)
11. Appendix B: Metrics (Formal Definitions)
12. Appendix C: Suggested Repo Layout

---

## 1 Project Overview

### 1.1 Background: MPM in One Page (Beginner-Friendly)

MPM simulates a continuum (fluid, soft body, cloth) as a swarm of particles. Each particle \(p\) carries mass \(m_p\), position \(x_p\), velocity \(v_p\), and a deformation gradient \(F_p\) (which tracks how much the material around that particle has stretched/sheared). Every timestep has three phases:

1. **P2G (Particle to Grid):** particle mass and momentum are scattered to the nearest background grid nodes using interpolation weights \(w_{ip}\).
2. **Grid update:** on the grid, internal forces are computed from the particles’ stress (for cloth: anisotropic constitutive model), external forces (gravity, contact) are added, and an explicit or implicit integration step updates grid velocities.
3. **G2P (Grid to Particle):** updated grid velocities are interpolated back to each particle; particle positions and \(F_p\) are updated accordingly.

Phase 2, the grid update, especially if done implicitly with contact resolution, is the expensive part and the one the seed paper replaces with a neural network. In this project, phase 2 is also where the neural solver will live, and the “fallback” means running a full MPM step (or an implicit mass–spring step on the same cloth particles treated as mesh nodes) instead.

### 1.2 High-Level Goals (Pattern-Aligned)

| Phase      | Goal |
|-----------|------|
| Discover (W1–2)  | Read core MPM papers, Neural-MPM for fluids (seed paper), cloth-MPM papers, and implicit-cloth fallback references. Lock scope: particle count, material family, scenarios, metrics, success band. |
| Instrument (W3–4) | Stand up a reference MPM cloth simulator (Taichi / Warp). Generate the trajectory dataset. Build the evaluation harness. Implement the implicit-cloth fallback solver. |
| Learn (W5–6)      | Train the neural solver on per-particle acceleration prediction using MPM trajectories. Characterize one-step and rollout accuracy. Study failure modes. |
| Hybridize (W7–9)  | Add the complexity detector. Build the hybrid rollout pipeline. Sweep thresholds; produce the speed / accuracy tradeoff curve. Decide the operating point. |
| Finalize (W10–12) | Real-time demo, ablations, honest error analysis, a short 6–8 page paper-style report, and a reproducible repo. |

### 1.3 Timeline at a Glance

| Milestone | Weeks | Key Deliverables |
|-----------|-------|------------------|
| M1: Foundations & Scope | 1–2 | Literature matrix (12–15 papers); scoped plan; metrics + target bands; risk/ethics memo. |
| M2: MPM Reference + Fallback + Dataset | 3–4 | Working MPM cloth simulator (reference); implicit-cloth fallback; trajectory dataset (≥10k clips); evaluation harness. |
| M3: Neural Grid-Update | 5–6 | Trained MLP and GNN solvers predicting per-particle accelerations; one-step + rollout error tables; failure-mode notebook. |
| M4: Complexity Detector + Hybrid Rollout | 7–9 | Detectors D1–D3 trained; hybrid pipeline (neural default, physics fallback); tradeoff curve; chosen operating point. |
| M5: Real-time Integration + Manuscript | 10–12 | Pre-rendered demo video at target FPS (≥15 FPS on T4); ablations (A1–A5); 6–8 page report; reproducible repo + Colab quickstart notebook. |

### 1.4 Recommended Stack (Keep It Simple)

| Component | Recommendation (Practical Defaults) |
|-----------|--------------------------------------|
| MPM reference simulator | Taichi (taichi_mpm / MLS-MPM sample) or NVIDIA Warp MPM. Both are GPU-friendly, open source, and have well-tested cloth/soft-body demos to adapt. Pick one in W3 and stop shopping. |
| Cloth representation | ∼4k particles arranged on a 64×64 quad sheet. Each particle has mass, velocity, and deformation gradient \(F_p\). For the implicit fallback, treat the same particles as mesh nodes with spring edges along the quad grid. |
| Material model | Start with a linear co-rotational cloth model (anisotropic stiffness along warp/weft). AnisoMPM-style constitutive laws are more realistic, but not required for the headline result. |
| Neural solver backbone | Start with a per-particle MLP that sees its own state plus its k-ring neighbors’ state (a small fixed neighborhood). Graduate to a MeshGraphNets-style GNN (3–5 message-passing steps) only if the MLP under-performs. |
| Training data | Synthetic MPM trajectories from the reference simulator: draping, wind gusts, sphere/plane collisions. Target 10k–50k short clips (∼1–2 s each). |
| Implicit-cloth fallback | Mass–spring implicit Euler (Baraff–Witkin 1998 style) on the same particles treated as mesh nodes. Simple to implement; this is the “physics safety net.” |
| Complexity detector | Start simplest: cosine similarity between \(a_t\) and \(a_{t-1}\), max-pooled over particles. Add strain-rate and \(\Delta F_p\) channels only if D1 is not separable enough. |
| Framework | PyTorch for the neural side. Taichi or Warp for the physics side. Single repo, single Python environment. |

### 1.5 Ethics, Licensing, and Responsible Use

- All training data is synthetic, generated by your own reference MPM run, so there are no dataset-licensing issues.
- Taichi is Apache 2.0; NVIDIA Warp is Apache 2.0. Verify, cite, and respect the licenses of any OSS MPM demo you start from.
- This is a research project. Every headline number must be reported relative to the chosen reference MPM solver, on a frozen evaluation set. Do not claim “production-ready” real-time cloth.
- Document failure cases honestly. The value of the work is in characterizing when the hybrid works, not in pretending it always does.

---

## 2 Milestone 1: Foundations & Scope (Weeks 1–2)

### Objectives

- Read and summarize 12–15 core papers on MPM, neural-MPM / learned physics, cloth MPM, and implicit cloth solvers.
- Lock the scope: one particle count, one material family, one scenario set, one metric suite.
- Write three concrete hypotheses (H1–H3) to be tested by Week 12.

### Core Reading List (read deeply)

**MPM foundations:**

1. Jiang, Schroeder, Teran, Stomakhin & Selle, *The Material Point Method for Simulating Continuum Materials* (SIGGRAPH Course Notes, 2016), the canonical MPM tutorial. Read before anything else.
2. Hu, Fang, Ge, Qu, Zhu, Pradhana, Jiang, *A Moving Least Squares Material Point Method with Displacement Discontinuity and Two-Way Rigid Body Coupling* (MLS-MPM, SIGGRAPH 2018), the modern, fast MPM variant most GPU codes use.
3. Stomakhin, Schroeder, Chai, Teran, Selle, *A Material Point Method for Snow Simulation* (SIGGRAPH 2013), a clean MPM case study with plasticity.

**Cloth with MPM:**

4. Jiang et al., *AnisoMPM: Animating Anisotropic Damage Mechanics* (SIGGRAPH 2020), anisotropic MPM models used for cloth-like materials.
5. Guo, Liu, Han, *A Material Point Method for Thin Shells with Frictional Contact* (or equivalent thin-shell/cloth MPM paper, pick one that is accessible).
6. A recent (2022–2024) cloth-MPM paper (SIGGRAPH / TOG / Eurographics). Verify access before adding it.

**Neural MPM / learned physics (seed papers):**

7. Seed paper: hybrid neural MPM for fluid, arXiv:2505.18926 (2025). Focus on: what part of the MPM cycle the neural network replaces, how the network is conditioned on neighborhood state, and how stability across rollouts is handled.
8. arXiv:2403.12820 (2024), second core paper. Verify title/authors and summarize how it relates.
9. Sanchez-Gonzalez et al., *Learning to Simulate Complex Physics with Graph Networks* (GNS, ICML 2020).
10. Pfaff et al., *Learning Mesh-Based Simulation with Graph Networks* (MeshGraphNets, ICLR 2021).
11. Um, Brand, Holl, Thuerey, *Solver-in-the-Loop* (NeurIPS 2020), learned correction combined with an iterative numerical solver.
12. Hu et al., *DiffTaichi / Differentiable MPM* (ICLR 2020), differentiable continuum simulation background.

**Implicit-cloth fallback references:**

13. Baraff & Witkin, *Large Steps in Cloth Simulation* (SIGGRAPH 1998), implicit-Euler mass–spring cloth.
14. Stuyck, *Cloth Simulation for Computer Graphics* (Synthesis Lecture, 2018), mass–spring and FEM chapters.
15. Erleben, *Numerical Methods for Linear Complementarity Problems in Physics-Based Animation* or a similar implicit-cloth reference that is accessible.

Note: this list is a starting point. Verify every arXiv / DOI link before adding a paper to the final matrix. Drop anything that is not accessible and replace it with a suitable alternative.

### Additional Excel Tracking (15+ papers)

Columns:

- Paper Title
- Year
- Venue
- Problem/Task
- Dataset(s)
- Method Summary
- Key Metrics
- Key Results
- Limitations
- Relevance (1–5)

### Hypotheses (write these into `docs/scope.pdf` or equivalent)

- **H1:** A per-particle MLP (or a small GNN) trained on MPM cloth trajectories predicts next-step accelerations within a pre-declared per-particle L2 budget for at least a target rollout horizon on held-out “easy” scenarios (draping, light wind).
- **H2:** A cosine-similarity detector on consecutive per-particle acceleration vectors, optionally augmented with strain-rate and \(\Delta F_p\) channels, separates “easy” frames from “hard” frames with AUROC ≥0.8 on a held-out set.
- **H3:** A hybrid rollout that runs the neural grid-update by default and falls back to a full physics step (MPM step or implicit mass–spring step) only when the detector fires is at least 3–5× faster per step than pure MPM, while keeping per-particle L2 inside a pre-declared tolerance band.

### Deliverables (M1)

- `docs/lit-matrix.xlsx` (15+ rows, fully filled).
- `docs/scope.pdf` (1–2 pages: hypotheses, particle count, material parameters, scenarios, metrics, success band).
- `docs/eval-plan.pdf` (metrics + frozen evaluation protocol, seeds).
- `docs/risk-ethics.pdf` (1 page).

### Acceptance Check (M1)

A reader can answer, from the docs alone: “What exact MPM reference, particle count, material, scenarios, metrics, and ablations will be run, and what counts as success by Week 12?”

---

## 3 Milestone 2: MPM Reference, Fallback & Dataset (Weeks 3–4)

### Objectives

- Have a stable reference MPM cloth simulator running on the chosen particle family.
- Have a stable implicit-cloth fallback (mass–spring implicit Euler / Stuyck-style) operating on the same particles as mesh nodes.
- Generate a synthetic trajectory dataset (draping, wind, collisions) and the evaluation harness.

### Tasks (M2)

1. **Repo skeleton:** `src/`, `configs/`, `notebooks/`, `results/`, `docs/`, `data/`.
2. **MPM reference (default: Taichi MLS-MPM or Warp MPM):** adapt a Taichi or Warp MPM sample to a 64×64 cloth sheet; validate on a static drape over a sphere (energy decay clean; particles settle to a rest state).
3. **Implicit-cloth fallback:** implement a mass–spring implicit Euler cloth solver that uses the same particles as mesh vertices. Reference: Baraff–Witkin 1998; Stuyck 2018 (Ch. 4). Validate on the same static drape as a sanity check (qualitatively matches MPM rest state).
4. **Dataset generation (from the MPM reference):**
   - Draping: ∼4k short clips (1–2 s each) of cloth falling onto primitive shapes.
   - Wind: ∼3k clips with varied wind direction/magnitude.
   - Collisions: ∼3k clips with a moving sphere passing through the cloth.
   - Per frame: particle positions, velocities, accelerations, deformation gradients \(F_p\), grid forces, contact flags.
5. **Eval harness:** `src/eval.py` runs N fixed-seed reference scenarios; reports per-particle L2, Chamfer distance, and energy-drift.
6. **Baseline timing:** measure wall-clock per step for (i) full MPM reference, (ii) implicit-cloth fallback. These are the two numbers the hybrid must beat.

### Deliverables (M2)

- `src/mpm_cloth.py` (reference MPM, thin wrapper over Taichi/Warp) with tests.
- `src/cloth_implicit.py` (implicit mass–spring fallback) with tests.
- `data/cloth_trajectories/` (≥10k short clips) + `data/README.md` (data card).
- `src/eval.py` + `configs/eval.yaml`.
- `notebooks/00-mpm-baseline.ipynb` with sanity plots.

### Acceptance Check (M2)

The MPM reference reproduces the static drape within small variance across seeds; any dataset clip can be replayed deterministically from its saved trajectory; the implicit fallback also reaches a visually similar rest state on the same scenario.

---

## 4 Milestone 3: Neural Grid-Update (Weeks 5–6)

### Objectives

- Train a neural solver that maps (per-particle state + neighbor features) → post-grid-update per-particle acceleration.
- Evaluate one-step accuracy and multi-step rollout stability on held-out scenarios.
- Characterize failure modes: which scenarios, which particle regions, which frames.

### What the Network Replaces

In a standard MPM step the grid-update computes particle accelerations implicitly from the stress field. The network takes the pre-grid-update particle state (position, velocity, \(F_p\)) plus external forces and k-ring neighbor features, and directly outputs the per-particle acceleration that the grid update would have produced. The solver then applies that acceleration to update velocities and positions without ever touching the grid. The P2G+G2P transfers are also bypassed in the neural path (since the computation stays in particle space), though grid-like features may still be logged if helpful.

### Methods (Core Set)

- **Model A (MLP-per-particle):** input per particle = \(x_p, v_p, F_p, f^{ext}_p\) and the same for its k-ring neighbors. Output: \(a_p\). No message passing; fast and local.
- **Model B (MeshGraphNets-style GNN):** 3–5 rounds of message passing on the cloth mesh topology. Node features = particle state; edge features = rest length, current length, relative velocity. Output: \(a_p\).
- **Training:** start with one-step teacher forcing; add a short rollout curriculum (predict 2, 4, 8 steps) once one-step error is stable.
- **Losses:** per-particle acceleration L2. Optional auxiliary: penalize divergence of total kinetic+potential energy from the MPM reference.

### Tasks (M3)

1. **Data loaders** that sample (state\(_t\), state\(_{t+1}\)) pairs and build k-ring neighborhood tensors.
2. **Train Model A.** Report one-step L2 and rollout error at 1 s, 2 s, 5 s.
3. **Train Model B.** Compare to Model A on accuracy and rollout length.
4. **Failure-mode analysis:** plot per-frame error vs. time; cluster failing clips by scenario; tag high-error particle regions.
5. **Per-scenario breakdown** (draping / wind / collision).

### Deliverables (M3)

- `src/neural_solver.py` (both models), `configs/{mlp,gnn}.yaml`.
- `results/neural-one-step.csv`, `results/neural-rollout-error.csv`.
- `notebooks/10-neural-solver.ipynb` (rollout vs. MPM reference, per-scenario error bars).
- `docs/m3-neural-summary.pdf` (1–2 pages).

### Acceptance Check (M3)

On held-out clips, at least one model maintains per-particle L2 below a pre-declared threshold for the target rollout horizon on easy scenarios, and failure modes are documented with concrete named clips.

---

## 5 Milestone 4: Complexity Detector & Hybrid Rollout (Weeks 7–9)

### Objectives

- Build a lightweight complexity detector that predicts, at every step, whether the neural grid-update is likely to produce a bad prediction.
- Integrate the detector into a hybrid rollout: neural by default, fall back to physics (either a full MPM step or an implicit mass–spring step) when the detector fires.
- Quantify the speed / accuracy tradeoff; pick an operating point.

### Core Analyses

- **Detector inputs:**
  - (i) cosine similarity between \(a_t\) and \(a_{t-1}\) (the simplest signal).
  - (ii) per-particle strain-rate (\(\|\dot{F}_p\|\) or equivalent).
  - (iii) contact-flag fraction.
  - (iv) max change in edge length over the k-ring.
- **Detector variants:**
  - D1: single threshold on cosine similarity.
  - D2: logistic regression over features (i)–(iv).
  - D3: tiny MLP over the same features.
- **Fallback modes:**
  - F1 (MPM fallback): run one full MPM step (P2G + grid update + G2P) when detector fires.
  - F2 (implicit-cloth fallback): run one implicit mass–spring step on the same particles treated as mesh nodes.
  - F3 (residual correction): keep the neural prediction and add a small implicit correction.
- **Threshold sweep:** run the hybrid across a grid of detector thresholds; measure per-frame error, fallback fraction, wall-clock.

### Tasks (M4)

1. From M3 rollouts, pre-compute per-frame “neural error” labels (\(y_t = \mathbf{1}[L2(t) > \varepsilon]\)).
2. Train D1–D3. Report AUROC and FPR@95%TPR on the held-out split.
3. Integrate the best detector into `src/hybrid.py` (neural-by-default loop with per-step detector call).
4. Run hybrid rollouts; record per-frame error, fallback fraction, wall-clock.
5. Sweep the detector threshold; produce the main tradeoff plot.
6. Compare F1 vs. F2 vs. F3.

### Deliverables (M4)

- `src/detector.py`, `src/hybrid.py`.
- `results/detector-metrics.csv`.
- `results/hybrid-sweep.csv` (threshold → fallback fraction, error, wall-clock).
- `notebooks/20-hybrid-analysis.ipynb` (headline tradeoff figure; F1 vs. F2 vs. F3 comparison).

### Acceptance Check (M4)

There is a concrete operating point where the hybrid uses the physics fallback on at most a pre-declared fraction of frames and keeps per-particle error inside the M1 success band on the frozen held-out set.

---

## 6 Milestone 5: Real-time Integration, Ablations & Manuscript (Weeks 10–12)

### Objectives

- Integrate the hybrid into a minimal demo scene (one scene, one camera, one cloth), exported as an MP4 video rendered on Colab. (Colab does not support a live interactive window; a local GPU would be needed for true interactivity.)
- Run the ablation set required to answer H1–H3 honestly.
- Write the short manuscript (6–8 pages) and package the repo.

### Tasks (M5)

1. **Demo target:** real-time loop at ≥15 FPS on Google Colab T4 (equivalent to ∼30 FPS on RTX 4070 / Apple Silicon M3). One rotating camera + one scripted interaction (drop + sweep) is enough. Output as a rendered MP4, plus a wall-clock timing table. Report both the T4 number and a projected number for modern consumer GPUs.
2. **Ablations (small, high-value):**
   - A1: Model A (MLP) vs. Model B (GNN).
   - A2: Detector off (pure neural) vs. D1 vs. D2/D3.
   - A3: Fallback mode F1 (MPM) vs. F2 (implicit mass–spring) vs. F3 (residual).
   - A4: Particle count sensitivity, 1k vs. 4k particles (only if time permits).
   - A5: Dataset-size ablation, 25% / 50% / 100% of training data.
3. **Real-time timing:** wall-clock-per-step table for (i) full MPM, (ii) neural, (iii) hybrid; end-to-end FPS achieved.
4. **Error analysis:** half a page of honest failure discussion.
5. **Manuscript (6–8 pages):** target venues in order of preference: SIGGRAPH Poster, Eurographics Short Paper, NeurIPS Workshop on Machine Learning and the Physical Sciences, JuliaCon (if parts are ported to Julia / Lux.jl).
6. **Repo finalization:** README with 5-minute quickstart, reproducible configs + scripts, saved checkpoints, pinned requirements.

### Deliverables (M5)

- `results/ablations/` (one CSV per ablation).
- `results/realtime-timing.csv`.
- `demo/` with a runnable script.
- `report/manuscript.pdf` (6–8 pages).
- `README.md` with quickstart + repro instructions.

### Acceptance Check (M5)

A third party clones the repo, runs the quickstart, and reproduces (within small variance): (i) the headline hybrid-vs-MPM tradeoff plot (Figure 1) and (ii) the real-time timing table.

---

## 7 Metrics, Targets, and Reporting (What Results to Obtain)

### 7.1 Primary Metrics

- **Accuracy:** per-particle L2 error vs. the full-MPM reference, averaged over a held-out evaluation set (per-scenario breakdown).
- **Rollout stability:** maximum rollout length (seconds of simulated time) before per-particle L2 exceeds \(\tau\).
- **Chamfer distance:** between predicted and reference particle clouds per frame.
- **Energy drift:** \(|E_{pred}(t) - E_{ref}(t)| / E_{ref}(t)\).
- **Runtime:** wall-clock per step for full-MPM / neural / hybrid; end-to-end FPS of the hybrid demo.
- **Fallback fraction:** fraction of frames on which the hybrid used the physics fallback.

### 7.2 Concrete Target Bands (Educational, Not SOTA)

Set targets relative to the chosen full-MPM reference; do not chase external leaderboards.

- **M3 neural solver:** per-particle L2 within a small multiple of MPM step-to-step noise for ≥1 s of rollout on easy scenarios (draping, light wind).
- **M4 detector:** AUROC ≥0.8 for predicting “high-error frames” on the held-out split.
- **M4 hybrid:** at least 3–5× faster per step than full MPM while keeping per-particle L2 within 2× of pure-neural on non-fallback frames.
- **M5 demo:** ≥15 FPS end-to-end on Google Colab T4 (projected ∼30 FPS on RTX 4070 / Apple Silicon M3), with visually stable cloth for 30+ seconds of scripted interaction, delivered as a rendered MP4.

### 7.3 Required Figures and Tables

- **Table 1:** one-step + rollout error per model (MLP / GNN) per scenario.
- **Figure 1:** rollout error vs. time for full-MPM / neural / hybrid on a held-out clip.
- **Figure 2:** detector tradeoff, fallback fraction vs. max rollout error for D1/D2/D3.
- **Table 2:** wall-clock timing per step: full-MPM / neural / hybrid.
- **Figure 3:** qualitative frame strips, full-MPM vs. neural vs. hybrid on a hard collision scenario.

---

## 8 Milestone Details (Week-by-Week)

| Week | Focus | Concrete Outputs |
|------|-------|------------------|
| W1 | Literature sprint (MPM + neural MPM) | 8 papers summarized; material family chosen; hypotheses drafted. |
| W2 | Scope lock | Metrics + success bands frozen; eval protocol agreed; risk/ethics memo. |
| W3 | MPM reference | MPM cloth simulator running on Taichi/Warp; static drape validated. |
| W4 | Implicit fallback + dataset | Mass–spring implicit fallback working; 10k+ MPM trajectory clips generated; `eval.py` end-to-end. |
| W5 | MLP neural grid-update | MLP baseline trained; one-step + rollout error tables produced. |
| W6 | GNN + failure modes | GNN trained; per-scenario error bars; failing-clip notebook. |
| W7 | Detector training | D1–D3 trained; AUROC + FPR@95%TPR table. |
| W8 | Hybrid rollout | Hybrid pipeline runs end-to-end; first tradeoff curve. |
| W9 | Fallback mode sweep | F1 vs. F2 vs. F3 compared; operating point chosen. |
| W10 | Ablations | A1–A3 complete; A4/A5 started if feasible. |
| W11 | Real-time demo | Pre-rendered MP4 demo at ≥15 FPS on T4; timing table locked (T4 and projected RTX 4070 numbers). |
| W12 | Manuscript + repo | 6–8 page report drafted; repo packaged with quickstart. |

---

## 9 Deliverables Summary (Per Milestone)

- **M1:** literature matrix; scoped plan; eval plan; risk/ethics memo.
- **M2:** MPM reference; implicit-cloth fallback; trajectory dataset; eval harness; baseline timing.
- **M3:** neural grid-update checkpoints; rollout-error tables; failure-mode notebook.
- **M4:** detector; hybrid pipeline; tradeoff curves; chosen operating point.
- **M5:** real-time demo; ablation CSVs; reproducible repo; 6–8 page report.

---

## 10 Appendix A: Excel Columns (copy/paste)

- Paper Title
- Year
- Venue (conference / journal / arXiv)
- Problem / Task (MPM, neural MPM, cloth MPM, implicit cloth, learned corrector, …)
- Dataset(s) used
- Method Summary (3–4 lines)
- Key Metrics reported
- Key Results (numbers or qualitative)
- Limitations / failure modes
- Relevance to this project, 1–5

---

## 11 Appendix B: Metrics (Formal Definitions)

Let \(\hat{x}_p(t)\) be the predicted position of particle \(p\) at time \(t\), \(x_p(t)\) the full-MPM reference position, \(N\) the particle count, and \(T\) the number of evaluation frames.

\[
L2(t) = \sqrt{\frac{1}{N} \sum_{p=1}^N \|\hat{x}_p(t) - x_p(t)\|_2^2}
\]  
\[
Chamfer(t) = \frac{1}{N} \sum_p \min_q \|\hat{x}_p(t) - x_q(t)\|^2 + \frac{1}{N} \sum_q \min_p \|\hat{x}_p(t) - x_q(t)\|^2
\]
\[
EnergyDrift(t) = \frac{|E_{pred}(t) - E_{ref}(t)|}{E_{ref}(t)}
\]
\[
RolloutHorizon(\tau) = \max \{ t : L2(t') \leq \tau \; \forall t' \leq t \}
\]
\[
CosSim(a_t, a_{t-1}) = \frac{\langle a_t, a_{t-1} \rangle}{\|a_t\| \|a_{t-1}\|}
\]

Detector AUROC is computed on per-frame binary labels \(y_t = \mathbf{1}[L2(t) > \varepsilon]\) vs. the detector score.

---

## 12 Appendix C: Suggested Repo Layout

```text
cloth-hybrid-mpm/
  src/
    mpm_cloth.py          # reference MPM (Taichi/Warp wrapper)
    cloth_implicit.py     # implicit mass-spring fallback
    neural_solver.py      # MLP and GNN neural grid-update
    detector.py           # D1 / D2 / D3 complexity detectors
    hybrid.py             # hybrid rollout loop
    eval.py               # frozen evaluation protocol
  configs/
    mpm.yaml
    mlp.yaml
    gnn.yaml
    hybrid.yaml
    eval.yaml
  data/
    cloth_trajectories/   # synthetic MPM clips
    README.md             # data card
  notebooks/
    00-mpm-baseline.ipynb
    10-neural-solver.ipynb
    20-hybrid-analysis.ipynb
  results/
    neural-one-step.csv
    neural-rollout-error.csv
    detector-metrics.csv
    hybrid-sweep.csv
    realtime-timing.csv
    ablations/
  demo/
    run_demo.py
  docs/
    lit-matrix.xlsx
    scope.pdf
    eval-plan.pdf
    risk-ethics.pdf
    m3-neural-summary.pdf
  report/
    manuscript.pdf
  README.md
  requirements.txt
```
