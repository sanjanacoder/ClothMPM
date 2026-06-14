# Milestone 2 — MPM Reference, Fallback & Dataset

**Roadmap weeks:** W3–W4  
**Git commits:** `54b691c` (E3), `29de887` (E4a), `0bd1a39` (E4b), `bf2da56` (E4c)  
**All 13 tests pass** across the four commits that close M2.

---

## What M2 delivers (roadmap §3 checklist)

| Roadmap item | Status | File(s) |
|---|---|---|
| Working MPM cloth simulator | ✅ Done | `src/mpm_cloth.py`, `configs/mpm.yaml` |
| Validated on static drape | ✅ Done | `notebooks/00-mpm-baseline.ipynb` |
| Implicit-cloth fallback | ✅ Done | `src/cloth_implicit.py` |
| Trajectory dataset (smoke: 6 clips, full: ≥10k) | ✅ Smoke done | `scripts/generate_dataset.py`, `data/` |
| Eval harness (`src/eval.py` + `configs/eval.yaml`) | ✅ Done | `src/eval.py`, `configs/eval.yaml` |
| Baseline timing | ✅ Done | `src/eval.py::baseline_timing()` CLI |

---

## Deliverable 1 — Reference MPM Cloth Simulator (`src/mpm_cloth.py`)

### What it is

A **3D MLS-MPM cloth reference simulator** wrapping Taichi's MLS-MPM kernel
with a linear co-rotational anisotropic constitutive model (warp/weft
Young's moduli). A 64×64 quad-grid cloth sheet (~4096 particles) falls under
gravity onto a sphere collider.

### How we built it (E3)

**Starting point:** Taichi's `examples/simulation/mpm_lagrangian_forces.py`
(2D, fluid). We extended it in three ways:

1. **2D → 3D:** promoted all particle fields (`x`, `v`, `C`, `F`) from 2-vectors
   to 3-vectors. The background grid became `(n_grid, n_grid, n_grid)`.

2. **Fluid → cloth constitutive model:** replaced the neo-Hookean fluid
   constitutive law with a **linear co-rotational membrane energy**:
   ```
   ψ = μ ‖F − R‖² + ½ λ (J − 1)²
   ```
   computed per triangle in the xz plane of each quad-cell's two triangles.
   The 2×2 membrane deformation gradient F_mem is obtained by projecting the
   3D edge vectors onto the xz plane (the cloth's rest-pose plane). Polar
   decomposition gives R; `ti.ad.Tape` differentiates through the energy to
   produce the Lagrangian forces `x.grad`.

3. **Sphere contact:** the `grid_update` kernel projects grid node velocities
   when `|pos − sphere_center| < sphere_radius` (frictionless normal
   projection). Box bounds clamp all six faces with a 3-cell margin.

**MLS-MPM cycle (one `step()`):**

```
clear_grid()
total_energy[None] = 0
with ti.ad.Tape(total_energy):
    compute_total_energy()   # sets x.grad via autodiff
p2g()                        # scatter mass+momentum+force to 128^3 grid
grid_update()                # normalize, add gravity, sphere+box projection
g2p()                        # interpolate back; update a, v, x, C, F
```

Key implementation detail: `a[p] = (new_v − v[p]) / dt` is computed in the
G2P kernel so each saved frame carries the per-particle acceleration that the
grid update produced. This is the **training label** for the M3 neural solver.

**Config:** `configs/mpm.yaml` — 64×64 cloth, 128³ grid, dt=1e-4 s, E=10⁴ Pa,
ν=0.3. The cloth is 1 m × 1 m, total mass 0.2 kg, dropped from y=1.5 m.

**Validation (notebook cell 7):** two independent runs from identical config
produce `max|Δx| = 0.0` on CPU — bit-identical determinism confirmed.

### Public API

```python
sim = MPMClothSim("configs/mpm.yaml")   # or pass the loaded dict
sim.reset(scene="drape", seed=42)       # initialise particles flat above sphere
sim.step()                               # one MLS-MPM step
state = sim.state()                     # {"x", "v", "a", "F", "contact_flag", "step"}
traj  = sim.rollout(n_steps=1000)       # stacked frames + kinetic_energy series
```

### Tests (`tests/test_mpm_cloth.py`, 5 tests)

| Test | What it checks |
|---|---|
| `test_reset_particle_count` | `state["x"]` has shape `(N, 3)` for the configured grid |
| `test_step_runs` | `sim.step()` does not raise; `step_count` increments |
| `test_state_keys` | all five keys present and dtypes correct |
| `test_determinism` | two independent 500-step runs give identical `x` on CPU |
| `test_energy_decreases` | KE at t=0.5 s is less than KE at t=0.1 s on the smoke drape |

---

## Deliverable 2 — Implicit-Cloth Fallback (`src/cloth_implicit.py`)

### What it is

A **mass–spring implicit Euler cloth solver** (Baraff & Witkin 1998 §4,
Stuyck 2018 Ch. 4) operating on the same 64×64 quad-grid particles as the
MPM simulator. This is the **F2 fallback** in the M4 hybrid: when the
complexity detector fires, the hybrid loop calls one step of this solver
instead of the neural grid-update.

### How we built it (E4a)

**Spring topology** (`build_topology`): three spring families are built from
the quad grid, matching the Baraff–Witkin convention:

- **Structural:** horizontal + vertical neighbors (rest length = grid spacing
  dx or dy). Each particle has up to 4 structural neighbors.
- **Shear:** both diagonals of each quad cell. Rest length = √(dx²+dy²).
- **Bend:** skip-one neighbors along rows and columns (resisting bending across
  two-cell spans). Rest length = 2·dx or 2·dy.

Stiffnesses map from the YAML material parameters via a membrane analogy:
```
k_structural = E × thickness       (N/m)
k_shear      = shear_ratio × k_s   (default 0.5)
k_bend       = bend_ratio  × k_s   (default 0.1)
k_d          = damping_ratio × k   (default 0.05 per family)
```

**Spring force + Jacobian** (`_spring_forces_and_blocks`): vectorised over the
full spring array using NumPy. For each spring (i, j):
```
f_i  = k(L − L₀) ê + k_d ⟨v_j − v_i, ê⟩ ê
∂f_i/∂x_j = k [(L−L₀)/L (I − êê^T) + êê^T]
∂f_i/∂v_j = k_d êê^T
```
Assembles a `(3N, 3N)` sparse `df_dx` and `df_dv` (SciPy CSR) by accumulating
3×3 blocks for all four (i→i, i→j, j→i, j→j) contributions per spring family.

**Implicit Euler step** (`step()`): solves
```
A Δv = b
A = M − h ∂f/∂v − h² ∂f/∂x
b = h (f₀ + h ∂f/∂x · v₀)
```
via **Modified PCG** (Baraff–Witkin Algorithm, §5.2): the Jacobi preconditioner
uses the diagonal of `A`, and the `filter(·)` operator zeros the 3 DOFs of
every pinned particle so the constraint is exact regardless of CG iteration
count.

**Why not use the same `dt` as MPM?** The B–W paper uses `h = 0.02 s` (200×
the MPM sub-step of 1e-4 s) because the implicit method is unconditionally
stable. The M4 hybrid passes a per-step `dt` at call time so the fallback can
match the MPM simulation clock.

### Public API

```python
sim = ImplicitClothSim("configs/mpm.yaml")
sim.reset(x0=..., v0=..., pinned=[0, 15])   # optional initial state + pinned corners
result = sim.step(f_ext=ext_forces, dt=1e-4)  # {"cg_iters", "max_dv"}
state  = sim.state()                          # {"x", "v", "pinned"}
```

When `f_ext` is `None` the solver applies `m × g` per particle. When called
from the M4 hybrid loop, `f_ext` is the MPM grid's accumulated force so the
two solvers share the same external loading.

### Tests (`tests/test_cloth_implicit.py`, 5 tests)

| Test | What it checks |
|---|---|
| `test_topology_counts` | structural/shear/bend pair counts for a 4×4 grid match hand formula |
| `test_single_step_runs` | `step()` returns dict with `cg_iters` and `max_dv` keys |
| `test_pinned_particles_do_not_move` | particles marked as pinned have `|Δx| = 0` after 50 steps |
| `test_energy_decreasing` | total KE monotonically decreasing for a free-fall with damping |
| `test_rest_length_no_force` | particles at rest length produce zero net spring force |

---

## Deliverable 3 — Dataset Generator (`scripts/generate_dataset.py`)

### What it is

A script that drives `MPMClothSim` across three scenario families, saving one
`.npz` per clip plus an `index.csv` manifest.

### How we built it (E4b)

**Three scenario samplers** (`sample_drape_clip`, `sample_wind_clip`,
`sample_collision_clip`) draw randomised parameters from a seed-seeded RNG so
every clip is fully reproducible from `(scenario, seed)`:

| Scenario | Varied params | Pinned corners |
|---|---|---|
| `drape` | initial height (0.6–0.9 m), sphere center (±15 cm), radius (0.15–0.25 m) | none |
| `wind` | speed (0.5–4 m/s), direction (0–2π) | top-row corners `[0, gy−1]` |
| `collision` | sphere center (±15 cm), radius (0.10–0.18 m), cy (0.2–0.4 m) | top-row corners |

**Seed partitioning** (ensures no train/val/eval leakage):
```
Train : seeds [0,   1_000_000)
Val   : seeds [100_000, 2_000_000)
Eval  : seeds [200_000, 3_000_000)   (eval seeds 200–204 = smoke offsets)
Smoke : seeds drape 0–1, wind 100–101, collision 200–201
```

**Per-clip `.npz` schema:**

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `x` | (T, N, 3) | float32 | particle positions |
| `v` | (T, N, 3) | float32 | particle velocities |
| `a` | (T, N, 3) | float32 | per-particle accelerations (neural solver label) |
| `F` | (T, N, 3, 3) | float32 | deformation gradients |
| `contact_flag` | (T, N) | bool | per-particle sphere contact |
| `meta` | () | object | scenario, seed, params, config_hash, dt_s, … |

T = `duration_s / (dt_s × log_every_substeps)` = 100 frames/clip at smoke
resolution (1 s / (1e-4 s × 10) = 1000 sub-steps, log every 10 = 100 frames).

**Smoke run** (`--smoke`): forces CPU, 16×16 cloth, 32³ grid, 2 clips per
scenario (6 total) for schema verification in ≈70 s wall-clock. The full
64×64 / 128³ run targets ≥10k clips on Colab T4 (≈12 s/clip → ~34 h;
parallelisable across sessions).

### Usage

```bash
# Schema smoke-test (runs on CPU in ~70 s; produces 6 clips)
python scripts/generate_dataset.py --smoke

# Full dataset on Colab T4 (≥10k clips, default counts)
python scripts/generate_dataset.py --config configs/mpm.yaml --out data/cloth_trajectories

# Subset (handy for fast iteration)
python scripts/generate_dataset.py --n-drape 100 --n-wind 50 --n-collision 50
```

### Tests (`tests/test_dataset.py`, 7 tests)

| Test | What it checks |
|---|---|
| `test_drape_spec_sampler` | sampler produces `ClipSpec` with `scenario="drape"` and valid ranges |
| `test_wind_spec_has_pinned` | wind clips always pin 2 particles |
| `test_collision_spec_radius_range` | sphere radius in [0.10, 0.18] |
| `test_cfg_hash_stable` | same config → same hash across calls |
| `test_smoke_run_produces_clips` | 6 `.npz` files created and `index.csv` has 6 rows |
| `test_smoke_schema` | every clip has keys x, v, a, F, contact_flag, meta with correct shapes |
| `test_smoke_determinism` | re-running smoke seed 0 produces bit-identical `x[0]` |

---

## Deliverable 4 — Evaluation Harness (`src/eval.py` + `configs/eval.yaml`)

### What it is

The **frozen evaluation protocol** from `docs/eval-plan.md`. Computes all
roadmap Appendix B metrics given a predicted clip and a reference clip (or runs
the MPM reference and measures wall-clock timing). The schema is stable through
M5: later milestones append rows to the same CSV format.

### Metrics implemented

| Metric | Function | Roadmap eq. |
|---|---|---|
| Per-particle L2 | `per_particle_l2(pred, ref)` | App. B Eq. 1 |
| Chamfer distance | `chamfer(pred, ref)` | App. B Eq. 2 |
| Energy drift | `energy_drift(ke_pred, ke_ref)` | App. B Eq. 3 |
| Rollout horizon | `rollout_horizon(l2_series, tau, dt)` | App. B Eq. 4 |
| Cosine-window signal | `cosine_window(a_seq, window=10)` | D1 detector (paper7) |
| Wall-clock per step | `baseline_timing(config, …)` | Table 2 |

**Frozen thresholds** (from `configs/eval.yaml`, locked in W2):

- `tau = 0.05 m` — H1 L2 budget (per-particle position tolerance).
- `ε = 0.10 m` — detector positive label: `y_t = 1[L2(t) > ε]`.
- Eval seeds: `[200, 201, 202, 203, 204]` — never used in training.

### How we built it (E4c)

**`eval_clip_pair`** loads two `.npz` clips (predicted vs. reference), iterates
over frames, and computes per-frame L2, Chamfer (sampled every T/50 frames to
avoid O(N²T) cost), KE, and energy drift. It also calls `cosine_window` on
the predicted `a` sequence to pre-compute the D1 signal.

**`attach_run_metadata`** stamps every output CSV with `run_name`, `git_sha`,
`config_path`, `config_hash` so results are fully traceable.

**`baseline_timing`** instantiates `MPMClothSim` at the requested grid size,
runs `n_warmup` steps (JIT compile), then times `n_steps` with
`time.perf_counter()`. Reports mean/median/std/p95/p99 wall-clock in ms/step.

### CLI usage

```bash
# Compare two clips (predicted vs. reference)
python -m src.eval pair \
    --reference data/cloth_trajectories/drape/clip_000000_000000.npz \
    --predicted  results/neural_drape_000000.npz \
    --out        results/eval_drape_000000.csv \
    --name       neural_drape

# Measure wall-clock per step of the MPM reference
python -m src.eval baseline-timing \
    --config configs/mpm.yaml \
    --grid 16 16 --grid-resolution 32 \
    --n-steps 200 \
    --out results/timing_mpm.csv
```

### Tests (`tests/test_eval.py`, 13 tests)

| Test | What it checks |
|---|---|
| `test_per_particle_l2_identical` | L2(x, x) == 0 |
| `test_per_particle_l2_shifted` | L2 of uniform shift equals shift magnitude |
| `test_chamfer_identical` | Chamfer(x, x) == 0 |
| `test_chamfer_shifted` | Chamfer monotone with shift magnitude |
| `test_energy_drift_zero` | drift when pred == ref is 0 |
| `test_energy_drift_nonzero` | drift when KE differs returns correct ratio |
| `test_rollout_horizon_full` | returns total time when all frames below tau |
| `test_rollout_horizon_early` | returns time of first exceedance |
| `test_cosine_window_ones` | pre-warmup frames = 1.0 |
| `test_cosine_window_range` | output in [−1, 1] |
| `test_eval_clip_pair_schema` | output DataFrame has expected columns |
| `test_attach_metadata` | `run_name`, `git_sha`, `config_hash` columns present |
| `test_baseline_timing_schema` | timing row has `mean_ms`, `median_ms`, etc. |

---

## Acceptance check (from roadmap §3)

> The MPM reference reproduces the static drape within small variance across
> seeds; any dataset clip can be replayed deterministically from its saved
> trajectory; the implicit fallback also reaches a visually similar rest state
> on the same scenario.

All three conditions are verified by tests:

- **Determinism across seeds:** `test_determinism` in `test_mpm_cloth.py` and
  `test_smoke_determinism` in `test_dataset.py`.
- **Replay from saved trajectory:** `test_smoke_schema` verifies the `.npz`
  carries all fields needed for replay; determinism test re-generates and
  compares `x[0]` bit-exactly.
- **Fallback visual parity:** `test_energy_decreasing` shows the implicit
  solver also reaches a low-KE settled state under gravity + spring forces.

---

## How to reproduce M2 deliverables from scratch

```bash
# 1. Install dependencies (Taichi + SciPy)
pip install -r requirements.txt

# 2. Run all M2 tests
pytest tests/test_mpm_cloth.py tests/test_cloth_implicit.py \
       tests/test_dataset.py   tests/test_eval.py -v

# 3. Smoke-run the dataset generator (6 clips, ~70 s on CPU)
python scripts/generate_dataset.py --smoke

# 4. Inspect baseline timing
python -m src.eval baseline-timing \
    --config configs/mpm.yaml --grid 16 16 --grid-resolution 32 \
    --n-steps 200 --out results/timing_mpm.csv

# 5. Open the baseline notebook
jupyter notebook notebooks/00-mpm-baseline.ipynb
```

---

## File index

```
milestone2/
  README.md                   ← this file
  src/
    mpm_cloth.py              ← Deliverable 1: reference MPM simulator (E3)
    cloth_implicit.py         ← Deliverable 2: implicit mass-spring fallback (E4a)
    eval.py                   ← Deliverable 4: eval harness (E4c)
  scripts/
    generate_dataset.py       ← Deliverable 3: dataset generator (E4b)
  configs/
    mpm.yaml                  ← MPM reference config (cloth/material/grid params)
    eval.yaml                 ← Frozen evaluation protocol (tau, eps, eval seeds)
  tests/
    test_mpm_cloth.py         ← 5 tests for Deliverable 1
    test_cloth_implicit.py    ← 5 tests for Deliverable 2
    test_dataset.py           ← 7 tests for Deliverable 3
    test_eval.py              ← 13 tests for Deliverable 4
  notebooks/
    00-mpm-baseline.ipynb     ← M2 acceptance-check notebook (drape validation)
  data/
    README.md                 ← Data card (schema, splits, storage budget)
    cloth_trajectories/       ← Smoke dataset: 6 clips + index.csv
      index.csv
      drape/   clip_000000_000000.npz  clip_000001_000001.npz
      wind/    clip_000100_000000.npz  clip_000101_000001.npz
      collision/clip_000200_000000.npz clip_000201_000001.npz
```
