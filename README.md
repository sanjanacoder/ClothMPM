# ClothMPM — Physics-Based Cloth Simulation

A 3D cloth simulator built on Material Point Method (MLS-MPM) physics, with an implicit mass-spring fallback, a dataset generator, and an evaluation harness. Runs on CPU for local development; designed to scale on GPU (Colab T4).

---

## Overview

The project has four main components:

| Component | File(s) | Purpose |
|---|---|---|
| MPM cloth simulator | `src/mpm_cloth.py` | Reference physics engine |
| Implicit cloth fallback | `src/cloth_implicit.py` | Stable mass-spring solver |
| Dataset generator | `scripts/generate_dataset.py` | Generates trajectory clips |
| Evaluation harness | `src/eval.py` | Accuracy metrics and timing benchmarks |

---

## Setup

Python 3.10 or 3.11 is recommended (Taichi 1.7 does not support Python 3.12+).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Verify the installation:

```bash
pytest tests/ -v
```

All 30 tests should pass on CPU without any GPU or special hardware.

---

## MPM Cloth Simulator (`src/mpm_cloth.py`)

A 3D MLS-MPM cloth simulator wrapping Taichi's MLS-MPM kernel with a linear co-rotational anisotropic constitutive model (separate warp/weft Young's moduli). A 64×64 quad-grid cloth sheet (~4096 particles) falls under gravity onto a sphere collider.

### Physics

The simulator uses a linear co-rotational membrane energy:

```
ψ = μ ‖F − R‖² + ½ λ (J − 1)²
```

computed per triangle in the xz plane of each quad-cell's two triangles. The 2×2 membrane deformation gradient F_mem is obtained by projecting 3D edge vectors onto the cloth's rest-pose plane. Polar decomposition gives R; `ti.ad.Tape` differentiates through the energy to produce Lagrangian forces via autodiff.

One simulation step runs the standard MLS-MPM cycle:

```
clear_grid()
total_energy[None] = 0
with ti.ad.Tape(total_energy):
    compute_total_energy()   # sets x.grad via autodiff
p2g()                        # scatter mass + momentum + force to 128³ grid
grid_update()                # normalize, add gravity, sphere + box projection
g2p()                        # interpolate back; update a, v, x, C, F
```

Contact is handled by projecting grid node velocities when `|pos − sphere_center| < sphere_radius` (frictionless normal projection). Box bounds clamp all six faces with a 3-cell margin.

### API

```python
sim = MPMClothSim("configs/mpm.yaml")   # or pass the loaded dict
sim.reset(scene="drape", seed=42)       # initialise particles flat above sphere
sim.step()                               # one MLS-MPM step
state = sim.state()                     # {"x", "v", "a", "F", "contact_flag", "step"}
traj  = sim.rollout(n_steps=1000)       # stacked frames + kinetic_energy series
```

Key config values (`configs/mpm.yaml`): 64×64 cloth, 128³ grid, dt=1e-4 s, E=10⁴ Pa, ν=0.3. The cloth is 1 m × 1 m, total mass 0.2 kg, dropped from y=1.5 m.

### Tests (`tests/test_mpm_cloth.py`)

| Test | What it checks |
|---|---|
| `test_reset_particle_count` | `state["x"]` has shape `(N, 3)` for the configured grid |
| `test_step_runs` | `sim.step()` does not raise; `step_count` increments |
| `test_state_keys` | all five keys present and dtypes correct |
| `test_determinism` | two independent 500-step runs give identical `x` on CPU |
| `test_energy_decreases` | KE at t=0.5 s is less than KE at t=0.1 s on the smoke drape |

---

## Implicit Cloth Fallback (`src/cloth_implicit.py`)

A mass-spring implicit Euler cloth solver (Baraff & Witkin 1998 §4, Stuyck 2018 Ch. 4) operating on the same 64×64 quad-grid particles. It serves as a stable fallback for high-complexity scenarios where MPM accuracy degrades.

### Physics

**Spring topology** (`build_topology`): three spring families built from the quad grid:

- **Structural:** horizontal + vertical neighbors (rest length = grid spacing dx or dy). Each particle has up to 4 structural neighbors.
- **Shear:** both diagonals of each quad cell. Rest length = √(dx²+dy²).
- **Bend:** skip-one neighbors along rows and columns. Rest length = 2·dx or 2·dy.

Stiffnesses map from the YAML material parameters via a membrane analogy:
```
k_structural = E × thickness       (N/m)
k_shear      = shear_ratio × k_s   (default 0.5)
k_bend       = bend_ratio  × k_s   (default 0.1)
k_d          = damping_ratio × k   (default 0.05 per family)
```

**Implicit Euler step** solves:
```
A Δv = b
A = M − h ∂f/∂v − h² ∂f/∂x
b = h (f₀ + h ∂f/∂x · v₀)
```
via Modified PCG (Baraff–Witkin §5.2). The Jacobi preconditioner uses the diagonal of A; a filter operator zeros the DOFs of pinned particles so constraints are exact regardless of iteration count.

Because implicit Euler is unconditionally stable, the fallback can take larger timesteps (e.g. `h = 0.02 s`) than the MPM sub-step (1e-4 s). The caller passes `dt` explicitly so the two solvers can share the same simulation clock.

### API

```python
sim = ImplicitClothSim("configs/mpm.yaml")
sim.reset(x0=..., v0=..., pinned=[0, 15])   # optional initial state + pinned corners
result = sim.step(f_ext=ext_forces, dt=1e-4)  # {"cg_iters", "max_dv"}
state  = sim.state()                          # {"x", "v", "pinned"}
```

When `f_ext` is `None` the solver applies `m × g` per particle.

### Tests (`tests/test_cloth_implicit.py`)

| Test | What it checks |
|---|---|
| `test_topology_counts` | structural/shear/bend pair counts for a 4×4 grid match hand formula |
| `test_single_step_runs` | `step()` returns dict with `cg_iters` and `max_dv` keys |
| `test_pinned_particles_do_not_move` | particles marked as pinned have `|Δx| = 0` after 50 steps |
| `test_energy_decreasing` | total KE monotonically decreasing for free-fall with damping |
| `test_rest_length_no_force` | particles at rest length produce zero net spring force |

---

## Dataset Generator (`scripts/generate_dataset.py`)

Drives `MPMClothSim` across three scenario families, saving one `.npz` per clip plus an `index.csv` manifest.

### Scenarios

| Scenario | Varied params | Pinned corners |
|---|---|---|
| `drape` | initial height (0.6–0.9 m), sphere center (±15 cm), radius (0.15–0.25 m) | none |
| `wind` | speed (0.5–4 m/s), direction (0–2π) | top-row corners `[0, gy−1]` |
| `collision` | sphere center (±15 cm), radius (0.10–0.18 m), cy (0.2–0.4 m) | top-row corners |

Parameters are drawn from a seed-seeded RNG, making every clip fully reproducible from `(scenario, seed)`. Seeds are partitioned to prevent train/val/eval leakage:

```
Train : seeds [0,       1_000_000)
Val   : seeds [100_000, 2_000_000)
Eval  : seeds [200_000, 3_000_000)
```

### Clip format (`.npz`)

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `x` | (T, N, 3) | float32 | particle positions |
| `v` | (T, N, 3) | float32 | particle velocities |
| `a` | (T, N, 3) | float32 | per-particle accelerations |
| `F` | (T, N, 3, 3) | float32 | deformation gradients |
| `contact_flag` | (T, N) | bool | per-particle sphere contact |
| `meta` | () | object | scenario, seed, params, config_hash, dt_s, … |

T = 100 frames/clip at default resolution (1 s / (1e-4 s × 10) = 1000 sub-steps, logged every 10).

### Usage

```bash
# Quick schema smoke-test (~70 s on CPU, produces 6 clips)
python scripts/generate_dataset.py --smoke

# Full dataset on Colab T4 (≥10k clips, ~12 s/clip on GPU)
python scripts/generate_dataset.py --config configs/mpm.yaml --out data/cloth_trajectories

# Subset for fast iteration
python scripts/generate_dataset.py --n-drape 100 --n-wind 50 --n-collision 50
```

### Tests (`tests/test_dataset.py`)

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

## Evaluation Harness (`src/eval.py`)

Computes physics-accuracy metrics given a predicted clip and a reference clip, or benchmarks the MPM simulator directly.

### Metrics

| Metric | Function | Description |
|---|---|---|
| Per-particle L2 | `per_particle_l2(pred, ref)` | Mean per-particle position error |
| Chamfer distance | `chamfer(pred, ref)` | Point-cloud distance between frames |
| Energy drift | `energy_drift(ke_pred, ke_ref)` | KE ratio between predicted and reference |
| Rollout horizon | `rollout_horizon(l2_series, tau, dt)` | Time until per-particle L2 exceeds threshold |
| Cosine-window signal | `cosine_window(a_seq, window=10)` | Windowed acceleration signal for complexity detection |
| Wall-clock per step | `baseline_timing(config, …)` | Mean/median/p95/p99 step time in ms |

Thresholds are configured in `configs/eval.yaml`:
- `tau = 0.05 m` — per-particle position tolerance for rollout horizon
- `ε = 0.10 m` — complexity detector threshold

### CLI

```bash
# Compare two clips (predicted vs. reference)
python -m src.eval pair \
    --reference data/cloth_trajectories/drape/clip_000000_000000.npz \
    --predicted  results/neural_drape_000000.npz \
    --out        results/eval_drape_000000.csv \
    --name       neural_drape

# Measure wall-clock per step
python -m src.eval baseline-timing \
    --config configs/mpm.yaml \
    --grid 16 16 --grid-resolution 32 \
    --n-steps 200 \
    --out results/timing_mpm.csv
```

### Tests (`tests/test_eval.py`, 13 tests)

Covers all metrics (`per_particle_l2`, `chamfer`, `energy_drift`, `rollout_horizon`, `cosine_window`), the `eval_clip_pair` output schema, metadata attachment, and timing output schema.

---

## Quickstart

```bash
# 1. Set up the environment
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Run all tests
pytest tests/ -v

# 3. Smoke-run the dataset generator (6 clips, ~70 s on CPU)
python scripts/generate_dataset.py --smoke

# 4. Check simulator timing
python -m src.eval baseline-timing \
    --config configs/mpm.yaml --grid 16 16 --grid-resolution 32 \
    --n-steps 200 --out results/timing_mpm.csv

# 5. Open the baseline notebook
jupyter notebook notebooks/00-mpm-baseline.ipynb
```

---

## File structure

```
ClothMPM/
  src/
    mpm_cloth.py              # Reference MPM simulator
    cloth_implicit.py         # Implicit mass-spring fallback
    eval.py                   # Evaluation harness
  scripts/
    generate_dataset.py       # Dataset generator
  configs/
    mpm.yaml                  # MPM config (cloth, material, grid params)
    eval.yaml                 # Evaluation thresholds and eval seeds
  tests/
    test_mpm_cloth.py         # 5 tests for MPM simulator
    test_cloth_implicit.py    # 5 tests for implicit solver
    test_dataset.py           # 7 tests for dataset generator
    test_eval.py              # 13 tests for eval harness
  notebooks/
    00-mpm-baseline.ipynb     # Drape validation and sanity plots
  data/
    README.md                 # Data card (schema, splits, storage budget)
    cloth_trajectories/       # Dataset clips + index.csv
      index.csv
      drape/      clip_000000_000000.npz  clip_000001_000001.npz
      wind/       clip_000100_000000.npz  clip_000101_000001.npz
      collision/  clip_000200_000000.npz  clip_000201_000001.npz
```
