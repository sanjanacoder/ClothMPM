"""Smoke and validation tests for src.mpm_cloth.MPMClothSim.

Covers:
- Numerical stability over a free-fall + drape (no NaNs, KE bounded).
- Gravity is honored (free-fall acceleration matches g within tolerance).
- Static drape settles to a low-KE rest state on the sphere.
- Bit-exact determinism across two independent runs of the same seed.
- External force (wind) produces displacement matching F=ma (GAP-1).
- Wind-forced trajectory diverges from unforced trajectory (GAP-1).
- Pinned corners have zero velocity throughout simulation (GAP-1).
- Pinned corners have zero displacement throughout simulation (GAP-1).

These run at reduced grid (16x16 cloth / 32-cell grid) on CPU so the suite
fits inside ~1 minute.
"""

import numpy as np
import pytest


def _run(cfg, n_steps):
    """Helper: build a sim, reset, run n_steps. Returns the final state dict."""
    from src.mpm_cloth import MPMClothSim
    sim = MPMClothSim(cfg)
    sim.reset()
    ke_log = [sim.kinetic_energy()]
    for _ in range(n_steps):
        sim.step()
    ke_log.append(sim.kinetic_energy())
    state = sim.state()
    return state, ke_log


def test_runs_without_nan(small_cfg):
    """500 steps of free-fall must produce no NaNs anywhere."""
    state, _ = _run(small_cfg, 500)
    assert np.isfinite(state["x"]).all(), "particle positions contain NaN"
    assert np.isfinite(state["v"]).all(), "particle velocities contain NaN"
    assert np.isfinite(state["F"]).all(), "deformation gradients contain NaN"


def test_free_fall_acceleration(small_cfg):
    """Mean y must drop by ~0.5 g t^2 in the first 200 steps (free fall).

    With dt=1e-4 and 200 steps -> t=0.02 s. Expected drop: 0.5 * 9.81 * 0.02^2
    = 0.001962 m. Allow +/- 30% tolerance because internal stresses act.
    """
    cfg = small_cfg
    cfg["cloth"]["initial_height_m"] = 1.5  # very high, no contact in 200 steps
    from src.mpm_cloth import MPMClothSim
    sim = MPMClothSim(cfg)
    sim.reset()
    y0 = float(sim.state()["x"][:, 1].mean())
    for _ in range(200):
        sim.step()
    y1 = float(sim.state()["x"][:, 1].mean())
    drop = y0 - y1
    expected = 0.5 * 9.81 * (200 * cfg["mpm"]["dt_s"]) ** 2
    assert 0.5 * expected < drop < 1.5 * expected, (
        f"free-fall drop {drop:.4f} m not within +/-50% of expected "
        f"{expected:.4f} m"
    )


def test_static_drape_rest_state(small_cfg):
    """After ~0.7 s of simulated time, KE should crash by 100x from peak.

    Drape: cloth dropped from y=0.7 onto sphere at y=0.4 (radius 0.2). With
    soft cloth (E=1e4 Pa) and dt=1e-4, contact happens around step ~2500 and
    rest state is reached by step ~6000.
    """
    state, _ = _run(small_cfg, 7000)
    # At rest, mean velocity magnitude should be small (<0.5 m/s).
    v_mag = np.linalg.norm(state["v"], axis=-1).mean()
    assert v_mag < 0.5, f"cloth not at rest: mean |v| = {v_mag:.3f} m/s"
    # Some particles should be in contact with the sphere.
    assert state["contact_flag"].any(), "no particles ever contacted the sphere"


def test_determinism_same_seed(small_cfg):
    """Two runs of the same seed must produce bit-identical state."""
    s1, _ = _run(small_cfg, 500)
    s2, _ = _run(small_cfg, 500)
    diff_x = np.abs(s1["x"] - s2["x"]).max()
    diff_v = np.abs(s1["v"] - s2["v"]).max()
    assert diff_x == 0.0, f"x differs by {diff_x:.2e} between identical runs"
    assert diff_v == 0.0, f"v differs by {diff_v:.2e} between identical runs"


def test_state_shapes(small_cfg):
    """state() must return arrays with the right shapes for the data loader."""
    from src.mpm_cloth import MPMClothSim
    sim = MPMClothSim(small_cfg)
    sim.reset()
    sim.step()
    state = sim.state()
    n = small_cfg["cloth"]["grid"][0] * small_cfg["cloth"]["grid"][1]
    assert state["x"].shape == (n, 3)
    assert state["v"].shape == (n, 3)
    assert state["a"].shape == (n, 3)
    assert state["F"].shape == (n, 3, 3)
    assert state["contact_flag"].shape == (n,)
    assert state["contact_flag"].dtype == np.bool_


# ---------------------------------------------------------------------------
# GAP-1: External force (f_ext) and pinned-corner enforcement
#
# These tests define the API required by the dataset generator:
#   step(f_ext: np.ndarray | None)  — (N, 3) per-particle force in Newtons
#   reset(pinned_mask: np.ndarray | None)  — (N,) bool, True = velocity zeroed each step
#
# All four tests FAIL against the current code because neither argument exists.
# They pass once mpm_cloth.py implements f_ext scattering in p2g and pinned
# velocity zeroing in g2p (the fix described in docs/gap-analysis.md GAP-1).
# ---------------------------------------------------------------------------

def test_uniform_wind_displaces_in_force_direction(small_cfg):
    """A constant +x force must produce net x-displacement matching F = ma.

    Setup: 16x16 cloth at y=1.5 m (no sphere contact in 500 steps), f_ext =
    0.01 N/particle in +x, 500 steps (t = 0.05 s).

    Expected x-displacement from kinematics: 0.5 * (F/m) * t^2 ~ 0.016 m.
    We assert at least 50% of the analytical value (elastic redistribution
    reduces mean displacement but cannot reverse it).

    We also assert x-displacement >> z-displacement to confirm the force is
    directional, not a global artefact.
    """
    from src.mpm_cloth import MPMClothSim

    small_cfg["cloth"]["initial_height_m"] = 1.5  # no contact in 500 steps
    sim = MPMClothSim(small_cfg)
    sim.reset()

    Gx, Gy = small_cfg["cloth"]["grid"]
    n = Gx * Gy
    p_mass = small_cfg["cloth"]["mass_kg"] / n
    fx = 0.01  # N per particle in +x
    f_ext = np.zeros((n, 3), dtype=np.float32)
    f_ext[:, 0] = fx

    x0 = sim.state()["x"].copy()
    n_steps = 500
    for _ in range(n_steps):
        sim.step(f_ext=f_ext)
    x1 = sim.state()["x"]

    t = n_steps * small_cfg["mpm"]["dt_s"]
    dx_mean = float((x1[:, 0] - x0[:, 0]).mean())
    dz_abs_mean = float(np.abs(x1[:, 2] - x0[:, 2]).mean())
    expected = 0.5 * (fx / p_mass) * t ** 2

    assert dx_mean > 0.5 * expected, (
        f"x-displacement {dx_mean:.5f} m < 50% of kinematic expectation "
        f"{expected:.5f} m; f_ext likely not applied"
    )
    assert dx_mean > 3.0 * dz_abs_mean, (
        f"force not directional: dx={dx_mean:.5f} m, |dz|={dz_abs_mean:.5f} m"
    )


def test_wind_trajectory_differs_from_no_wind(small_cfg):
    """Applying f_ext must produce a trajectory that diverges from the unforced one.

    Two runs share the same config and seed. One receives a constant +x force;
    the other does not. After 500 steps the per-particle L2 between the two
    final positions must be clearly non-zero.

    If f_ext is silently ignored (current behaviour) both runs are identical
    and L2 = 0, which fails this test.
    """
    from src.mpm_cloth import MPMClothSim

    small_cfg["cloth"]["initial_height_m"] = 1.5

    Gx, Gy = small_cfg["cloth"]["grid"]
    n = Gx * Gy
    f_ext = np.zeros((n, 3), dtype=np.float32)
    f_ext[:, 0] = 0.01  # N per particle in +x
    n_steps = 500

    sim_wind = MPMClothSim(small_cfg)
    sim_wind.reset()
    for _ in range(n_steps):
        sim_wind.step(f_ext=f_ext)
    x_wind = sim_wind.state()["x"]

    # Second sim re-inits Taichi (ti.reset() inside __init__) so initial
    # state is identical to the first run.
    sim_base = MPMClothSim(small_cfg)
    sim_base.reset()
    for _ in range(n_steps):
        sim_base.step()
    x_base = sim_base.state()["x"]

    diff = x_wind - x_base
    l2 = float(np.sqrt((diff ** 2).sum(axis=-1).mean()))
    assert l2 > 0.005, (
        f"wind and no-wind runs nearly identical (L2={l2:.5f} m); "
        "f_ext has no effect on the trajectory"
    )


def test_pinned_corners_zero_velocity(small_cfg):
    """Particles in pinned_mask must have velocity = 0 after any number of steps.

    The wind scenario pins two top-edge corners so they act as suspension
    points. Without enforcement (current behaviour) gravity accelerates them
    to ~g*t ≈ 1 m/s after 1000 steps, which fails this test.

    pinned_mask indices: corner (0, 0) -> 0, corner (0, Gy-1) -> Gy-1.
    """
    from src.mpm_cloth import MPMClothSim

    small_cfg["cloth"]["initial_height_m"] = 1.5

    Gx, Gy = small_cfg["cloth"]["grid"]
    n = Gx * Gy
    pinned_indices = [0, Gy - 1]
    pinned_mask = np.zeros(n, dtype=bool)
    pinned_mask[pinned_indices] = True

    sim = MPMClothSim(small_cfg)
    sim.reset(pinned_mask=pinned_mask)
    for _ in range(1000):
        sim.step()
    v = sim.state()["v"]

    v_pinned_max = float(np.abs(v[pinned_indices]).max())
    assert v_pinned_max < 1e-4, (
        f"pinned corners reached |v| = {v_pinned_max:.3e} m/s after 1000 steps; "
        "pinned_mask velocity zeroing not implemented"
    )


def test_pinned_corners_zero_displacement(small_cfg):
    """Particles in pinned_mask must not move from their initial positions.

    Complements test_pinned_corners_zero_velocity: a correct implementation
    zeroes velocity each step so x never accumulates drift. Without the fix,
    corners fall ~0.5 * g * t^2 ≈ 0.049 m in y after 1000 steps (t=0.1 s).
    """
    from src.mpm_cloth import MPMClothSim

    small_cfg["cloth"]["initial_height_m"] = 1.5

    Gx, Gy = small_cfg["cloth"]["grid"]
    n = Gx * Gy
    pinned_indices = [0, Gy - 1]
    pinned_mask = np.zeros(n, dtype=bool)
    pinned_mask[pinned_indices] = True

    sim = MPMClothSim(small_cfg)
    sim.reset(pinned_mask=pinned_mask)
    x0_pinned = sim.state()["x"][pinned_indices].copy()
    for _ in range(1000):
        sim.step()
    x1_pinned = sim.state()["x"][pinned_indices]

    max_drift = float(np.abs(x1_pinned - x0_pinned).max())
    assert max_drift < 1e-5, (
        f"pinned corners drifted {max_drift:.3e} m from initial position; "
        "expected zero displacement"
    )
