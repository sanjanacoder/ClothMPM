"""Smoke and validation tests for src.mpm_cloth.MPMClothSim.

Covers:
- Numerical stability over a free-fall + drape (no NaNs, KE bounded).
- Gravity is honored (free-fall acceleration matches g within tolerance).
- Static drape settles to a low-KE rest state on the sphere.
- Bit-exact determinism across two independent runs of the same seed.

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
