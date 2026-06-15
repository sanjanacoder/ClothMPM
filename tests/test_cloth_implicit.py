"""Tests for src.cloth_implicit.ImplicitClothSim.

Covers:
- Spring topology counts on a 16x16 grid (struct + shear + bend).
- Pinned corners stay exactly at their initial position regardless of CG
  iteration count (Baraff-Witkin filter property -- the load-bearing claim).
- Free fall reaches gravity-consistent velocity (no internal forces yet).
- CG converges within the iteration budget on a typical step.
- Determinism: same x0/v0 -> same trajectory.
"""

from copy import deepcopy

import numpy as np
import pytest


def _small_cfg(base):
    cfg = deepcopy(base)
    cfg["cloth"]["grid"] = [16, 16]
    cfg["cloth"]["initial_height_m"] = 1.0
    cfg["implicit"]["dt_s"] = 0.005
    return cfg


@pytest.fixture
def small_implicit_cfg(base_cfg):
    """Reuses the mpm.yaml fixture; load_implicit_config adds the implicit block."""
    from src.cloth_implicit import load_implicit_config
    cfg = load_implicit_config()  # reads configs/mpm.yaml
    return _small_cfg(cfg)


def test_topology_counts(small_implicit_cfg):
    """A 16x16 grid has the expected spring counts."""
    from src.cloth_implicit import ImplicitClothSim
    sim = ImplicitClothSim(small_implicit_cfg)
    # Structural: (gx-1)*gy + gx*(gy-1) = 2 * 15 * 16 = 480
    assert sim.topology.structural_pairs.shape == (480, 2)
    # Shear: 2 * (gx-1) * (gy-1) = 2 * 15 * 15 = 450
    assert sim.topology.shear_pairs.shape == (450, 2)
    # Bend: (gx-2)*gy + gx*(gy-2) = 14*16 + 16*14 = 448
    assert sim.topology.bend_pairs.shape == (448, 2)


def test_pinned_corners_exact(small_implicit_cfg):
    """Pinned particles must remain at their initial position regardless of
    CG iteration count -- this is the Baraff-Witkin filter property."""
    from src.cloth_implicit import ImplicitClothSim
    sim = ImplicitClothSim(small_implicit_cfg)
    gy = small_implicit_cfg["cloth"]["grid"][1]
    pinned = [0, gy - 1]  # top-left and top-right of row 0
    sim.reset(pinned=pinned)
    initial = sim.x[pinned].copy()
    # Run for 100 steps with intentionally low CG iters -- pinned corners must
    # still be exact even when CG is cut short.
    for _ in range(100):
        sim.step(cg_max_iters=5, cg_tol=1e-2)
    after = sim.x[pinned]
    assert np.allclose(after, initial, atol=1e-12), (
        f"pinned corners drifted: max delta = {np.abs(after - initial).max():.2e}"
    )


def test_free_fall_no_springs(small_implicit_cfg):
    """With all spring stiffnesses zeroed, the cloth must fall under gravity
    at the analytic rate.

    This is the cleanest sanity check that the implicit solver doesn't
    introduce extra damping outside the spring system.
    """
    from src.cloth_implicit import ImplicitClothSim
    cfg = deepcopy(small_implicit_cfg)
    cfg["material"]["young_modulus_warp_pa"] = 0.0
    cfg["material"]["young_modulus_weft_pa"] = 0.0
    sim = ImplicitClothSim(cfg)
    sim.reset(pinned=None)
    h = cfg["implicit"]["dt_s"]
    n_steps = 20
    for _ in range(n_steps):
        sim.step()
    t = n_steps * h
    expected_dy = 0.5 * 9.81 * t * t       # 0.5 g t^2
    actual_dy = float(sim.x[:, 1].mean()) - cfg["cloth"]["initial_height_m"]
    actual_dy = -actual_dy   # we expect a drop, expected is positive magnitude
    assert abs(actual_dy - expected_dy) / expected_dy < 0.05, (
        f"free fall drop {actual_dy:.4f} m differs from expected {expected_dy:.4f} m by "
        f"more than 5%"
    )


def test_cg_converges_within_budget(small_implicit_cfg):
    """In a typical drape step, modified PCG should converge in < 50 iters."""
    from src.cloth_implicit import ImplicitClothSim
    sim = ImplicitClothSim(small_implicit_cfg)
    sim.reset()
    # warm up so springs are loaded
    for _ in range(10):
        sim.step()
    info = sim.step(cg_max_iters=80, cg_tol=1e-4)
    assert info["cg_iters"] < 50, f"CG didn't converge within 50 iters (took {info['cg_iters']})"


def test_determinism(small_implicit_cfg):
    """Two runs from the same x0/v0 must produce bit-identical trajectories."""
    from src.cloth_implicit import ImplicitClothSim

    def go():
        s = ImplicitClothSim(small_implicit_cfg)
        s.reset()
        for _ in range(20):
            s.step()
        return s.state()

    a = go()
    b = go()
    assert np.abs(a["x"] - b["x"]).max() == 0.0
    assert np.abs(a["v"] - b["v"]).max() == 0.0
