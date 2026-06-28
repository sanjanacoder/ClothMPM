"""Integration tests for switching between the MPM solver and the implicit
fallback (Phase 5 / FB-5).

These test the *handoff*, not the individual solvers:
  1. Continuity  -- copying MPM state into the fallback must not jump position,
     velocity, or kinetic energy at the transition.
  2. Fallback-on-instability -- when the grid solver is marked unstable, the
     fallback takes over from the last good state and stays finite and bounded.

Note on the dt gap: the MPM step is dt=1e-4 while the implicit step is dt=0.02
(200x). Velocities carry over directly (same m/s units), so continuity holds at
the handoff step; the two trajectories then diverge by design. We assert
continuity at the seam, not long-term agreement.
"""

from copy import deepcopy

import numpy as np

from src.contact import ContactGeometry
from src.cloth_implicit import ImplicitClothSim
from src.mpm_cloth import MPMClothSim


def _implicit_cfg(mpm_cfg: dict) -> dict:
    """Reuse the MPM config (cloth/material/contact/mpm sections) and add the
    implicit-solver defaults, so both solvers share the same geometry."""
    cfg = deepcopy(mpm_cfg)
    cfg.setdefault("implicit", {})
    cfg["implicit"].setdefault("damping_per_unit_stiffness", 0.05)
    cfg["implicit"].setdefault("bend_stiffness_ratio", 0.1)
    cfg["implicit"].setdefault("shear_stiffness_ratio", 0.5)
    cfg["implicit"].setdefault("dt_s", 0.02)
    return cfg


def test_mpm_to_implicit_continuity(small_cfg):
    mpm = MPMClothSim(small_cfg)
    mpm.reset()
    for _ in range(20):
        mpm.step()
    s = mpm.state()
    ke_mpm = mpm.kinetic_energy()

    imp = ImplicitClothSim(_implicit_cfg(small_cfg))
    imp.reset(s["x"].astype(np.float64), s["v"].astype(np.float64))

    # Handoff is exact: state copied in, nothing moved yet.
    np.testing.assert_allclose(imp.state()["x"], s["x"], atol=1e-6)
    np.testing.assert_allclose(imp.state()["v"], s["v"], atol=1e-6)
    # Kinetic energy is continuous across the transition (same mass, same v).
    assert np.isclose(imp.kinetic_energy(), ke_mpm, rtol=1e-5, atol=1e-12)

    # One fallback step must not jump discontinuously from the handoff position.
    x_before = imp.state()["x"].copy()
    imp.step()
    x_after = imp.state()["x"]
    assert np.isfinite(x_after).all()
    # Loose tolerance: the two schemes integrate differently; we only require
    # there is no discontinuous jump at the seam.
    assert np.abs(x_after - x_before).max() < 1e-2
    # KE stays finite and bounded across the first fallback step (no spike).
    ke_after = imp.kinetic_energy()
    assert np.isfinite(ke_after) and ke_after < 1.0


def test_instability_triggers_fallback(small_cfg):
    mpm = MPMClothSim(small_cfg)
    mpm.reset()
    for _ in range(10):
        mpm.step()
    good = mpm.state()  # last known-good state

    # A clean step is stable -- the trigger must discriminate.
    diag_ok = mpm.step(diagnostics=True)
    assert diag_ok["is_unstable"] is False

    # Inject an instability: a grid solver that drives velocity above the CFL
    # threshold. is_unstable must fire.
    threshold = mpm.cfl_factor * mpm._d.dx / mpm._d.dt

    def blow_up(grid_v, grid_m, dt):
        out = grid_v.copy()
        out[grid_m > 0.0] += 3.0 * threshold
        return out

    diag_bad = mpm.step(grid_solver=blow_up, diagnostics=True)
    assert diag_bad["is_unstable"] is True
    assert diag_bad["max_velocity"] > threshold

    # Fallback takes over from the last good state (the bad step is discarded).
    icfg = _implicit_cfg(small_cfg)
    imp = ImplicitClothSim(icfg)
    imp.reset(good["x"].astype(np.float64), good["v"].astype(np.float64))
    for _ in range(30):
        imp.step()
    st = imp.state()

    # Continues producing finite, bounded output.
    assert np.isfinite(st["x"]).all() and np.isfinite(st["v"]).all()
    g = ContactGeometry(icfg)
    assert (st["x"] >= -1e-6).all() and (st["x"] <= g.domain + 1e-6).all()
    assert np.linalg.norm(st["v"], axis=-1).max() < 1e3
