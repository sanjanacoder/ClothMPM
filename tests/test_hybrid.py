"""Tests for src.hybrid.HybridRollout — neural and fallback paths,
detector dispatch, no-NaN, fallback fraction sanity."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from src.cloth_implicit import ImplicitClothSim, load_implicit_config
from src.data import NormalizationStats
from src.detector import CosineThresholdDetector
from src.hybrid import HybridRollout, _approximate_C_from_velocities
from src.neural_solver import build_solver

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "results" / "checkpoints" / "mlp_smoke" / "best.pt"


@pytest.fixture(scope="module")
def smoke_mlp_ckpt():
    """Reuse the smoke MLP checkpoint produced by E5's training run.
    If it doesn't exist (fresh clone), skip the hybrid tests."""
    if not CKPT.exists():
        pytest.skip("smoke MLP checkpoint missing; run src.train --name mlp_smoke first")
    return torch.load(CKPT, map_location="cpu", weights_only=False)


def _make_initial_state(grid_x: int = 16, grid_y: int = 16, height: float = 0.7):
    i, j = np.meshgrid(np.arange(grid_x), np.arange(grid_y), indexing='ij')
    x0 = np.stack([
        (i + 0.5) / grid_x + 0.5,
        np.full_like(i, height, dtype=float),
        (j + 0.5) / grid_y + 0.5,
    ], axis=-1).reshape(-1, 3).astype(np.float32)
    return x0, grid_x, grid_y


def _make_hybrid(ckpt, threshold: float, fallback_dt: float = 1e-3):
    stats = NormalizationStats.from_dict(ckpt["stats"])
    model = build_solver(ckpt["model_cfg"]).eval()
    model.load_state_dict(ckpt["model"])
    icfg = load_implicit_config()
    icfg["cloth"]["grid"] = [16, 16]
    icfg["cloth"]["initial_height_m"] = 0.7
    icfg["implicit"]["dt_s"] = fallback_dt
    fb = ImplicitClothSim(icfg)
    fb.reset()
    return HybridRollout(
        model=model, model_kind="mlp", stats=stats, dt=1e-3,
        detector=CosineThresholdDetector(rc=0.8),
        threshold=threshold,
        fallback_sim=fb,
        history_C=5,
        cosine_window_steps=10,
        device=torch.device("cpu"),
    )


# -----------------------------------------------------------------------------
# State helpers
# -----------------------------------------------------------------------------

def test_approximate_C_constant_velocity_is_zero():
    """If every particle has the same velocity, C must be zero."""
    n = 16
    x = torch.randn(n, 3)
    v = torch.full((n, 3), 1.0)
    from src.data import build_kring_index
    kring = build_kring_index(4, 4)
    C = _approximate_C_from_velocities(v, x, kring)
    # Allow small numerical error from the eps regularizer
    assert C.abs().max().item() < 1e-3


def test_approximate_C_linear_velocity_field():
    """v(x) = A x => C should converge to A on the in-plane components.

    Cloth particles live on the xy plane in this test (z=0), so the affine
    velocity gradient is genuinely under-determined in the z-row of A; we
    check only the 2x2 in-plane block.
    """
    n = 16
    rng = np.random.default_rng(0)
    A = torch.tensor(rng.normal(scale=0.5, size=(3, 3)), dtype=torch.float32)
    from src.data import build_kring_index
    kring = build_kring_index(4, 4)
    i, j = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
    x = torch.tensor(np.stack([i.ravel(), j.ravel(), np.zeros(16)], axis=-1),
                     dtype=torch.float32)
    v = (A @ x.T).T
    C = _approximate_C_from_velocities(v, x, kring)
    # In-plane block: rows 0-1, cols 0-1. We can recover this exactly (modulo
    # the boundary-pad-with-self degradation).
    err_xy = (C[:, :2, :2] - A[:2, :2]).abs().reshape(n, -1).max(dim=-1).values
    assert err_xy.median().item() < 0.1, f"in-plane median err {err_xy.median().item()}"


# -----------------------------------------------------------------------------
# Hybrid rollout
# -----------------------------------------------------------------------------

def test_hybrid_runs_no_nan(smoke_mlp_ckpt):
    h = _make_hybrid(smoke_mlp_ckpt, threshold=0.5)
    x0, gx, gy = _make_initial_state()
    h.reset(x0=x0, grid_x=gx, grid_y=gy)
    out = h.rollout(50)
    assert not np.isnan(out["x"]).any()
    assert not np.isnan(out["v"]).any()


def test_hybrid_threshold_zero_uses_fallback(smoke_mlp_ckpt):
    """With threshold = 0 the detector always fires once warmup ends."""
    h = _make_hybrid(smoke_mlp_ckpt, threshold=-1.0)
    x0, gx, gy = _make_initial_state()
    h.reset(x0=x0, grid_x=gx, grid_y=gy)
    out = h.rollout(40)
    # First 20 steps are warmup (no fallback). After that, every step fires.
    used = [r["used_fallback"] for r in out["rows"]]
    assert sum(used[:20]) == 0
    assert sum(used[20:]) > 0


def test_hybrid_threshold_high_uses_neural(smoke_mlp_ckpt):
    """With threshold = 1.5 (above any cosine score's max of 1.0) the detector
    never fires -- pure neural rollout."""
    h = _make_hybrid(smoke_mlp_ckpt, threshold=2.0)
    x0, gx, gy = _make_initial_state()
    h.reset(x0=x0, grid_x=gx, grid_y=gy)
    out = h.rollout(30)
    assert out["fallback_fraction"] == 0.0


def test_hybrid_step_count_consistent(smoke_mlp_ckpt):
    h = _make_hybrid(smoke_mlp_ckpt, threshold=0.5)
    x0, gx, gy = _make_initial_state()
    h.reset(x0=x0, grid_x=gx, grid_y=gy)
    out = h.rollout(35, log_every=5)
    assert len(out["rows"]) == 35
    assert out["x"].shape[0] == 7   # ceil(35 / 5)
    assert all(r["step"] == i + 1 for i, r in enumerate(out["rows"]))


def test_reset_seeds_velocity_history(smoke_mlp_ckpt):
    """The rollout must accept and use the true velocity history. The model
    reads acceleration from the velocity *slope*, so a flat (v0-repeated)
    history vs a real sloped one must give different predictions -- this is the
    seeding bug that caused the spurious rollout collapse."""
    h = _make_hybrid(smoke_mlp_ckpt, threshold=float("inf"))
    x0, gx, gy = _make_initial_state()
    N, C = x0.shape[0], h.C
    # a sloped history (velocity ramping downward over the C frames)
    vh = np.stack([np.full((N, 3), -0.02 * k, dtype=np.float32) for k in range(C)])
    h.reset(x0=x0, v0=vh[-1], grid_x=gx, grid_y=gy, v_history=vh)
    seeded = torch.stack(list(h._v_history)).numpy()
    assert np.allclose(seeded, vh, atol=1e-6), "reset must seed the passed history"
    a_sloped = h._neural_accel().detach().clone()

    # default (flat) seeding repeats v0 -> zero slope -> different prediction
    h.reset(x0=x0, v0=vh[-1], grid_x=gx, grid_y=gy)
    a_flat = h._neural_accel().detach()
    assert not torch.allclose(a_sloped, a_flat, atol=1e-4), \
        "velocity-history slope must affect the predicted acceleration"
