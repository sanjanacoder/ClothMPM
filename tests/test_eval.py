"""Tests for src/eval.py — metric primitives + CLI smoke.

We rely on the smoke dataset (test_dataset.py session fixture) being present
on disk; if it is not, this test module will skip gracefully.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eval import (chamfer, cosine_window, energy_drift, eval_clip_pair,
                      kinetic_energy, per_particle_l2, rollout_horizon,
                      attach_run_metadata)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "cloth_trajectories"


def test_l2_identical_is_zero():
    a = np.random.default_rng(0).normal(size=(64, 3))
    assert per_particle_l2(a, a) == 0.0


def test_l2_simple():
    """Two clouds offset by a constant: L2 = ||offset||."""
    a = np.zeros((10, 3))
    b = np.zeros((10, 3))
    b[:, 0] = 0.5
    assert abs(per_particle_l2(a, b) - 0.5) < 1e-9


def test_chamfer_identical_is_zero():
    a = np.random.default_rng(0).normal(size=(32, 3))
    assert abs(chamfer(a, a)) < 1e-9


def test_chamfer_simple_pair():
    """Two single-point clouds at distance d -> Chamfer = 2 * d^2."""
    a = np.array([[0.0, 0.0, 0.0]])
    b = np.array([[1.0, 0.0, 0.0]])
    assert abs(chamfer(a, b) - 2.0) < 1e-9


def test_kinetic_energy_simple():
    v = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    # KE = 0.5 * m * (1^2 + 2^2) = 0.5 * m * 5
    assert abs(kinetic_energy(v, mass_per=2.0) - 5.0) < 1e-9


def test_energy_drift_simple():
    assert energy_drift(1.1, 1.0) == pytest.approx(0.1)
    assert energy_drift(0.0, 0.0) == 0.0


def test_rollout_horizon_all_within():
    l2 = np.full(100, 0.01)  # below tau
    assert rollout_horizon(l2, tau=0.05, dt=0.01) == 100 * 0.01


def test_rollout_horizon_threshold_crossed():
    l2 = np.array([0.01, 0.02, 0.06, 0.07])
    # First t where l2 > 0.05 is index 2; horizon = 2 * dt
    assert rollout_horizon(l2, tau=0.05, dt=0.01) == pytest.approx(0.02)


def test_cosine_window_constant_zero_stays_one():
    """If accelerations are all zero (perfect rest), pre-warmup window is 1.0."""
    a = np.zeros((40, 8, 3))
    cs = cosine_window(a, window_steps=10)
    # Not asserting the post-warmup values (zero-divided handled by epsilon),
    # but the pre-warmup half must be initialized to 1.0.
    assert (cs[: 2 * 10] == 1.0).all()


def test_cosine_window_drifting_signal():
    """If acceleration direction flips between two windows, cos sim is ~-1."""
    rng = np.random.default_rng(0)
    T, N = 40, 8
    a = np.zeros((T, N, 3))
    a[:20] = rng.normal(size=(20, N, 3)) * 0.1 + np.array([1.0, 0.0, 0.0])
    a[20:] = rng.normal(size=(20, N, 3)) * 0.1 + np.array([-1.0, 0.0, 0.0])
    cs = cosine_window(a, window_steps=10)
    # At t=30: a_prev = a[10:20] (all +x), a_curr = a[20:30] (all -x) -> ~ -1
    assert cs[30] < -0.5, f"expected negative cosine at t=30, got {cs[30]}"
    # Sanity: at t=20 both windows are +x -> ~ +1
    assert cs[20] > 0.5, f"expected positive cosine at t=20, got {cs[20]}"


# -----------------------------------------------------------------------------
# CLI / pair eval against the smoke dataset
# -----------------------------------------------------------------------------

def _smoke_clips():
    idx = DATA_DIR / "index.csv"
    if not idx.exists():
        pytest.skip("smoke dataset not present; run test_dataset.py first")
    df = pd.read_csv(idx)
    drape = df[df["scenario"] == "drape"].head(2)
    if len(drape) < 2:
        pytest.skip("not enough drape clips in smoke dataset")
    return drape.iloc[0]["path"], drape.iloc[1]["path"]


def test_pair_self_eval_zero(tmp_path):
    """Identity eval (clip vs itself) must give L2 = 0 on every frame."""
    p1, _ = _smoke_clips()
    df = eval_clip_pair(ROOT / p1, ROOT / p1)
    assert (df["l2"] == 0.0).all(), f"non-zero L2 in self-eval: {df['l2'].max()}"
    # chamfer is computed on a subsampled grid of frames; the rest are NaN
    sub = df["chamfer"].dropna()
    assert (sub.abs() < 1e-9).all(), \
        f"non-zero chamfer in self-eval: {sub.abs().max()}"


def test_pair_different_clips_nonzero():
    p1, p2 = _smoke_clips()
    df = eval_clip_pair(ROOT / p1, ROOT / p2)
    # Drape clips with different seeds vary in initial conditions; L2 should be > 0.
    assert df["l2"].max() > 0.0


def test_attach_run_metadata_columns(tmp_path):
    df = pd.DataFrame({"frame": [0, 1, 2], "l2": [0.0, 0.1, 0.2]})
    df = attach_run_metadata(df, config_path=ROOT / "configs" / "eval.yaml",
                             run_name="unit_test")
    assert list(df.columns[:4]) == ["run_name", "git_sha", "config_path", "config_hash"]
    assert df["run_name"].unique().tolist() == ["unit_test"]
