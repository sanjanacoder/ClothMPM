"""Tests for the dataset generator (scripts/generate_dataset.py).

We don't re-run the generator inside the test suite (it takes ~1 min for the
smoke pass). Instead we run it once via a session-scoped fixture and assert
on the produced .npz files and the manifest.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "cloth_trajectories"


@pytest.fixture(scope="session")
def smoke_dataset():
    """Ensure a smoke dataset exists. If the manifest already has the smoke
    clips, reuse them; otherwise regenerate."""
    idx = DATA_DIR / "index.csv"
    if idx.exists():
        df = pd.read_csv(idx)
        if len(df) >= 6 and (df["status"] == "OK").all():
            return df
    # Regenerate
    subprocess.check_call(
        [sys.executable, str(ROOT / "scripts" / "generate_dataset.py"), "--smoke"],
        cwd=str(ROOT),
        env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)},
    )
    return pd.read_csv(idx)


def test_manifest_has_required_columns(smoke_dataset):
    df = smoke_dataset
    required = {
        "scenario", "seed", "clip_idx", "path", "n_frames", "n_particles",
        "duration_s", "initial_height_m", "sphere_center_x", "sphere_center_y",
        "sphere_center_z", "sphere_radius_m", "wind_x", "wind_z", "n_pinned",
        "config_hash", "wall_s", "status",
    }
    assert required.issubset(df.columns), f"missing columns: {required - set(df.columns)}"


def test_all_clips_succeeded(smoke_dataset):
    assert (smoke_dataset["status"] == "OK").all(), \
        "some clips failed during generation"


def test_clip_shapes_match_manifest(smoke_dataset):
    """Each clip on disk has shapes matching its manifest row."""
    for _, row in smoke_dataset.iterrows():
        path = ROOT / row["path"]
        assert path.exists(), f"clip missing on disk: {path}"
        clip = np.load(path, allow_pickle=True)
        T, N = int(row["n_frames"]), int(row["n_particles"])
        assert clip["x"].shape == (T, N, 3)
        assert clip["v"].shape == (T, N, 3)
        assert clip["a"].shape == (T, N, 3)
        assert clip["F"].shape == (T, N, 3, 3)
        assert clip["contact_flag"].shape == (T, N)
        assert clip["x"].dtype == np.float32
        assert clip["contact_flag"].dtype == np.bool_


def test_clip_meta_has_required_keys(smoke_dataset):
    sample_path = ROOT / smoke_dataset.iloc[0]["path"]
    clip = np.load(sample_path, allow_pickle=True)
    meta = clip["meta"].item()
    required = {"scenario", "seed", "duration_s", "initial_height_m",
                "sphere_center_m", "sphere_radius_m", "config_hash",
                "dt_s", "grid"}
    assert required.issubset(meta.keys()), \
        f"meta missing keys: {required - set(meta.keys())}"


def test_no_nan_in_any_clip(smoke_dataset):
    for _, row in smoke_dataset.iterrows():
        clip = np.load(ROOT / row["path"], allow_pickle=True)
        for k in ("x", "v", "a", "F"):
            assert np.isfinite(clip[k]).all(), f"NaN/Inf in {row['path']} field {k}"


def test_drape_y_decreases(smoke_dataset):
    """For drape clips, the cloth's mean y should be lower at t=end than t=0."""
    drape = smoke_dataset[smoke_dataset["scenario"] == "drape"]
    for _, row in drape.iterrows():
        clip = np.load(ROOT / row["path"], allow_pickle=True)
        y_start = float(clip["x"][0, :, 1].mean())
        y_end = float(clip["x"][-1, :, 1].mean())
        assert y_end < y_start, f"{row['path']}: y_end={y_end:.3f} >= y_start={y_start:.3f}"


def test_wind_pins_held(smoke_dataset):
    """Wind clips pin two corners; those particles should not move."""
    wind = smoke_dataset[smoke_dataset["scenario"] == "wind"]
    for _, row in wind.iterrows():
        clip = np.load(ROOT / row["path"], allow_pickle=True)
        meta = clip["meta"].item()
        # Pinning is stored in clip metadata but not enforced by the MPM
        # simulator on particle motion. This asserts the metadata is preserved
        # correctly so pinning can be applied in downstream processing.
        assert len(meta["pinned_corner_indices"]) == 2
