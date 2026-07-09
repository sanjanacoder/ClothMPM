"""Tests for src.data: k-ring topology, feature assembly, dataset."""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.data import (NormalizationStats, TrajectoryDataset, assemble_features,
                      build_kring_index, fit_normalization_stats)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "cloth_trajectories"


def test_kring_4x4_interior_particle():
    """For a 4x4 grid, particle (1, 1) at index 5 should have 8 valid neighbors:
    (0,1), (0,2), (1,2), (2,2), (2,1), (2,0), (1,0), (0,0).
    On row-major indexing (i*gy + j), those map to 1, 2, 6, 10, 9, 8, 4, 0.
    """
    kring = build_kring_index(4, 4)
    assert kring[5].tolist() == [1, 2, 6, 10, 9, 8, 4, 0]


def test_kring_corner_particle_pads_with_self():
    """The (0,0) corner has only 3 valid neighbors; the missing 5 must equal
    its own index (pad-with-self)."""
    kring = build_kring_index(4, 4)
    n = kring[0].tolist()
    valid_neighbors = {1, 4, 5}
    self_pads = sum(1 for k in n if k == 0)
    real = sum(1 for k in n if k in valid_neighbors)
    assert real == 3
    assert self_pads == 5


def test_assemble_features_shapes():
    """Feature tensor for a tiny 2x2 grid must be (4, 9, F)."""
    gx, gy = 2, 2
    n = gx * gy
    C = 5
    x = torch.randn(n, 3)
    v_history = torch.randn(n, C, 3)
    F = torch.eye(3).expand(n, 3, 3).contiguous()
    f_ext = torch.zeros(n, 3)
    kring = build_kring_index(gx, gy)
    feats = assemble_features(x, v_history, F, f_ext, kring)
    assert feats.shape == (n, 9, 30)
    # Self position must always be (0,0,0).
    assert (feats[:, 0, :3] == 0).all()


def test_assemble_features_translation_invariance():
    """Translating all positions by a constant must not change the features."""
    gx, gy = 4, 4
    n = gx * gy
    C = 5
    x0 = torch.randn(n, 3)
    v = torch.randn(n, C, 3)
    F = torch.eye(3).expand(n, 3, 3).contiguous()
    f_ext = torch.zeros(n, 3)
    kring = build_kring_index(gx, gy)
    f1 = assemble_features(x0, v, F, f_ext, kring)
    f2 = assemble_features(x0 + 7.5, v, F, f_ext, kring)
    assert torch.allclose(f1, f2, atol=1e-6)


# -----------------------------------------------------------------------------
# Dataset tests (require the smoke dataset to exist)
# -----------------------------------------------------------------------------

def _ensure_smoke_dataset():
    if not (DATA_DIR / "index.csv").exists():
        pytest.skip("smoke dataset not present; run scripts/generate_dataset.py --smoke")


def test_dataset_length_and_shapes():
    _ensure_smoke_dataset()
    ds = TrajectoryDataset(DATA_DIR / "index.csv", velocity_history_C=5)
    assert len(ds) > 0
    feats, target = ds[0]
    assert feats.shape == (ds.N, 9, 30)
    assert target.shape == (ds.N, 3)
    assert feats.dtype == torch.float32
    assert target.dtype == torch.float32


def test_normalization_round_trip():
    _ensure_smoke_dataset()
    ds_raw = TrajectoryDataset(DATA_DIR / "index.csv")
    stats = fit_normalization_stats(ds_raw, n_samples=50)
    # Round-trip via from_dict / to_dict
    s2 = NormalizationStats.from_dict(stats.to_dict())
    assert torch.allclose(s2.feat_mean, stats.feat_mean)
    assert torch.allclose(s2.target_std, stats.target_std)
    # Normalized dataset returns finite values
    ds = TrajectoryDataset(DATA_DIR / "index.csv", stats=stats)
    feats, target = ds[0]
    assert torch.isfinite(feats).all()
    assert torch.isfinite(target).all()


def test_dataset_filtering_by_scenario():
    _ensure_smoke_dataset()
    drape = TrajectoryDataset(DATA_DIR / "index.csv", scenarios=["drape"])
    wind = TrajectoryDataset(DATA_DIR / "index.csv", scenarios=["wind"])
    assert len(drape) > 0 and len(wind) > 0
    assert len(drape) + len(wind) <= len(TrajectoryDataset(DATA_DIR / "index.csv"))


def test_include_F_narrows_features_and_skips_F_load():
    """include_F=False drops the 9-dim F block (30->21) and does not even read
    F from disk (halves per-clip RAM)."""
    _ensure_smoke_dataset()
    ds_on = TrajectoryDataset(DATA_DIR / "index.csv", include_F=True)
    ds_off = TrajectoryDataset(DATA_DIR / "index.csv", include_F=False)
    f_on, _ = ds_on[0]
    f_off, _ = ds_off[0]
    assert f_on.shape == (ds_on.N, 9, 30)
    assert f_off.shape == (ds_off.N, 9, 21)
    # x (0:3) and v_history (3:18) blocks are identical; only F is removed.
    assert torch.allclose(f_on[..., :18], f_off[..., :18])
    # F must not be materialized into the cached clip when unused.
    ds_off._load_clip(0)
    assert "F" not in ds_off._clip_cache[0]
    assert "F" in ds_on._clip_cache[0]


def test_clip_cache_lru_eviction_bounds_memory():
    """max_cached_clips bounds how many clips stay resident; LRU-evicts the rest."""
    _ensure_smoke_dataset()
    ds = TrajectoryDataset(DATA_DIR / "index.csv", max_cached_clips=1)
    # Touch two different clips: index 0 (first clip) and the last frame
    # (a later clip, since the smoke set has multiple clips).
    clip_a, _ = ds._locate(0)
    clip_b, _ = ds._locate(len(ds) - 1)
    assert clip_a != clip_b, "smoke set should span >1 clip for this test"
    ds[0]
    assert list(ds._clip_cache.keys()) == [clip_a]
    ds[len(ds) - 1]
    # Bound respected: only the most-recent clip is retained.
    assert len(ds._clip_cache) == 1
    assert list(ds._clip_cache.keys()) == [clip_b]


def test_clip_cache_unbounded_when_none():
    """max_cached_clips=None keeps legacy unbounded behavior."""
    _ensure_smoke_dataset()
    ds = TrajectoryDataset(DATA_DIR / "index.csv", max_cached_clips=None)
    ds[0]
    ds[len(ds) - 1]
    assert len(ds._clip_cache) >= 2


def test_mmap_store_matches_npz(tmp_path):
    """Re-exporting to the per-array .npy (mmap) store yields byte-identical
    features/targets, uses true memmaps, and honors include_F."""
    _ensure_smoke_dataset()
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import npz_to_mmap

    out = tmp_path / "mmap"
    manifest = npz_to_mmap.convert(DATA_DIR / "index.csv", out,
                                   include_F=True, scenarios=None)

    npz = TrajectoryDataset(DATA_DIR / "index.csv", include_F=True)
    mm = TrajectoryDataset(manifest, include_F=True)
    assert npz._mmap_store is False and mm._mmap_store is True
    assert (mm.grid_x, mm.grid_y) == (npz.grid_x, npz.grid_y)
    assert len(mm) == len(npz)
    for i in (0, len(npz) // 2, len(npz) - 1):
        fn, an = npz[i]
        fm, am = mm[i]
        assert torch.equal(fn, fm) and torch.equal(an, am)
    # Cache entries are memmaps (flat memory), and the returned tensor is writable.
    mm._load_clip(0)
    assert isinstance(mm._clip_cache[0]["x"], np.memmap)
    f, _ = mm[0]
    f += 1.0  # must not raise (owned, writable copy)
    # include_F=False skips F.npy entirely.
    mm_off = TrajectoryDataset(manifest, include_F=False)
    feats, _ = mm_off[0]
    assert feats.shape[-1] == 21
    assert "F" not in mm_off._clip_cache.get(0, mm_off._load_clip(0))
