"""Tests for the M4 complexity detectors and feature extraction."""

from pathlib import Path

import numpy as np
import pytest

from src.detector import (CosineThresholdDetector, LogRegDetector,
                          TinyMLPDetector, build_detector,
                          compute_detector_features, warmup_mask)

ROOT = Path(__file__).resolve().parents[1]


def _synthetic_clip(T: int = 60, N: int = 16) -> dict:
    rng = np.random.default_rng(0)
    a = rng.normal(size=(T, N, 3)).astype(np.float32)
    F = (np.eye(3, dtype=np.float32)[None, None]
         + 0.01 * rng.normal(size=(T, N, 3, 3))).astype(np.float32)
    x = rng.normal(size=(T, N, 3)).astype(np.float32)
    cf = (rng.uniform(size=(T, N)) > 0.9)
    return {"a": a, "F": F, "x": x, "contact_flag": cf}


def test_compute_detector_features_shape():
    clip = _synthetic_clip()
    feats = compute_detector_features(clip)
    T = clip["a"].shape[0]
    assert feats.shape == (T, 4)
    # The first 2*W frames are warmup -> cos_sim is 1.0
    assert (feats[:20, 0] == 1.0).all()


def test_warmup_mask():
    feats = compute_detector_features(_synthetic_clip())
    mask = warmup_mask(feats)
    assert mask.sum() == feats.shape[0] - 20
    assert not mask[0] and not mask[19]
    assert mask[20] and mask[-1]


# -----------------------------------------------------------------------------
# D1: cosine threshold (untrained)
# -----------------------------------------------------------------------------

def test_d1_score_is_one_minus_cos():
    feats = np.array([[1.0, 0, 0, 0],
                      [0.5, 0, 0, 0],
                      [0.0, 0, 0, 0]], dtype=np.float32)
    d = CosineThresholdDetector(rc=0.8)
    np.testing.assert_allclose(d.score(feats), [0.0, 0.5, 1.0])


def test_d1_predict_at_default_threshold():
    feats = np.array([[1.0, 0, 0, 0],     # cos=1 -> easy
                      [0.5, 0, 0, 0],     # cos=0.5 -> hard
                      [0.85, 0, 0, 0]], dtype=np.float32)
    d = CosineThresholdDetector(rc=0.8)
    out = d.predict(feats)
    assert out.tolist() == [False, True, False]


# -----------------------------------------------------------------------------
# D2: logistic regression
# -----------------------------------------------------------------------------

def test_d2_can_separate_obvious_signal():
    """If column 1 (strain rate) is the only varying feature and labels are
    determined by it, D2 must pick that up."""
    rng = np.random.default_rng(0)
    n = 200
    cos = rng.uniform(0.7, 1.0, size=n).astype(np.float32)
    sr = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
    cf = np.zeros(n, dtype=np.float32)
    elc = np.zeros(n, dtype=np.float32)
    feats = np.stack([cos, sr, cf, elc], axis=-1)
    labels = (sr > 0.5).astype(int)
    d = LogRegDetector().fit(feats, labels)
    # AUROC-like separation
    s = d.score(feats)
    sep = float(s[labels == 1].mean() - s[labels == 0].mean())
    assert sep > 0.3, f"D2 failed to separate; sep={sep}"


# -----------------------------------------------------------------------------
# D3: tiny MLP
# -----------------------------------------------------------------------------

def test_d3_overfit_simple_signal():
    rng = np.random.default_rng(0)
    n = 200
    sr = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
    feats = np.stack([np.ones(n, dtype=np.float32), sr,
                      np.zeros(n, dtype=np.float32),
                      np.zeros(n, dtype=np.float32)], axis=-1)
    labels = (sr > 0.5).astype(int)
    d = TinyMLPDetector(epochs=300).fit(feats, labels)
    s = d.score(feats)
    sep = float(s[labels == 1].mean() - s[labels == 0].mean())
    assert sep > 0.4, f"D3 failed to overfit; sep={sep}"


def test_d3_handles_degenerate_single_class():
    """If labels are all 0, fit should not crash and score should not be NaN."""
    feats = np.random.default_rng(0).normal(size=(20, 4)).astype(np.float32)
    labels = np.zeros(20, dtype=int)
    # D2's degenerate case is mitigated; D3's pos_weight handles 0 positives by
    # falling back to a 1-weight (loss is well-defined).
    d2 = LogRegDetector().fit(feats, labels)
    d3 = TinyMLPDetector(epochs=20).fit(feats, labels)
    assert np.isfinite(d2.score(feats)).all()
    assert np.isfinite(d3.score(feats)).all()


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------

def test_build_detector_factory():
    assert isinstance(build_detector("D1"), CosineThresholdDetector)
    assert isinstance(build_detector("D2"), LogRegDetector)
    assert isinstance(build_detector("D3", epochs=10), TinyMLPDetector)
    with pytest.raises(ValueError):
        build_detector("D4")
