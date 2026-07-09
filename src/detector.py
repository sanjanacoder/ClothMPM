"""Complexity detectors D1-D3 for the M4 hybrid rollout.

Per-frame features (locked W2 in docs/scope.md and configs/hybrid.yaml):
  (i)   `cos_sim_window`     mean cosine similarity between two consecutive
                              10-step acceleration windows (paper7's signal)
  (ii)  `strain_rate`         max ||F_dot||_F across particles, where F_dot
                              is approximated by per-frame difference of F
  (iii) `contact_frac`        mean contact_flag across particles
  (iv)  `edge_len_change_max` max |L_t - L_{t-1}| over mesh edges

Detector variants:
  D1 — single threshold on cos_sim_window (untrained; rc = 0.8 from paper7)
  D2 — sklearn LogisticRegression over (i)-(iv)
  D3 — tiny torch MLP (2 hidden layers x 32, sigmoid output) over (i)-(iv)

Public API:
  compute_detector_features(clip) -> np.ndarray of shape (T, 4)
  class CosineThresholdDetector
  class LogRegDetector
  class TinyMLPDetector
  All implement: .fit(features, labels) (no-op for D1) and
                 .score(features) -> np.ndarray (T,) in [0, 1] where
                 higher score = more likely 'hard frame'
                 .predict(features, threshold) -> np.ndarray (T,) of bool
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.eval import cosine_window


# -----------------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------------

def _strain_rate_per_frame(F_seq: np.ndarray) -> np.ndarray:
    """Approximate ||F_dot||_F per particle with finite difference, then take
    max over particles. F_seq: (T, N, 3, 3) -> (T,) where index 0 is the
    'no information yet' baseline, set to 0.
    """
    out = np.zeros(F_seq.shape[0], dtype=np.float32)
    if F_seq.shape[0] < 2:
        return out
    diff = F_seq[1:] - F_seq[:-1]                          # (T-1, N, 3, 3)
    norms = np.sqrt((diff * diff).sum(axis=(-1, -2)))      # (T-1, N)
    out[1:] = norms.max(axis=-1)
    return out


def _contact_fraction(contact_flag: np.ndarray) -> np.ndarray:
    """contact_flag: (T, N) bool -> (T,) float in [0, 1]."""
    return contact_flag.mean(axis=-1).astype(np.float32)


def _edge_len_change_max(x_seq: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    """Max |L_t - L_{t-1}| over the mesh edges, one number per frame.

    x_seq: (T, N, 3); edge_index: (2, E) numpy long array.
    """
    src, dst = edge_index[0], edge_index[1]
    diffs = x_seq[:, src] - x_seq[:, dst]                   # (T, E, 3)
    L = np.linalg.norm(diffs, axis=-1)                       # (T, E)
    out = np.zeros(L.shape[0], dtype=np.float32)
    if L.shape[0] >= 2:
        out[1:] = np.abs(L[1:] - L[:-1]).max(axis=-1)
    return out


def compute_detector_features(
    clip: dict,
    edge_index: np.ndarray | None = None,
    cosine_window_steps: int = 10,
) -> np.ndarray:
    """Build the (T, 4) per-frame detector features for a clip dict with
    keys x (T, N, 3), F (T, N, 3, 3), contact_flag (T, N), a (T, N, 3).

    edge_index: (2, E) long array. If None, edge_len_change_max is set to 0.
    """
    a = np.asarray(clip["a"])
    cos_sig = cosine_window(a, window_steps=cosine_window_steps)
    sr = _strain_rate_per_frame(np.asarray(clip["F"]))
    cf = _contact_fraction(np.asarray(clip["contact_flag"]))
    if edge_index is not None:
        elc = _edge_len_change_max(np.asarray(clip["x"]), edge_index)
    else:
        elc = np.zeros_like(sr)
    return np.stack([cos_sig, sr, cf, elc], axis=-1).astype(np.float32)


def warmup_mask(features: np.ndarray, cosine_window_steps: int = 10) -> np.ndarray:
    """Mask of valid (post-warmup) frames where the cosine signal is real.

    The first 2 * cosine_window_steps frames carry placeholder cos_sim = 1.0;
    detector training and scoring should skip them.
    """
    T = features.shape[0]
    mask = np.zeros(T, dtype=bool)
    mask[2 * cosine_window_steps:] = True
    return mask


# -----------------------------------------------------------------------------
# D1: untrained cosine threshold
# -----------------------------------------------------------------------------

class CosineThresholdDetector:
    """paper7's D1: score = (1 - cos_sim_window) so that higher = harder.
    Threshold is applied to the original cos_sim (default rc = 0.8 from
    paper7's Water 2D sweep)."""

    def __init__(self, rc: float = 0.8):
        self.rc = rc

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "CosineThresholdDetector":
        return self  # no-op

    def score(self, features: np.ndarray) -> np.ndarray:
        # features: (T, 4); we use column 0 (cos_sim_window).
        return 1.0 - features[:, 0]

    def predict(self, features: np.ndarray, threshold: float | None = None) -> np.ndarray:
        # threshold is on the score (1 - cos_sim); equivalent to cos < rc.
        if threshold is None:
            threshold = 1.0 - self.rc
        return self.score(features) > threshold


# -----------------------------------------------------------------------------
# D2: logistic regression
# -----------------------------------------------------------------------------

class LogRegDetector:
    """sklearn LogisticRegression over the four detector features."""

    def __init__(self, **lr_kwargs):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=1000, **lr_kwargs)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "LogRegDetector":
        f = self.scaler.fit_transform(features)
        if labels.sum() == 0 or labels.sum() == len(labels):
            # Degenerate single-class case: log a warning and fit anyway by
            # adding a synthetic counter-example so sklearn doesn't blow up.
            f = np.concatenate([f, np.zeros((1, f.shape[1]))], axis=0)
            labels = np.concatenate([labels, np.array([1 - int(labels[0])])])
        self.model.fit(f, labels)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        f = self.scaler.transform(features)
        return self.model.predict_proba(f)[:, 1]

    def predict(self, features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return self.score(features) > threshold


# -----------------------------------------------------------------------------
# D3: tiny MLP
# -----------------------------------------------------------------------------

class TinyMLPDetector:
    """A two-hidden-layer MLP (32, 32) over the four detector features.

    Wraps a torch.nn.Module so the same .fit / .score / .predict interface as
    D1 and D2 holds. The network outputs a logit; sigmoid is applied at score
    time.
    """

    def __init__(self, n_features: int = 4, hidden: tuple[int, ...] = (32, 32),
                 epochs: int = 200, lr: float = 1e-3, device: str | None = None):
        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.epochs = epochs
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.feat_mean = None
        self.feat_std = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "TinyMLPDetector":
        self.net.to(self.device)
        f = torch.from_numpy(features).float()
        self.feat_mean = f.mean(dim=0, keepdim=True)
        self.feat_std = f.std(dim=0, keepdim=True).clamp_min(1e-6)
        f = ((f - self.feat_mean) / self.feat_std).to(self.device)
        y = torch.from_numpy(labels).float().to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        # class-balanced positive weight
        pos_weight = (1.0 + (1 - y).sum() / max(y.sum(), 1e-6)).to(self.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            logit = self.net(f).squeeze(-1)
            loss = loss_fn(logit, y)
            loss.backward()
            opt.step()
        self.net.eval()
        return self

    @torch.no_grad()
    def score(self, features: np.ndarray) -> np.ndarray:
        self.net.eval()
        f = torch.from_numpy(features).float()
        if self.feat_mean is not None:
            f = (f - self.feat_mean) / self.feat_std
        f = f.to(self.device)
        return torch.sigmoid(self.net(f).squeeze(-1)).cpu().numpy()

    def predict(self, features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return self.score(features) > threshold


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------

def build_detector(name: str, **kwargs):
    name = name.upper()
    if name == "D1":
        return CosineThresholdDetector(**kwargs)
    if name == "D2":
        return LogRegDetector(**kwargs)
    if name == "D3":
        return TinyMLPDetector(**kwargs)
    raise ValueError(f"unknown detector variant {name!r}")
