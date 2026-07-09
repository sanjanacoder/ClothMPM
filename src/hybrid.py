"""Hybrid rollout loop (M4/W8-W9).

Default-neural / fallback-physics scheme: at every step the detector inspects
a sliding window of recent state, scores the current frame, and either:
  - runs the neural solver to predict per-particle acceleration (cheap), OR
  - runs an implicit mass-spring step (F2, the `ImplicitClothSim` fallback).

State (`x`, `v`, `F`) lives in NumPy / torch tensors so the rollout works on
any device without Taichi calls in the neural branch. F is updated via the
APIC formula F^{n+1} = (I + dt * C^{n+1}) F^n, where we approximate C as the
finite-difference velocity gradient computed from neighbor velocities.

Public API:
  class HybridRollout:
      def __init__(self, model, model_kind, stats, dt, detector,
                   threshold, fallback_sim, history_C=5,
                   cosine_window_steps=10, dtype=torch.float32, device=None)
      def reset(self, x0, v0, F0=None, kring=None, edge_index=None) -> None
      def step(self) -> dict
      def rollout(self, n_steps: int) -> dict
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data import (assemble_edge_features, assemble_features,
                      assemble_node_features_for_gnn, build_kring_index,
                      build_mesh_edge_index, NormalizationStats)
from src.detector import compute_detector_features, warmup_mask


def _approximate_C_from_velocities(v: torch.Tensor,
                                   x: torch.Tensor,
                                   kring: torch.LongTensor,
                                   eps: float = 1e-6) -> torch.Tensor:
    """Estimate the affine velocity gradient C_p (3x3) per particle from the
    1-ring neighborhood. Used to update F via F^{n+1} = (I + dt*C) F^n.

    For 8 1-ring neighbors `j_k`, fit C such that (v_{j_k} - v_p) ~= C (x_{j_k} - x_p)
    in a least-squares sense per particle.
    """
    N = v.shape[0]
    nbr_x = x[kring]                    # (N, 8, 3)
    nbr_v = v[kring]                    # (N, 8, 3)
    dx = nbr_x - x.unsqueeze(1)         # (N, 8, 3)
    dv = nbr_v - v.unsqueeze(1)         # (N, 8, 3)
    # C_p (3x3) = (sum_k dv_k dx_k^T) (sum_k dx_k dx_k^T)^{-1}
    A = torch.einsum("nki,nkj->nij", dv, dx)             # (N, 3, 3)
    B = torch.einsum("nki,nkj->nij", dx, dx)             # (N, 3, 3)
    B = B + eps * torch.eye(3, device=B.device).expand_as(B)
    C = A @ torch.linalg.inv(B)
    return C


class HybridRollout:
    """Drives a neural solver + physics fallback on a particle state.

    Bookkeeping conventions:
      - `x, v`     : torch.Tensor of shape (N, 3), current particle state.
      - `F`        : torch.Tensor of shape (N, 3, 3), deformation gradient.
      - `v_history`: deque of past velocity tensors of length C, oldest first.
      - `a_history`: deque of past acceleration tensors of length 2*W (where W
                     is cosine_window_steps), used to compute the cosine
                     signal on each step.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 model_kind: str,
                 stats: NormalizationStats,
                 dt: float,
                 detector,
                 threshold: float,
                 fallback_sim,
                 history_C: int = 5,
                 cosine_window_steps: int = 10,
                 device: str | torch.device | None = None,
                 include_F: bool = True):
        if model_kind not in ("mlp", "gnn"):
            raise ValueError(model_kind)
        self.model = model
        self.model_kind = model_kind
        self.stats = stats
        # Must match how the model was trained (see checkpoint["include_F"]);
        # a model trained without F expects the narrower feature vector.
        self.include_F = bool(include_F)
        self.dt = float(dt)
        self.detector = detector
        self.threshold = float(threshold)
        self.fallback_sim = fallback_sim
        self.C = int(history_C)
        self.W = int(cosine_window_steps)
        self.device = torch.device(device) if device is not None else next(model.parameters()).device

        self.x: torch.Tensor | None = None
        self.v: torch.Tensor | None = None
        self.F: torch.Tensor | None = None
        self.f_ext: torch.Tensor | None = None
        self.kring: torch.LongTensor | None = None
        self.edge_index: torch.LongTensor | None = None
        self._v_history: deque[torch.Tensor] = deque(maxlen=self.C)
        self._x_history: deque[torch.Tensor] = deque(maxlen=2 * self.W + 2)
        self._a_history: deque[torch.Tensor] = deque(maxlen=2 * self.W + 2)
        self._F_history: deque[torch.Tensor] = deque(maxlen=2 * self.W + 2)
        self._cf_history: deque[torch.Tensor] = deque(maxlen=2 * self.W + 2)
        self._step_idx = 0

    # -- setup ---------------------------------------------------------------

    def reset(self,
              x0: np.ndarray | torch.Tensor,
              v0: np.ndarray | torch.Tensor | None = None,
              F0: np.ndarray | torch.Tensor | None = None,
              grid_x: int | None = None,
              grid_y: int | None = None) -> None:
        x = torch.as_tensor(x0, dtype=torch.float32, device=self.device)
        N = x.shape[0]
        if v0 is None:
            v = torch.zeros_like(x)
        else:
            v = torch.as_tensor(v0, dtype=torch.float32, device=self.device)
        if F0 is None:
            F = torch.eye(3, device=self.device).expand(N, 3, 3).contiguous()
        else:
            F = torch.as_tensor(F0, dtype=torch.float32, device=self.device)
        f_ext = torch.zeros_like(x)
        # Build mesh / k-ring indices if not yet known
        if grid_x is None or grid_y is None:
            # Try to recover from sqrt(N); only valid for square grids.
            side = int(round(np.sqrt(N)))
            if side * side != N:
                raise ValueError("grid_x / grid_y must be passed for non-square N")
            grid_x = grid_y = side
        self.kring = build_kring_index(grid_x, grid_y).to(self.device)
        self.edge_index = build_mesh_edge_index(grid_x, grid_y).to(self.device)
        self.grid_x, self.grid_y = grid_x, grid_y
        self.x, self.v, self.F, self.f_ext = x, v, F, f_ext
        # Pre-fill velocity history with v0 so the first feature build is valid
        self._v_history.clear()
        for _ in range(self.C):
            self._v_history.append(v.clone())
        self._x_history.clear(); self._a_history.clear(); self._F_history.clear(); self._cf_history.clear()
        self._x_history.append(x.clone())
        self._a_history.append(torch.zeros_like(x))
        self._F_history.append(F.clone())
        self._cf_history.append(torch.zeros(N, dtype=torch.bool, device=self.device))
        self._step_idx = 0

    # -- features and detector ------------------------------------------------

    def _detector_features_now(self) -> np.ndarray:
        """Compute the (1, 4) detector feature row for the current step from
        the rolling histories."""
        if len(self._a_history) < 2:
            return np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        # Stack histories along time
        a = torch.stack(list(self._a_history)).cpu().numpy()    # (T, N, 3)
        F = torch.stack(list(self._F_history)).cpu().numpy()    # (T, N, 3, 3)
        x = torch.stack(list(self._x_history)).cpu().numpy()    # (T, N, 3)
        cf = torch.stack(list(self._cf_history)).cpu().numpy()  # (T, N)
        clip = {"a": a, "F": F, "x": x, "contact_flag": cf}
        feats = compute_detector_features(
            clip,
            edge_index=self.edge_index.cpu().numpy(),
            cosine_window_steps=self.W,
        )
        # Return only the latest row
        return feats[-1:].astype(np.float32)

    # -- per-step forward paths -----------------------------------------------

    @torch.no_grad()
    def _neural_accel(self) -> torch.Tensor:
        """Run the neural model on the current state and return per-particle
        acceleration (m/s^2, unnormalized)."""
        v_hist = torch.stack(list(self._v_history), dim=1)      # (N, C, 3)
        if self.model_kind == "mlp":
            feats = assemble_features(self.x, v_hist, self.F, self.f_ext, self.kring,
                                      include_F=self.include_F)
            feat_mean = self.stats.feat_mean.to(self.device)
            feat_std = self.stats.feat_std.to(self.device)
            feats_n = (feats - feat_mean) / feat_std
            pred_n = self.model(feats_n.unsqueeze(0)).squeeze(0)
        else:
            node_feat = assemble_node_features_for_gnn(self.x, v_hist, self.F, self.f_ext,
                                                       mean_center=True,
                                                       include_F=self.include_F)
            edge_feat = assemble_edge_features(self.x, self.edge_index)
            feat_mean = self.stats.feat_mean.to(self.device)
            feat_std = self.stats.feat_std.to(self.device)
            node_n = (node_feat - feat_mean) / feat_std
            if self.stats.edge_mean is not None:
                edge_n = (edge_feat - self.stats.edge_mean.to(self.device)) / self.stats.edge_std.to(self.device)
            else:
                edge_n = edge_feat
            pred_n = self.model(node_n, edge_n, self.edge_index)
        target_std = self.stats.target_std.to(self.device)
        target_mean = self.stats.target_mean.to(self.device)
        return pred_n * target_std + target_mean

    def _step_neural(self) -> torch.Tensor:
        a = self._neural_accel()
        # Semi-implicit Euler
        new_v = self.v + self.dt * a
        new_x = self.x + self.dt * new_v
        # F update via approximate C
        C = _approximate_C_from_velocities(new_v, self.x, self.kring)
        eye = torch.eye(3, device=self.device).expand_as(self.F)
        new_F = (eye + self.dt * C) @ self.F
        self.v = new_v
        self.x = new_x
        self.F = new_F
        return a

    def _step_fallback(self) -> torch.Tensor:
        """Run one F2 implicit-mass-spring step, then read state back into our tensors."""
        sim = self.fallback_sim
        # Hand current state to the implicit sim
        sim.x = self.x.detach().cpu().numpy().astype(np.float64)
        sim.v = self.v.detach().cpu().numpy().astype(np.float64)
        # Use gravity only as f_ext for now (fallback's own gravity is also added
        # if f_ext=None). Pass f_ext=None so the implicit sim adds m*g internally.
        info = sim.step(f_ext=None, dt=self.dt, cg_max_iters=50, cg_tol=1e-4)
        # Pull state back
        new_x = torch.as_tensor(sim.x, dtype=torch.float32, device=self.device)
        new_v = torch.as_tensor(sim.v, dtype=torch.float32, device=self.device)
        a = (new_v - self.v) / self.dt
        self.x, self.v = new_x, new_v
        # F update with approximate C from the new velocity field
        C = _approximate_C_from_velocities(new_v, self.x, self.kring)
        eye = torch.eye(3, device=self.device).expand_as(self.F)
        self.F = (eye + self.dt * C) @ self.F
        return a

    # -- public step ----------------------------------------------------------

    def step(self) -> dict[str, Any]:
        # Detector decision based on the state BEFORE this step
        feat_row = self._detector_features_now()              # (1, 4)
        score = float(self.detector.score(feat_row)[0])
        in_warmup = self._step_idx < 2 * self.W
        used_fallback = (not in_warmup) and (score > self.threshold)

        if used_fallback:
            a = self._step_fallback()
        else:
            a = self._step_neural()

        # Update histories with the post-step data
        self._x_history.append(self.x.clone())
        self._v_history.append(self.v.clone())
        self._a_history.append(a.detach().clone())
        self._F_history.append(self.F.clone())
        # Cheap contact flag: mark particles inside the fallback's sphere if
        # we have the config; otherwise leave at zero. Real contact_flag from
        # the MPM reference isn't observable here.
        self._cf_history.append(torch.zeros(self.x.shape[0], dtype=torch.bool, device=self.device))
        self._step_idx += 1

        return {
            "step": self._step_idx,
            "used_fallback": bool(used_fallback),
            "detector_score": score,
            "max_v": float(self.v.norm(dim=-1).max().item()),
        }

    def rollout(self, n_steps: int, log_every: int = 1) -> dict[str, np.ndarray]:
        rows = []
        states = []
        for _ in range(n_steps):
            info = self.step()
            rows.append(info)
            if (info["step"] - 1) % log_every == 0:
                states.append({
                    "x": self.x.detach().cpu().numpy().copy(),
                    "v": self.v.detach().cpu().numpy().copy(),
                })
        return {
            "rows": rows,
            "x": np.stack([s["x"] for s in states]),
            "v": np.stack([s["v"] for s in states]),
            "fallback_fraction": float(np.mean([r["used_fallback"] for r in rows])),
        }
