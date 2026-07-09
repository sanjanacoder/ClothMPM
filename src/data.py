"""Trajectory loaders and k-ring neighborhood builders for the M3 neural solver.

Maps saved MPM trajectory clips into per-particle (state_t, a_t) training
pairs. Each pair consists of:
  features (N, 9, F)   - 9 nodes (self + 8 1-ring neighbors), each with F
                         scalar features. Boundary particles use the self
                         feature for missing neighbors (clamped index).
  target   (N, 3)      - per-particle acceleration at frame t (the supervised
                         label produced by the MPM grid update).
The position channel is mean-centered per particle so the MLP sees relative
positions only - this is the translation-invariance trick recommended in
scope.md section 4.2 and matches GNS / paper7 conventions.

Per-particle feature layout (F = 1 + 5*3 + 9 + 3 = 28 by default):
  - x_centered    : 3 scalars  (own = (0,0,0), neighbors = relative offset)
  - v_history (C) : 3*C scalars (most recent C velocities; default C=5)
  - F flattened   : 9 scalars  (deformation gradient)
  - f_ext         : 3 scalars  (external per-particle force; placeholder zero
                                in M3 because MPM's f_ext is implicit in the
                                grid kernel; we'll read it from a future
                                field once mpm_cloth exposes it)

Public surface:
    build_kring_index(grid_x, grid_y, k=1) -> torch.LongTensor (N, 8)
    @dataclass NormalizationStats
    class TrajectoryDataset(torch.utils.data.Dataset)
    fit_normalization_stats(dataset, n_samples=1000) -> NormalizationStats

Run via:
    python -m src.data --manifest data/cloth_trajectories/index.csv \\
                       --split train --out results/norm_stats.pt
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]


# -----------------------------------------------------------------------------
# k-ring index (regular grid topology)
# -----------------------------------------------------------------------------

def build_kring_index(grid_x: int, grid_y: int, k: int = 1) -> torch.LongTensor:
    """Return a (N, 8) LongTensor of 1-ring neighbor indices on a regular
    grid_x by grid_y quad mesh. Uses (N, K=8) for k=1 (the 8 neighbors of
    each interior particle on the 8-conn topology).

    For boundary particles whose k-th neighbor doesn't exist, that slot is
    filled with the particle's own index. The model can then treat 'missing
    neighbor' as 'self', which gives a zero relative offset and matches the
    'pad with self' convention in MeshGraphNets.
    """
    if k != 1:
        raise NotImplementedError("only k=1 (8-conn 1-ring) is implemented for the MLP path")

    n = grid_x * grid_y
    out = torch.full((n, 8), -1, dtype=torch.long)
    # Order: N, NE, E, SE, S, SW, W, NW
    offsets = [(-1, 0), (-1, 1), (0, 1), (1, 1),
               (1, 0), (1, -1), (0, -1), (-1, -1)]
    for i in range(grid_x):
        for j in range(grid_y):
            p = i * grid_y + j
            for k_idx, (di, dj) in enumerate(offsets):
                ni, nj = i + di, j + dj
                if 0 <= ni < grid_x and 0 <= nj < grid_y:
                    out[p, k_idx] = ni * grid_y + nj
                else:
                    out[p, k_idx] = p   # pad missing with self
    return out


def build_mesh_edge_index(grid_x: int, grid_y: int) -> torch.LongTensor:
    """Return (2, E) LongTensor for the 8-connected mesh-edge graph.

    Each undirected edge between particles is represented as two directed
    entries (i->j and j->i) so message passing is symmetric. Diagonal edges
    (NE, NW) are included so the GNN sees the same connectivity the implicit
    fallback's shear springs see.

    For a 16x16 grid: ~ 4*16*16 - boundary corrections ~= 900 directed edges.
    """
    edges: list[tuple[int, int]] = []
    # Horizontal + vertical (structural) -- 4-connected
    for i in range(grid_x):
        for j in range(grid_y):
            p = i * grid_y + j
            if i + 1 < grid_x:
                q = (i + 1) * grid_y + j
                edges.append((p, q)); edges.append((q, p))
            if j + 1 < grid_y:
                q = i * grid_y + (j + 1)
                edges.append((p, q)); edges.append((q, p))
            # Diagonal (shear)
            if i + 1 < grid_x and j + 1 < grid_y:
                q = (i + 1) * grid_y + (j + 1)
                edges.append((p, q)); edges.append((q, p))
            if i + 1 < grid_x and j - 1 >= 0:
                q = (i + 1) * grid_y + (j - 1)
                edges.append((p, q)); edges.append((q, p))
    arr = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return arr   # shape (2, E)


def assemble_edge_features(x: torch.Tensor,
                           edge_index: torch.Tensor) -> torch.Tensor:
    """Per-edge features: relative position (3) + magnitude (1) = 4 dims.

    Matches the MGN cloth setup of edge feature [(p_i - p_j), ||p_i - p_j||].
    """
    src, dst = edge_index[0], edge_index[1]
    diff = x[src] - x[dst]                         # (E, 3)
    norm = diff.norm(dim=-1, keepdim=True)         # (E, 1)
    return torch.cat([diff, norm], dim=-1)


def assemble_node_features_for_gnn(
    x: torch.Tensor,
    v_history: torch.Tensor,
    F: torch.Tensor,
    f_ext: torch.Tensor,
    mean_center: bool = True,
    include_F: bool = True,
) -> torch.Tensor:
    """Per-node feature vector for the GNN: (N, F_node).

    Includes x_centered (or zeros if mean_center), v_history flattened, F
    flattened, f_ext. Matches the per-particle slice of the MLP feature.

    include_F=False drops the 9-dim deformation-gradient block. For thin-shell
    cloth F is codimensional (out-of-plane component numerically undefined ->
    det F drift), so it is treated as out-of-scope for the neural features; the
    model learns from x, v (and f_ext) only. See docs/f-out-of-scope.md.
    """
    N, C, _ = v_history.shape
    if mean_center:
        x_in = torch.zeros_like(x)         # absolute position is masked
    else:
        x_in = x
    parts = [x_in, v_history.reshape(N, C * 3)]
    if include_F:
        parts.append(F.reshape(N, 9))
    parts.append(f_ext)
    return torch.cat(parts, dim=-1)


# -----------------------------------------------------------------------------
# Per-frame feature assembly
# -----------------------------------------------------------------------------

def assemble_features(
    x: torch.Tensor,         # (N, 3)
    v_history: torch.Tensor, # (N, C, 3)
    F: torch.Tensor,         # (N, 3, 3)
    f_ext: torch.Tensor,     # (N, 3)
    kring: torch.LongTensor, # (N, 8)
    include_F: bool = True,
) -> torch.Tensor:
    """Pack per-particle features into (N, 9, F) where the 9 nodes are
    self + 8 neighbors. Position is mean-centered per particle (own = 0).

    include_F=False drops the 9-dim deformation-gradient block from every node,
    taking the per-particle feature width 30 -> 21 (with C=5). For thin-shell
    cloth F is codimensional and its det drifts, so it is out-of-scope for the
    neural features; the model learns from x, v (and f_ext). See
    docs/f-out-of-scope.md."""
    N, C, _ = v_history.shape

    # Per-particle feature vector (F dims). Layout with include_F=True:
    #   0:3   x_centered (self -> 0)   3:18  v_history (C=5 -> 15)
    #   18:27 F flattened              27:30 f_ext
    # With include_F=False the F block is dropped -> width 21 (f_ext at 18:21).
    v_flat = v_history.reshape(N, C * 3)
    self_parts = [torch.zeros_like(x), v_flat]
    if include_F:
        self_parts.append(F.reshape(N, 9))
    self_parts.append(f_ext)
    self_feat = torch.cat(self_parts, dim=-1)

    # Neighbor features: gather neighbor states then re-center positions
    # relative to the centering particle.
    nbr_x = x[kring]                                  # (N, 8, 3)
    nbr_v = v_history[kring].reshape(N, 8, C * 3)
    nbr_fext = f_ext[kring]
    # Center neighbor position: pos = nbr_x - x[center]
    nbr_x_centered = nbr_x - x.unsqueeze(1)
    nbr_parts = [nbr_x_centered, nbr_v]
    if include_F:
        nbr_parts.append(F[kring].reshape(N, 8, 9))
    nbr_parts.append(nbr_fext)
    nbr_feat = torch.cat(nbr_parts, dim=-1)

    # Stack: (N, 1, F) self + (N, 8, F) neighbors -> (N, 9, F)
    return torch.cat([self_feat.unsqueeze(1), nbr_feat], dim=1)


# -----------------------------------------------------------------------------
# Normalization stats
# -----------------------------------------------------------------------------

@dataclass
class NormalizationStats:
    feat_mean: torch.Tensor          # (F,)  per-channel node feature mean
    feat_std:  torch.Tensor          # (F,)  per-channel node feature std
    target_mean: torch.Tensor        # (3,)  acceleration mean
    target_std:  torch.Tensor        # (3,)  acceleration std
    feature_dim: int
    edge_mean: torch.Tensor | None = None   # (4,)  per-channel edge feature mean (GNN only)
    edge_std:  torch.Tensor | None = None   # (4,)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "feat_mean":   self.feat_mean.tolist(),
            "feat_std":    self.feat_std.tolist(),
            "target_mean": self.target_mean.tolist(),
            "target_std":  self.target_std.tolist(),
            "feature_dim": self.feature_dim,
        }
        if self.edge_mean is not None:
            d["edge_mean"] = self.edge_mean.tolist()
            d["edge_std"] = self.edge_std.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NormalizationStats":
        em = (torch.tensor(d["edge_mean"], dtype=torch.float32)
              if "edge_mean" in d else None)
        es = (torch.tensor(d["edge_std"], dtype=torch.float32)
              if "edge_std" in d else None)
        return cls(
            feat_mean=torch.tensor(d["feat_mean"], dtype=torch.float32),
            feat_std=torch.tensor(d["feat_std"], dtype=torch.float32),
            target_mean=torch.tensor(d["target_mean"], dtype=torch.float32),
            target_std=torch.tensor(d["target_std"], dtype=torch.float32),
            feature_dim=int(d["feature_dim"]),
            edge_mean=em,
            edge_std=es,
        )


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class TrajectoryDataset(Dataset):
    """Per-particle (state_t, a_t) sampler over a list of clip .npz files.

    Each item is a tuple (features, target) where:
      features: (N, 9, F) float32 - assembled per-particle feature tensor
      target:   (N, 3)    float32 - per-particle acceleration

    Sampling: each __getitem__ returns ONE frame from ONE clip. Use a DataLoader
    with batch_size=K to stack K frames; downstream code reshapes to (K*N, ...).

    The dataset is deterministic given the manifest order; randomness comes
    from the DataLoader's shuffle.
    """

    def __init__(self,
                 manifest: Path | str,
                 split: str | None = None,
                 scenarios: Sequence[str] | None = None,
                 velocity_history_C: int = 5,
                 stats: NormalizationStats | None = None,
                 mmap: bool = True,
                 mode: str = "mlp",
                 include_F: bool = True,
                 max_cached_clips: int | None = 16):
        """mode: 'mlp' returns (features (N,9,F), target (N,3));
                 'gnn' returns dict with node_feat / edge_feat / edge_index / target.

        include_F=False drops the deformation-gradient block from the
        per-particle features (thin-shell F is codimensional / drifts). It also
        skips loading F from disk, roughly halving per-clip memory.

        max_cached_clips bounds how many decompressed clips are held in RAM at
        once (LRU eviction). .npz members cannot be true-mmapped, so each clip
        is fully decompressed on access (~300MB at 64x64 with F, ~150MB
        without); without a bound, random access across a large manifest loads
        every clip and OOMs. None = unbounded (legacy behavior; fine for the
        small 16x16 smoke clips).
        """
        if mode not in ("mlp", "gnn"):
            raise ValueError(f"mode must be 'mlp' or 'gnn', got {mode!r}")
        self.manifest_path = Path(manifest)
        df = pd.read_csv(self.manifest_path)
        df = df[df["status"] == "OK"].reset_index(drop=True)
        if scenarios is not None:
            df = df[df["scenario"].isin(scenarios)].reset_index(drop=True)
        self.df = df
        self.split = split
        self.C = int(velocity_history_C)
        self.stats = stats
        self.mmap = mmap
        self.mode = mode
        self.include_F = bool(include_F)
        self.max_cached_clips = max_cached_clips

        if len(self.df) == 0:
            raise ValueError(f"manifest {manifest} has no clips after filters")

        # Build the (frame_idx -> clip_idx, frame_in_clip) flat index
        # while filtering frames that don't have C past velocity history yet.
        starts, lengths = [], []
        for _, row in self.df.iterrows():
            T = int(row["n_frames"])
            usable = max(0, T - self.C)   # need C past frames for v_history
            starts.append(usable)
            lengths.append(T)
        self._cumlens = np.cumsum(starts)  # cumulative usable frames
        self._lengths = lengths
        self._n_total = int(self._cumlens[-1])

        # Clip storage format: ".npz" archives (compressed; decompressed on
        # access) vs per-array ".npy" directories (true-mmap; see
        # scripts/npz_to_mmap.py). Detected from the first clip's path.
        first_path = ROOT / self.df.iloc[0]["path"]
        self._mmap_store = first_path.suffix != ".npz"

        # Topology inferred from the first clip's grid metadata.
        first_meta = self._read_meta(first_path)
        gx, gy = first_meta["grid"]
        self.grid_x, self.grid_y = int(gx), int(gy)
        self.N = self.grid_x * self.grid_y
        self.kring = build_kring_index(self.grid_x, self.grid_y, k=1)
        # Mesh-edge graph for the GNN path (built once, shared by every frame)
        self.edge_index = build_mesh_edge_index(self.grid_x, self.grid_y)

        # Lazy LRU clip cache: decompressed clips kept in insertion/MRU order,
        # evicting the least-recently-used once max_cached_clips is exceeded.
        self._clip_cache: "OrderedDict[int, dict]" = OrderedDict()

    def __len__(self) -> int:
        return self._n_total

    # -- internals ------------------------------------------------------------

    @staticmethod
    def _read_meta(path: Path) -> dict:
        """Clip metadata, from meta.json (mmap store) or the npz 'meta' member."""
        if path.suffix != ".npz":
            return json.loads((path / "meta.json").read_text())
        return np.load(path, allow_pickle=True)["meta"].item()

    def _load_clip(self, clip_idx: int) -> dict:
        cache = self._clip_cache
        if clip_idx in cache:
            cache.move_to_end(clip_idx)          # mark most-recently-used
            return cache[clip_idx]
        path = ROOT / self.df.iloc[clip_idx]["path"]
        # Only reference the arrays the dataset reads: x, v (history), a
        # (target), and F only when it feeds features. contact_flag is unused.
        keys = ["x", "v", "a"] + (["F"] if self.include_F else [])
        if self._mmap_store:
            # True memory-map: each entry is a memmap handle, not resident data;
            # get_state_at slices out just the accessed frame (OS page cache).
            clip = {k: np.load(path / f"{k}.npy", mmap_mode="r") for k in keys}
        else:
            arr = np.load(path, allow_pickle=True,
                          mmap_mode="r" if self.mmap else None)
            clip = {k: arr[k] for k in keys}   # NpzFile access decompresses fully
        cache[clip_idx] = clip
        cache.move_to_end(clip_idx)
        if self.max_cached_clips is not None:
            while len(cache) > self.max_cached_clips:
                cache.popitem(last=False)         # evict least-recently-used
        return clip

    def _locate(self, idx: int) -> tuple[int, int]:
        """Map flat idx -> (clip_idx, frame_in_clip), with frame >= C."""
        clip_idx = int(np.searchsorted(self._cumlens, idx, side="right"))
        prior = int(self._cumlens[clip_idx - 1]) if clip_idx > 0 else 0
        frame_in_clip = idx - prior + self.C   # offset by C past frames
        return clip_idx, frame_in_clip

    # -- fetch ----------------------------------------------------------------

    @staticmethod
    def _frame(a: np.ndarray) -> torch.Tensor:
        """Copy one frame slice into an owned, writable float32 tensor.

        The copy matters for the mmap store (slices are read-only views into
        the memory-map) and decouples the returned tensor from the cached clip,
        so in-place ops downstream (e.g. training noise) can't corrupt it."""
        return torch.from_numpy(np.array(a, dtype=np.float32))

    def get_state_at(self, clip_idx: int, frame: int):
        """Raw state read for one frame; useful for tests."""
        clip = self._load_clip(clip_idx)
        x = self._frame(clip["x"][frame])
        v_history = self._frame(
            clip["v"][frame - self.C + 1: frame + 1]
        ).permute(1, 0, 2)   # (N, C, 3)
        # F is skipped on load when include_F=False; feed identity in that case
        # (assemble_features ignores it, but callers still expect a valid tensor).
        if "F" in clip:
            F = self._frame(clip["F"][frame])
        else:
            F = torch.eye(3).expand(x.shape[0], 3, 3).contiguous()
        a = self._frame(clip["a"][frame])
        f_ext = torch.zeros_like(x)  # placeholder until mpm_cloth exposes f_ext
        return x, v_history, F, f_ext, a

    def __getitem__(self, idx: int):
        clip_idx, frame = self._locate(idx)
        x, v_history, F, f_ext, a = self.get_state_at(clip_idx, frame)
        if self.mode == "mlp":
            feats = assemble_features(x, v_history, F, f_ext, self.kring,
                                      include_F=self.include_F)
            if self.stats is not None:
                feats = (feats - self.stats.feat_mean) / self.stats.feat_std
                a = (a - self.stats.target_mean) / self.stats.target_std
            return feats, a
        # GNN path: node + edge features
        node_feat = assemble_node_features_for_gnn(x, v_history, F, f_ext,
                                                   mean_center=True,
                                                   include_F=self.include_F)
        edge_feat = assemble_edge_features(x, self.edge_index)
        if self.stats is not None:
            # Reuse the same per-feature stats as the MLP path for the node features
            # (they share the same feature layout); edge features get their own
            # normalization in fit_normalization_stats_gnn (added below).
            node_feat = (node_feat - self.stats.feat_mean) / self.stats.feat_std
            a = (a - self.stats.target_mean) / self.stats.target_std
            if self.stats.edge_mean is not None:
                edge_feat = (edge_feat - self.stats.edge_mean) / self.stats.edge_std
        return {
            "node_feat": node_feat,            # (N, F)
            "edge_feat": edge_feat,            # (E, 4)
            "edge_index": self.edge_index,     # (2, E)  -- shared
            "target": a,                       # (N, 3)
        }


# -----------------------------------------------------------------------------
# Normalization-stats fitter
# -----------------------------------------------------------------------------

def fit_normalization_stats(
    dataset: TrajectoryDataset,
    n_samples: int = 1000,
    eps: float = 1e-6,
    fit_edges: bool = True,
) -> NormalizationStats:
    """Compute per-channel mean and std over a random subset of frames.

    Reads samples in MLP mode (always returns (feats, target)) so that the
    same stats apply whether the dataset is later used with mode='mlp' or
    'gnn' (the per-particle feature layout is identical). When fit_edges is
    True and the dataset has a mesh_edge_index, also fits edge feature stats.
    """
    import random
    rng = random.Random(0)
    indices = rng.sample(range(len(dataset)), min(n_samples, len(dataset)))

    # We need MLP-shaped features for the per-channel stats. If the user's
    # dataset is in GNN mode, temporarily flip to MLP mode for sampling.
    saved_mode = dataset.mode
    dataset.mode = "mlp"
    feats0, _ = dataset[indices[0]]
    feat_dim = feats0.shape[-1]
    feat_sum = torch.zeros(feat_dim)
    feat_sqsum = torch.zeros(feat_dim)
    target_sum = torch.zeros(3)
    target_sqsum = torch.zeros(3)
    n_feat_obs = 0
    n_target_obs = 0
    for idx in indices:
        feats, target = dataset[idx]
        flat = feats.reshape(-1, feat_dim)
        feat_sum += flat.sum(dim=0)
        feat_sqsum += (flat * flat).sum(dim=0)
        n_feat_obs += flat.shape[0]
        target_sum += target.sum(dim=0)
        target_sqsum += (target * target).sum(dim=0)
        n_target_obs += target.shape[0]
    feat_mean = feat_sum / n_feat_obs
    feat_var = feat_sqsum / n_feat_obs - feat_mean * feat_mean
    feat_std = torch.sqrt(feat_var.clamp(min=eps))
    target_mean = target_sum / n_target_obs
    target_var = target_sqsum / n_target_obs - target_mean * target_mean
    target_std = torch.sqrt(target_var.clamp(min=eps))

    edge_mean = edge_std = None
    if fit_edges:
        # Sample raw positions to compute edge feature stats (4 dims).
        e_sum = torch.zeros(4)
        e_sqsum = torch.zeros(4)
        n_e_obs = 0
        for idx in indices:
            clip_idx, frame = dataset._locate(idx)
            x, _, _, _, _ = dataset.get_state_at(clip_idx, frame)
            edge_feat = assemble_edge_features(x, dataset.edge_index)
            e_sum += edge_feat.sum(dim=0)
            e_sqsum += (edge_feat * edge_feat).sum(dim=0)
            n_e_obs += edge_feat.shape[0]
        edge_mean = (e_sum / n_e_obs).float()
        edge_var = e_sqsum / n_e_obs - edge_mean * edge_mean
        edge_std = torch.sqrt(edge_var.clamp(min=eps)).float()

    dataset.mode = saved_mode
    return NormalizationStats(
        feat_mean=feat_mean.float(), feat_std=feat_std.float(),
        target_mean=target_mean.float(), target_std=target_std.float(),
        feature_dim=feat_dim,
        edge_mean=edge_mean, edge_std=edge_std,
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default=None)
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--velocity-history-C", type=int, default=5)
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--no-F", dest="include_F", action="store_false",
                    help="drop the deformation-gradient block from features "
                         "(thin-shell F is codimensional / drifts)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ds = TrajectoryDataset(args.manifest, split=args.split,
                           scenarios=args.scenarios,
                           velocity_history_C=args.velocity_history_C,
                           include_F=args.include_F)
    print(f"dataset: {len(ds)} frames, "
          f"{ds.grid_x}x{ds.grid_y} grid, N={ds.N}")
    stats = fit_normalization_stats(ds, n_samples=args.n_samples)
    print(f"feature_dim={stats.feature_dim}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats.to_dict(), args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
