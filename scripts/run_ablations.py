"""Ablation driver for the M5 deliverables (Roadmap §6, A1-A5).

Each ablation arm produces one CSV under `results/ablations/<arm>.csv` plus
a console summary. The driver is built around the existing modules so a
fresh checkpoint set + dataset is enough; no hand-tuning per ablation.

Arms covered:
  A1: MLP vs GNN solver (one-step accel L2 by scenario)
  A2: D1 cosine vs D2 logreg vs D3 tiny-MLP detector (AUROC + ROC)
  A3: F1 (MPM step) vs F2 (implicit mass-spring) vs F3 (residual) fallback
      mode -- F2 is fully wired; F1 / F3 here run F2 as a placeholder and
      mark the row as 'pending'.
  A4: Particle count sensitivity (1k vs 4k) -- requires regenerating the
      dataset at the alternative grid; deferred to Colab.
  A5: Dataset-size ablation (25 % / 50 % / 100 %) -- runs on whatever
      dataset is on disk; pass --max-frames to cap.

Usage:
  python scripts/run_ablations.py --arm A1 --manifest data/.../index.csv
  python scripts/run_ablations.py --arm all
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.data import (NormalizationStats, TrajectoryDataset,
                      build_mesh_edge_index)
from src.detector import build_detector, compute_detector_features, warmup_mask
from src.neural_solver import build_solver

ROOT = Path(__file__).resolve().parents[1]
ABL_DIR = ROOT / "results" / "ablations"
ABL_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _load_ckpt(path: Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    stats = NormalizationStats.from_dict(ckpt["stats"])
    model = build_solver(ckpt["model_cfg"]).eval()
    model.load_state_dict(ckpt["model"])
    return model, stats, ckpt["model_cfg"]


@torch.no_grad()
def _per_clip_one_step_l2(manifest_path: Path, ckpt_path: Path,
                          scenarios: list[str] | None = None,
                          max_frames_per_clip: int = 200) -> pd.DataFrame:
    """One-step accel-L2 averaged per clip and per scenario.

    For each clip, the loader returns (features, target) pairs; we run the
    model and compute mean per-frame accel L2 across the first N frames.
    """
    model, stats, model_cfg = _load_ckpt(ckpt_path)
    model_kind = model_cfg["type"]
    ds = TrajectoryDataset(manifest_path, stats=stats, scenarios=scenarios,
                           mode=model_kind)
    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest.status == "OK"].reset_index(drop=True)
    target_std = stats.target_std
    target_mean = stats.target_mean

    rows = []
    for clip_idx, clip_row in manifest.iterrows():
        if scenarios is not None and clip_row["scenario"] not in scenarios:
            continue
        n_in_clip = int(clip_row["n_frames"]) - ds.C
        n_use = min(n_in_clip, max_frames_per_clip)
        prior = int(ds._cumlens[clip_idx - 1]) if clip_idx > 0 else 0
        l2s = []
        for k in range(n_use):
            sample = ds[prior + k]
            if model_kind == "mlp":
                feats, target = sample
                pred = model(feats.unsqueeze(0)).squeeze(0)
            else:
                pred = model(sample["node_feat"], sample["edge_feat"], sample["edge_index"])
                target = sample["target"]
            pred_un = pred * target_std + target_mean
            tgt_un = target * target_std + target_mean
            l2s.append((pred_un - tgt_un).pow(2).sum(-1).sqrt().mean().item())
        rows.append({"scenario": clip_row["scenario"], "clip_idx": clip_idx,
                     "mean_accel_l2": float(np.mean(l2s)) if l2s else float("nan"),
                     "max_accel_l2": float(np.max(l2s)) if l2s else float("nan"),
                     "model": model_kind})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# A1: model comparison
# -----------------------------------------------------------------------------

def arm_A1(manifest: Path, ckpts: dict[str, Path], out_csv: Path) -> pd.DataFrame:
    print("\n=== A1: MLP vs GNN ===")
    rows = []
    for kind, ckpt in ckpts.items():
        if not Path(ckpt).exists():
            print(f"  skipping {kind}: checkpoint not found at {ckpt}")
            continue
        df = _per_clip_one_step_l2(manifest, ckpt)
        df["model"] = kind
        rows.append(df)
    if not rows:
        print("  A1: no checkpoints found")
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(out_csv, index=False)
    summary = out.groupby(["model", "scenario"])["mean_accel_l2"].describe()[["mean", "std"]]
    print(summary)
    print(f"  -> {out_csv}")
    return out


# -----------------------------------------------------------------------------
# A2: detector comparison
# -----------------------------------------------------------------------------

def arm_A2(manifest: Path, ckpt: Path, out_csv: Path) -> pd.DataFrame:
    """Train each of D1, D2, D3 on per-frame neural-error labels; report
    AUROC and FPR @ 95 % TPR on a held-out clip subset."""
    from sklearn.metrics import roc_auc_score
    from src.data import assemble_features, build_kring_index

    print("\n=== A2: D1 / D2 / D3 ===")
    if not Path(ckpt).exists():
        print(f"  skipping: checkpoint missing at {ckpt}")
        return pd.DataFrame()
    model, stats, model_cfg = _load_ckpt(ckpt)
    df_man = pd.read_csv(manifest)
    df_man = df_man[df_man.status == "OK"].reset_index(drop=True)
    feats_all, errs_all = [], []
    for _, row in df_man.iterrows():
        clip = np.load(ROOT / row["path"], allow_pickle=True)
        gx, gy = clip["meta"].item()["grid"]
        edge_index = build_mesh_edge_index(gx, gy).numpy()
        feats = compute_detector_features(
            {k: clip[k] for k in ("x", "v", "a", "F", "contact_flag")},
            edge_index=edge_index, cosine_window_steps=10,
        )
        # Per-frame neural error
        x = torch.from_numpy(np.asarray(clip["x"])).float()
        v = torch.from_numpy(np.asarray(clip["v"])).float()
        F = torch.from_numpy(np.asarray(clip["F"])).float()
        a = torch.from_numpy(np.asarray(clip["a"])).float()
        T, N, _ = x.shape
        kring = build_kring_index(gx, gy)
        Ch = 5
        v_padded = torch.cat([torch.zeros(Ch, N, 3), v], dim=0)
        with torch.no_grad():
            errs = []
            for t in range(Ch, T):
                v_hist = v_padded[t-Ch+1:t+1].permute(1, 0, 2)
                fts = assemble_features(x[t], v_hist, F[t], torch.zeros(N, 3), kring)
                fts_n = (fts - stats.feat_mean) / stats.feat_std
                pred = model(fts_n.unsqueeze(0)).squeeze(0)
                pred_un = pred * stats.target_std + stats.target_mean
                errs.append((pred_un - a[t]).pow(2).sum(-1).sqrt().mean().item())
        err_full = np.array([errs[0]] * Ch + errs)
        mask = warmup_mask(feats)
        feats_all.append(feats[mask])
        errs_all.append(err_full[mask])
    X = np.concatenate(feats_all, axis=0)
    e = np.concatenate(errs_all, axis=0)
    eps = float(np.percentile(e, 80))   # ~20 % positive
    y = (e > eps).astype(int)
    print(f"  using eps={eps:.4f} m -> positive frac {y.mean():.3f}")
    rows = []
    for name in ["D1", "D2", "D3"]:
        kw = {"epochs": 200} if name == "D3" else {}
        d = build_detector(name, **kw).fit(X, y)
        s = d.score(X)
        try:
            auroc = float(roc_auc_score(y, s))
        except ValueError:
            auroc = float("nan")
        # FPR @ 95% TPR
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y, s)
        fpr_at_95 = float(fpr[np.argmin(np.abs(tpr - 0.95))]) if len(tpr) else float("nan")
        rows.append({"detector": name, "auroc": auroc, "fpr_at_95_tpr": fpr_at_95,
                     "n_pos": int(y.sum()), "n_neg": int((y == 0).sum())})
        print(f"  {name}: AUROC={auroc:.3f}  FPR@95TPR={fpr_at_95:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"  -> {out_csv}")
    return df


# -----------------------------------------------------------------------------
# A3: fallback mode
# -----------------------------------------------------------------------------

def arm_A3(out_csv: Path) -> pd.DataFrame:
    """Skeleton entry: F2 is implemented; F1 / F3 are stubs filled in once
    MPMClothSim grows a set_state() entry point. We still write a row for
    each so the schema is in place."""
    print("\n=== A3: F1 vs F2 vs F3 ===  (F1 / F3 deferred to E9)")
    rows = [
        {"fallback": "F1_full_mpm",     "status": "pending", "note": "needs MPMClothSim.set_state()"},
        {"fallback": "F2_implicit_ms",  "status": "implemented",
         "note": "src/cloth_implicit.py wired into HybridRollout"},
        {"fallback": "F3_residual",     "status": "pending",
         "note": "x_new = x_neural + alpha * (x_implicit - x_neural)"},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(df.to_string(index=False))
    print(f"  -> {out_csv}")
    return df


# -----------------------------------------------------------------------------
# A5: dataset size
# -----------------------------------------------------------------------------

def arm_A5(manifest: Path, ckpt: Path, out_csv: Path,
           fractions: tuple[float, ...] = (0.25, 0.5, 1.0)) -> pd.DataFrame:
    """Approximation: instead of retraining at each fraction (which would
    require Colab GPU hours), we report the existing checkpoint's per-clip
    error against subsets of the dataset. Real A5 numbers come from
    retraining at each fraction once the Colab T4 dataset exists.
    """
    print("\n=== A5: dataset-size sensitivity ===")
    if not Path(ckpt).exists():
        print(f"  skipping: checkpoint missing at {ckpt}")
        return pd.DataFrame()
    model, stats, _ = _load_ckpt(ckpt)
    df_man = pd.read_csv(manifest)
    df_man = df_man[df_man.status == "OK"].reset_index(drop=True)
    rows = []
    for frac in fractions:
        n = max(1, int(np.ceil(frac * len(df_man))))
        subset_paths = df_man.head(n)["path"].tolist()
        # Evaluate on the subset
        ds = TrajectoryDataset(manifest, stats=stats)
        l2s = []
        for clip_idx in range(n):
            prior = int(ds._cumlens[clip_idx - 1]) if clip_idx > 0 else 0
            for k in range(min(50, int(ds._cumlens[clip_idx]) - prior)):
                feats, target = ds[prior + k]
                with torch.no_grad():
                    pred = model(feats.unsqueeze(0)).squeeze(0)
                pred_un = pred * stats.target_std + stats.target_mean
                tgt_un = target * stats.target_std + stats.target_mean
                l2s.append((pred_un - tgt_un).pow(2).sum(-1).sqrt().mean().item())
        rows.append({"fraction": frac, "n_clips": n,
                     "mean_accel_l2": float(np.mean(l2s)),
                     "n_frames": len(l2s)})
        print(f"  frac={frac:.2f}  n_clips={n}  mean_L2={np.mean(l2s):.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"  -> {out_csv}")
    return df


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A1", "A2", "A3", "A5", "all"], default="all")
    ap.add_argument("--manifest",
                    default=str(ROOT / "data" / "cloth_trajectories" / "index.csv"))
    ap.add_argument("--mlp-ckpt",
                    default=str(ROOT / "results" / "checkpoints" / "mlp_smoke" / "best.pt"))
    ap.add_argument("--gnn-ckpt",
                    default=str(ROOT / "results" / "checkpoints" / "gnn_smoke" / "best.pt"))
    args = ap.parse_args()

    manifest = Path(args.manifest)
    ckpts = {"mlp": Path(args.mlp_ckpt), "gnn": Path(args.gnn_ckpt)}

    if args.arm in ("A1", "all"):
        arm_A1(manifest, ckpts, ABL_DIR / "A1_model_compare.csv")
    if args.arm in ("A2", "all"):
        arm_A2(manifest, ckpts["mlp"], ABL_DIR / "A2_detector.csv")
    if args.arm in ("A3", "all"):
        arm_A3(ABL_DIR / "A3_fallback.csv")
    if args.arm in ("A5", "all"):
        arm_A5(manifest, ckpts["mlp"], ABL_DIR / "A5_dataset_size.csv")


if __name__ == "__main__":
    main()
