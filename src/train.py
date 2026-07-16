"""Training loop for the neural grid-update models (M3).

Trains an MLPSolver (or GNNSolver) on per-particle acceleration from saved MPM
trajectory clips. Supports one-step teacher forcing (default) and, with
--window>1, pushforward/short-unroll training: the model is unrolled K steps on
its own predictions (K ramped by the rollout curriculum) so it learns to correct
model-generated states -- the fix for rollout error compounding (see
docs/rollout-scaling-finding.md). --noise-sigma adds GNS/MGN input noise.

Run via:
  python -m src.train --config configs/mlp.yaml --manifest path/to/index.csv

Outputs:
  results/checkpoints/<run_name>/best.pt
  results/checkpoints/<run_name>/last.pt
  results/<run_name>_train_log.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.data import (NormalizationStats, TrajectoryDataset,
                      assemble_edge_features, assemble_features,
                      assemble_node_features_for_gnn, build_kring_index,
                      fit_normalization_stats)
from src.neural_solver import build_solver

ROOT = Path(__file__).resolve().parents[1]


def pick_device(force_cpu: bool = False) -> torch.device:
    """CUDA > CPU. Apple MPS is intentionally avoided for training because
    PyTorch MPS produces NaN losses on this MLP (likely an interaction with
    LayerNorm + small-batch grad clipping); local Mac runs fall back to CPU.
    Inference can still use MPS via an explicit device argument.
    """
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_val_split(ds: TrajectoryDataset, val_frac: float = 0.1, seed: int = 0):
    n = len(ds)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * val_frac)))
    val_idx = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()
    return Subset(ds, train_idx), Subset(ds, val_idx)


def _forward(model: nn.Module, batch, device: torch.device, model_kind: str):
    """Run the model on one batch from either the MLP or GNN data path.

    Returns (pred, target) both shaped consistently for MSE: MLP -> (B*N, 3),
    GNN -> (B*N, 3) by stacking the per-graph node tensors.
    """
    if model_kind == "mlp":
        feats, target = batch
        feats = feats.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        pred = model(feats)
        return pred, target
    # GNN: batch is a dict with stacked tensors. We treat each graph in the
    # batch independently (concatenate node and edge tensors with offsets on
    # edge_index so they form one big disconnected graph).
    node_feat = batch["node_feat"].to(device)
    edge_feat = batch["edge_feat"].to(device)
    edge_index = batch["edge_index"].to(device)
    target = batch["target"].to(device)
    B, N, F = node_feat.shape
    _, E, _ = edge_feat.shape
    # Offset edge_index for each graph in the batch
    offsets = torch.arange(B, device=device) * N
    ei = edge_index + offsets[:, None, None]   # (B, 2, E)
    pred = model(node_feat.reshape(B * N, F),
                 edge_feat.reshape(B * E, edge_feat.shape[-1]),
                 ei.permute(1, 0, 2).reshape(2, B * E))
    return pred, target.reshape(B * N, 3)


def _unroll(model, batch, device, model_kind, stats, kring, dt, K, include_F):
    """Pushforward / short-unroll: from the initial state, predict acceleration,
    integrate (semi-implicit Euler), and feed the model its OWN resulting state
    for K steps -- so it learns to correct model-generated (drifted) states, the
    fix for rollout error compounding. Loss = normalized-accel MSE at each step
    vs the true acceleration; gradients flow through the integration chain.

    K=1 reduces to one-step from the true initial state (used for validation).
    Loops over the batch and reuses the same assembly/model calls as the
    one-step path (correctness over speed; ~K x the one-step cost).
    """
    x0 = batch["x0"].to(device)             # (B, N, 3)
    v0 = batch["v_hist0"].to(device)        # (B, N, C, 3)
    a_win = batch["a_win"].to(device)       # (B, window, N, 3)  un-normalized
    fm, fs = stats.feat_mean.to(device), stats.feat_std.to(device)
    tm, ts = stats.target_mean.to(device), stats.target_std.to(device)
    em = stats.edge_mean.to(device) if stats.edge_mean is not None else None
    es = stats.edge_std.to(device) if stats.edge_std is not None else None
    kring = kring.to(device)
    B, N, _ = x0.shape
    f_ext = torch.zeros(N, 3, device=device)
    F_id = torch.eye(3, device=device).expand(N, 3, 3).contiguous()
    ei = batch["edge_index"][0].to(device) if model_kind == "gnn" else None

    loss = x0.new_zeros(())
    accel_l2 = []
    n = 0
    for b in range(B):
        x, vh = x0[b], v0[b]                 # (N,3), (N,C,3)
        for k in range(K):
            if model_kind == "mlp":
                feats = assemble_features(x, vh, F_id, f_ext, kring,
                                          include_F=include_F)
                pred_n = model(((feats - fm) / fs).unsqueeze(0)).squeeze(0)
            else:
                node = assemble_node_features_for_gnn(
                    x, vh, F_id, f_ext, mean_center=True, include_F=include_F)
                edge = assemble_edge_features(x, ei)
                node_n = (node - fm) / fs
                edge_n = (edge - em) / es if em is not None else edge
                pred_n = model(node_n, edge_n, ei)
            tgt_n = (a_win[b, k] - tm) / ts
            loss = loss + nn.functional.mse_loss(pred_n, tgt_n)
            accel_l2.append(((pred_n * ts) - (tgt_n * ts)).pow(2).sum(-1).sqrt().mean().item())
            n += 1
            # semi-implicit Euler on the model's own (un-normalized) accel
            a_pred = pred_n * ts + tm
            v_new = vh[:, -1] + dt * a_pred
            x = x + dt * v_new
            vh = torch.cat([vh[:, 1:], v_new.unsqueeze(1)], dim=1)
    return loss / n, float(np.mean(accel_l2))


def _evaluate_unroll(model, loader, device, stats, model_kind, unroll_kw):
    """Validation for pushforward mode: one-step (K=1) accel error on the clean
    windowed val set (no drift, comparable across arms)."""
    model.eval()
    losses, accel = [], []
    with torch.no_grad():
        for batch in loader:
            loss, a_l2 = _unroll(model, batch, device, model_kind, stats,
                                 K=1, **unroll_kw)
            losses.append(loss.item())
            accel.append(a_l2)
    return {"val_mse_norm": float(np.mean(losses)),
            "val_accel_l2_m_per_s2": float(np.mean(accel))}


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             stats: NormalizationStats, model_kind: str = "mlp") -> dict[str, float]:
    """Return mean per-particle L2 between predicted and true acceleration in
    the unnormalized (m/s^2) units."""
    model.eval()
    losses = []
    accel_l2 = []
    target_std = stats.target_std.to(device)
    target_mean = stats.target_mean.to(device)
    with torch.no_grad():
        for batch in loader:
            pred, target = _forward(model, batch, device, model_kind)
            losses.append(nn.functional.mse_loss(pred, target).item())
            pred_un = pred * target_std + target_mean
            tgt_un = target * target_std + target_mean
            accel_l2.append(
                (pred_un - tgt_un).pow(2).sum(-1).sqrt().mean().item()
            )
    return {
        "val_mse_norm": float(np.mean(losses)),
        "val_accel_l2_m_per_s2": float(np.mean(accel_l2)),
    }


def rollout_horizon_for_epoch(epoch: int, total_epochs: int,
                              schedule: tuple[int, ...] = (1, 2, 4, 8),
                              boundaries: tuple[float, ...] = (0.5, 0.7, 0.85)
                              ) -> int:
    """Epoch-fraction rollout curriculum. Default schedule (matches mlp.yaml):
        epochs 0-50%   -> horizon 1
        epochs 50-70%  -> horizon 2
        epochs 70-85%  -> horizon 4
        epochs 85-100% -> horizon 8
    The boundaries argument lists fractional thresholds for moving up.
    """
    f = epoch / max(1, total_epochs - 1)
    for i, b in enumerate(boundaries):
        if f < b:
            return schedule[i]
    return schedule[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "mlp.yaml"))
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-train-frames", type=int, default=0,
                    help="If >0, cap the training set size for fast smoke runs.")
    ap.add_argument("--name", default="mlp_smoke")
    ap.add_argument("--out-root", default=str(ROOT / "results"))
    ap.add_argument("--cpu", action="store_true",
                    help="Force CPU even if CUDA is available.")
    ap.add_argument("--num-workers", type=int, default=None,
                    help="DataLoader worker processes for parallel feature "
                         "assembly (overlaps CPU loading with GPU compute). "
                         "Default: train.num_workers in the config, else 0.")
    ap.add_argument("--no-F", dest="no_F", action="store_true",
                    help="Drop the deformation-gradient block from features "
                         "(thin-shell F is codimensional / drifts). Overrides "
                         "features.include_F in the config.")
    ap.add_argument("--noise-sigma", type=float, default=0.0,
                    help="GNS/MGN random-walk velocity noise (train only) to "
                         "teach rollout drift-recovery. 0 = off. Sweep this.")
    ap.add_argument("--window", type=int, default=1,
                    help="Pushforward/unroll horizon (max steps unrolled on the "
                         "model's own predictions). 1 = one-step. The rollout "
                         "curriculum ramps K up to this each run.")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    epochs = args.epochs or int(cfg["train"]["epochs"])
    batch_size = args.batch_size or int(cfg["train"]["batch_size"])
    # MLP config has features.velocity_history_C, GNN config has
    # model.velocity_history_C. Look in both.
    if "features" in cfg and "velocity_history_C" in cfg["features"]:
        C = int(cfg["features"]["velocity_history_C"])
    else:
        C = int(cfg["model"].get("velocity_history_C", 5))
    model_kind = cfg["model"].get("type", "mlp").lower()
    # F is out-of-scope for thin-shell cloth (codimensional / det drift). Config
    # default keeps it (M3/M5 reproducibility); --no-F forces it off. MLP config
    # holds it under features.*, GNN under model.* (mirrors velocity_history_C).
    include_F = bool(
        cfg.get("features", {}).get(
            "include_F", cfg["model"].get("include_F", True))
    )
    if args.no_F:
        include_F = False
    print(f"include_F={include_F}")
    out_root = Path(args.out_root)
    ckpt_dir = out_root / "checkpoints" / args.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_root / f"{args.name}_train_log.csv"

    device = pick_device(force_cpu=args.cpu)
    torch.manual_seed(int(cfg["run"]["seed"]))

    # Build dataset (no normalization yet) and fit stats
    ds_raw = TrajectoryDataset(
        args.manifest,
        scenarios=args.scenarios,
        velocity_history_C=C,
        mode=model_kind,
        include_F=include_F,
    )
    print(f"raw dataset: {len(ds_raw)} frames, "
          f"grid={ds_raw.grid_x}x{ds_raw.grid_y}, N={ds_raw.N}, mode={model_kind}")
    stats = fit_normalization_stats(
        ds_raw, n_samples=min(500, len(ds_raw)),
        fit_edges=(model_kind == "gnn"),
    )
    print(f"feature_dim={stats.feature_dim}, target_std={stats.target_std.tolist()}")

    # Two normalized views of the same data: the train set gets GNS/MGN velocity
    # noise (drift-recovery), val stays clean. Same underlying frames + same
    # split seed, so val is genuinely held-out from train.
    print(f"noise_sigma={args.noise_sigma}")

    # window>1 -> pushforward: train + val both use the windowed dataset (same
    # frames/split), the trainer unrolls K steps on the model's own predictions.
    print(f"noise_sigma={args.noise_sigma}  window={args.window}")

    def _make_ds(noise: float):
        base = TrajectoryDataset(
            args.manifest, scenarios=args.scenarios, velocity_history_C=C,
            stats=stats, mode=model_kind, include_F=include_F, noise_sigma=noise,
            window=args.window)
        d = base
        if args.max_train_frames > 0 and len(base) > args.max_train_frames:
            idx = np.linspace(0, len(base) - 1, args.max_train_frames).astype(int)
            d = Subset(base, idx.tolist())
        return d, base

    ds_noisy, base_noisy = _make_ds(args.noise_sigma)
    ds_clean, _ = _make_ds(0.0) if args.noise_sigma > 0 else (ds_noisy, base_noisy)
    seed = int(cfg["run"]["seed"])
    train_set, _ = train_val_split(ds_noisy, val_frac=0.1, seed=seed)
    _, val_set = train_val_split(ds_clean, val_frac=0.1, seed=seed)
    # Unroll needs the mesh k-ring, trajectory dt, and the acceleration stats.
    unroll_kw = dict(kring=base_noisy.kring, dt=base_noisy.dt,
                     include_F=include_F) if args.window > 1 else None
    # Parallel feature assembly: worker processes build upcoming batches while
    # the GPU trains the current one (the per-frame k-ring gather in
    # assemble_features is CPU-bound). Identical results; just overlaps I/O with
    # compute. pin_memory speeds host->GPU copies; persistent_workers avoids
    # re-forking (and re-opening mmap handles) every epoch.
    num_workers = (args.num_workers if args.num_workers is not None
                   else int(cfg["train"].get("num_workers", 0)))
    loader_kw = dict(num_workers=num_workers,
                     pin_memory=(device.type == "cuda"),
                     persistent_workers=(num_workers > 0))
    print(f"dataloader: num_workers={num_workers}, pin_memory={loader_kw['pin_memory']}")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              drop_last=True, **loader_kw)
    val_loader = DataLoader(val_set, batch_size=max(1, batch_size // 2),
                            shuffle=False, **loader_kw)

    # Build model -- patch the YAML model config with the actual feature dims
    model_cfg = deepcopy(cfg["model"])
    if model_kind == "mlp":
        model_cfg["feature_dim"] = int(stats.feature_dim)
        model_cfg["n_nodes"] = 9
    else:
        model_cfg["node_feature_dim"] = int(stats.feature_dim)
        model_cfg["edge_feature_dim"] = 4
    model = build_solver(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {model_cfg['type']} with {n_params} params, device={device}")

    optim = torch.optim.AdamW(model.parameters(),
                              lr=float(cfg["train"]["lr"]),
                              weight_decay=float(cfg["train"]["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    best_val = float("inf")
    log_f = open(log_path, "w", newline="")
    log_csv = csv.DictWriter(log_f, fieldnames=[
        "epoch", "horizon", "train_mse_norm", "val_mse_norm",
        "val_accel_l2_m_per_s2", "lr", "elapsed_s",
    ])
    log_csv.writeheader()

    rollout_schedule = tuple(cfg["train"].get("rollout_curriculum", (1, 2, 4, 8)))

    t_start = time.perf_counter()
    for epoch in range(epochs):
        horizon = rollout_horizon_for_epoch(epoch, epochs, schedule=rollout_schedule)
        model.train()
        train_losses = []
        # In pushforward mode, the rollout curriculum ramps the unroll horizon K
        # (capped by --window) -- start small, grow toward the window each run.
        K = min(horizon, args.window) if args.window > 1 else 1
        for batch in train_loader:
            if unroll_kw is not None:
                loss, _ = _unroll(model, batch, device, model_kind, stats,
                                  K=K, **unroll_kw)
            else:
                pred, target = _forward(model, batch, device, model_kind)
                loss = nn.functional.mse_loss(pred, target)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            train_losses.append(loss.item())
        sched.step()
        train_mse = float(np.mean(train_losses)) if train_losses else float("nan")
        if unroll_kw is not None:
            val_metrics = _evaluate_unroll(model, val_loader, device, stats,
                                           model_kind, unroll_kw)
        else:
            val_metrics = evaluate(model, val_loader, device, stats, model_kind=model_kind)
        elapsed = time.perf_counter() - t_start
        row = {
            "epoch": epoch,
            "horizon": K,   # applied unroll steps (was a no-op logged value before)
            "train_mse_norm": train_mse,
            "val_mse_norm": val_metrics["val_mse_norm"],
            "val_accel_l2_m_per_s2": val_metrics["val_accel_l2_m_per_s2"],
            "lr": optim.param_groups[0]["lr"],
            "elapsed_s": elapsed,
        }
        log_csv.writerow(row)
        log_f.flush()
        print(f"[ep {epoch:3d} h={K}] train_mse={train_mse:.4f}  "
              f"val_mse={val_metrics['val_mse_norm']:.4f}  "
              f"val_accel_l2={val_metrics['val_accel_l2_m_per_s2']:.3f} m/s^2  "
              f"lr={optim.param_groups[0]['lr']:.2e}  ({elapsed:.1f}s)",
              flush=True)
        if val_metrics["val_mse_norm"] < best_val:
            best_val = val_metrics["val_mse_norm"]
            torch.save({
                "model": model.state_dict(),
                "stats": stats.to_dict(),
                "model_cfg": model_cfg,
                "include_F": include_F,
                "epoch": epoch,
            }, ckpt_dir / "best.pt")

    torch.save({
        "model": model.state_dict(),
        "stats": stats.to_dict(),
        "model_cfg": model_cfg,
        "include_F": include_F,
        "epoch": epochs - 1,
    }, ckpt_dir / "last.pt")
    log_f.close()
    print(f"\nBest val MSE: {best_val:.4f}")
    print(f"checkpoints -> {ckpt_dir}")
    print(f"log -> {log_path}")


if __name__ == "__main__":
    main()
