"""Evaluation metrics and timing benchmarks for cloth simulation clips.

Metrics:
  - per-particle L2 between predicted and reference particle clouds
  - Chamfer distance (symmetric mean nearest-neighbor)
  - energy-drift proxy (kinetic-energy ratio vs reference)
  - rollout horizon at threshold tau
  - cosine similarity of consecutive accelerations (complexity detector signal)
  - wall-clock per step (timing helpers)

CLI:
  python -m src.eval pair \
      --reference path/to/clip_A.npz --predicted path/to/clip_B.npz \
      --out results/eval_<name>.csv
  python -m src.eval baseline-timing \
      --config configs/mpm.yaml --grid 16 16 --grid-resolution 32 \
      --n-steps 200 --out results/timing_mpm.csv

Every output CSV carries: git_sha, config_path, config_hash, run_name,
plus per-frame metric columns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


# -----------------------------------------------------------------------------
# Metric primitives (numpy-only so they're portable to torch / JAX later)
# -----------------------------------------------------------------------------

def per_particle_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    """L2(t) = sqrt((1/N) sum_p ||x_pred_p - x_ref_p||^2).

    pred, ref: (N, 3) particle positions at one frame.
    """
    diff = pred.reshape(-1, 3) - ref.reshape(-1, 3)
    return float(np.sqrt((diff * diff).sum(axis=-1).mean()))


def chamfer(pred: np.ndarray, ref: np.ndarray) -> float:
    """Symmetric Chamfer distance between two point clouds.

    Uses brute-force pairwise distances; fine for N ~ 4096.
    """
    pred = pred.reshape(-1, 3)
    ref = ref.reshape(-1, 3)
    # Pairwise squared distances
    d2 = ((pred[:, None, :] - ref[None, :, :]) ** 2).sum(axis=-1)
    a = d2.min(axis=1).mean()      # for each pred point, nearest ref
    b = d2.min(axis=0).mean()      # for each ref point, nearest pred
    return float(a + b)


def kinetic_energy(v: np.ndarray, mass_per: float) -> float:
    """Total kinetic energy from per-particle velocities."""
    v = v.reshape(-1, 3)
    return 0.5 * float(mass_per) * float((v * v).sum())


def energy_drift(ke_pred: float, ke_ref: float) -> float:
    """|E_pred - E_ref| / E_ref."""
    if abs(ke_ref) < 1e-12:
        return 0.0 if abs(ke_pred) < 1e-12 else float("inf")
    return float(abs(ke_pred - ke_ref) / abs(ke_ref))


def rollout_horizon(l2_per_frame: np.ndarray, tau: float, dt: float) -> float:
    """Largest contiguous time (seconds from t=0) with L2 <= tau.

    l2_per_frame is (T,); dt is the seconds between frames (not the MPM sub-step dt).
    """
    bad = np.where(l2_per_frame > tau)[0]
    if bad.size == 0:
        return float(len(l2_per_frame) * dt)
    return float(int(bad[0]) * dt)


def cosine_window(
    a_seq: np.ndarray, window_steps: int = 10
) -> np.ndarray:
    """Mean cosine similarity between two consecutive windows of accelerations.

    At frame t with window size W,
        s_t = (1/N) sum_i cos( mean_a_i_in_[t-2W, t-W],
                               mean_a_i_in_[t-W, t]  )
    a_seq shape (T, N, 3). Returns (T,) with the first 2W-1 values set to 1.0
    (pre-warmup; treated as "definitely-easy" for the detector).
    """
    T, N, _ = a_seq.shape
    out = np.ones((T,), dtype=np.float32)
    if T < 2 * window_steps:
        return out
    for t in range(2 * window_steps, T):
        a_prev = a_seq[t - 2 * window_steps:t - window_steps].mean(axis=0)
        a_curr = a_seq[t - window_steps:t].mean(axis=0)
        # cosine per particle, then mean
        num = (a_prev * a_curr).sum(axis=-1)
        den = (np.linalg.norm(a_prev, axis=-1)
               * np.linalg.norm(a_curr, axis=-1)
               + 1e-12)
        out[t] = float((num / den).mean())
    return out


# -----------------------------------------------------------------------------
# Per-clip evaluation
# -----------------------------------------------------------------------------

def eval_clip_pair(
    pred_clip_path: Path,
    ref_clip_path: Path,
    eval_cfg: dict[str, Any] | None = None,
    mass_per: float | None = None,
) -> pd.DataFrame:
    """Compute per-frame metrics between two clips with matching shapes.

    Returns a DataFrame with columns:
      frame, t, l2, chamfer, ke_pred, ke_ref, energy_drift, cos_sim_window
    """
    pred = np.load(pred_clip_path, allow_pickle=True)
    ref = np.load(ref_clip_path, allow_pickle=True)
    assert pred["x"].shape == ref["x"].shape, (
        f"pred shape {pred['x'].shape} != ref {ref['x'].shape}"
    )
    T = pred["x"].shape[0]
    # Frame dt = sub-step dt * log_every_substeps
    meta = pred["meta"].item()
    frame_dt = float(meta["dt_s"]) * int(meta.get("log_every_substeps", 1))
    if mass_per is None:
        # Mass per particle from the meta-recorded grid count and a default
        # cloth mass (we don't carry mass_kg in meta yet)
        gx, gy = meta["grid"]
        mass_per = 0.2 / (gx * gy)

    cos_sig = cosine_window(pred["a"], window_steps=10)

    rows = []
    for t in range(T):
        l2_t = per_particle_l2(pred["x"][t], ref["x"][t])
        ch_t = chamfer(pred["x"][t], ref["x"][t]) if t % max(1, T // 50) == 0 else float("nan")
        ke_p = kinetic_energy(pred["v"][t], mass_per)
        ke_r = kinetic_energy(ref["v"][t], mass_per)
        rows.append({
            "frame": t,
            "t": t * frame_dt,
            "l2": l2_t,
            "chamfer": ch_t,
            "ke_pred": ke_p,
            "ke_ref": ke_r,
            "energy_drift": energy_drift(ke_p, ke_r),
            "cos_sim_window": float(cos_sig[t]),
        })
    df = pd.DataFrame(rows)
    return df


def attach_run_metadata(df: pd.DataFrame, *, config_path: str | Path,
                        run_name: str) -> pd.DataFrame:
    """Add traceability columns (run_name, git_sha, config_path, config_hash) to the DataFrame."""
    cfg_path = Path(config_path)
    cfg_text = cfg_path.read_text() if cfg_path.exists() else ""
    cfg_hash = hashlib.sha256(cfg_text.encode()).hexdigest()[:12]
    git_sha = "unknown"
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        pass
    df = df.copy()
    df.insert(0, "run_name", run_name)
    df.insert(1, "git_sha", git_sha)
    df.insert(2, "config_path", str(cfg_path))
    df.insert(3, "config_hash", cfg_hash)
    return df


# -----------------------------------------------------------------------------
# Wall-clock baseline timing
# -----------------------------------------------------------------------------

def baseline_timing(
    config_path: str | Path,
    grid: tuple[int, int] = (16, 16),
    grid_resolution: int = 32,
    n_steps: int = 200,
    n_warmup: int = 20,
) -> pd.DataFrame:
    """Time the MPM reference for `n_steps` after `n_warmup` warmup steps.

    Returns a DataFrame with one row containing mean / std / median wall-clock
    per step. Compares apples-to-apples on whatever device Taichi picks.
    """
    from src.mpm_cloth import MPMClothSim, load_mpm_config
    cfg = load_mpm_config(config_path)
    cfg["cloth"]["grid"] = list(grid)
    cfg["mpm"]["grid_resolution"] = grid_resolution
    cfg["mpm"]["dx_m"] = cfg["mpm"]["domain_size_m"] / grid_resolution
    cfg["mpm"]["inv_dx"] = 1.0 / cfg["mpm"]["dx_m"]

    sim = MPMClothSim(cfg)
    sim.reset()
    # Warmup (JIT compile + first kernel)
    for _ in range(n_warmup):
        sim.step()
    times = np.empty(n_steps, dtype=np.float64)
    for i in range(n_steps):
        t0 = time.perf_counter()
        sim.step()
        times[i] = time.perf_counter() - t0

    return pd.DataFrame([{
        "stage": "mpm_reference",
        "grid_x": grid[0], "grid_y": grid[1],
        "grid_resolution": grid_resolution,
        "dt_s": float(cfg["mpm"]["dt_s"]),
        "n_steps": n_steps,
        "n_warmup": n_warmup,
        "mean_ms": 1000.0 * times.mean(),
        "median_ms": 1000.0 * np.median(times),
        "std_ms": 1000.0 * times.std(),
        "p95_ms": 1000.0 * np.quantile(times, 0.95),
        "p99_ms": 1000.0 * np.quantile(times, 0.99),
    }])


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    p_pair = sub.add_parser("pair", help="Compare two clips frame-by-frame.")
    p_pair.add_argument("--reference", required=True)
    p_pair.add_argument("--predicted", required=True)
    p_pair.add_argument("--config", default=str(ROOT / "configs" / "eval.yaml"))
    p_pair.add_argument("--out", required=True)
    p_pair.add_argument("--name", default="eval")

    p_time = sub.add_parser("baseline-timing", help="Wall-clock per step for the MPM reference.")
    p_time.add_argument("--config", default=str(ROOT / "configs" / "mpm.yaml"))
    p_time.add_argument("--grid", nargs=2, type=int, default=[16, 16])
    p_time.add_argument("--grid-resolution", type=int, default=32)
    p_time.add_argument("--n-steps", type=int, default=200)
    p_time.add_argument("--n-warmup", type=int, default=20)
    p_time.add_argument("--out", required=True)
    p_time.add_argument("--name", default="mpm_baseline_timing")

    args = ap.parse_args()

    if args.mode == "pair":
        df = eval_clip_pair(Path(args.predicted), Path(args.reference))
        df = attach_run_metadata(df, config_path=args.config, run_name=args.name)
    else:
        df = baseline_timing(
            args.config,
            grid=tuple(args.grid),
            grid_resolution=args.grid_resolution,
            n_steps=args.n_steps,
            n_warmup=args.n_warmup,
        )
        df = attach_run_metadata(df, config_path=args.config, run_name=args.name)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows)")
    if args.mode == "baseline-timing":
        for _, row in df.iterrows():
            print(f"  {row['stage']:20s} mean={row['mean_ms']:.2f} ms  "
                  f"median={row['median_ms']:.2f} ms  p95={row['p95_ms']:.2f} ms")


if __name__ == "__main__":
    main()
