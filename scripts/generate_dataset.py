"""Generate the synthetic MPM cloth trajectory dataset.

Drives `src.mpm_cloth.MPMClothSim` across drape, wind, and collision
scenarios over a list of seeds, saves one `.npz` per clip plus a
`data/cloth_trajectories/index.csv` manifest.

Schema per clip (matches docs/data/README.md):
  x            : (T, N, 3)    float32  particle positions
  v            : (T, N, 3)    float32  particle velocities
  a            : (T, N, 3)    float32  per-particle accelerations (label)
  F            : (T, N, 3, 3) float32  deformation gradients
  contact_flag : (T, N)       bool     per-particle contact with sphere
  meta         : 0-d object   dict     scenario, seed, params, config_hash

Usage:
  # local CPU smoke run (6 clips on a small grid)
  python scripts/generate_dataset.py --smoke

  # full pass on Colab T4 (default counts: 4k drape, 3k wind, 3k collision)
  python scripts/generate_dataset.py --config configs/mpm.yaml --out data/cloth_trajectories

The full pass is intentionally the default; --smoke provides a faster path
that overrides the cloth/grid resolution and shrinks per-scenario clip counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


# -----------------------------------------------------------------------------
# Scenario specs
# -----------------------------------------------------------------------------

@dataclass
class ClipSpec:
    """One trajectory clip's parameter set, fully captured for reproducibility."""

    scenario: str
    seed: int
    duration_s: float
    initial_height_m: float
    sphere_center_m: tuple[float, float, float]
    sphere_radius_m: float
    pinned_corner_indices: list[int]
    wind_force_n: tuple[float, float, float]
    log_every_substeps: int

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["sphere_center_m"] = list(self.sphere_center_m)
        d["wind_force_n"] = list(self.wind_force_n)
        return d


def sample_drape_clip(rng: np.random.Generator, base: dict, seed: int) -> ClipSpec:
    h = float(rng.uniform(0.6, 0.9))
    cx = float(rng.uniform(0.85, 1.15))
    cz = float(rng.uniform(0.85, 1.15))
    cy = float(rng.uniform(0.35, 0.5))
    r = float(rng.uniform(0.15, 0.25))
    return ClipSpec(
        scenario="drape",
        seed=seed,
        duration_s=1.0,
        initial_height_m=h,
        sphere_center_m=(cx, cy, cz),
        sphere_radius_m=r,
        pinned_corner_indices=[],
        wind_force_n=(0.0, 0.0, 0.0),
        log_every_substeps=10,
    )


def sample_wind_clip(rng: np.random.Generator, base: dict, seed: int,
                     gx: int, gy: int) -> ClipSpec:
    # Pin the two top-row corners (i=0, j=0 and i=0, j=gy-1).
    pinned = [0 * gy + 0, 0 * gy + (gy - 1)]
    h = 1.0
    speed = float(rng.uniform(0.5, 4.0))
    angle = float(rng.uniform(0.0, 2 * np.pi))
    fx = speed * float(np.cos(angle))
    fz = speed * float(np.sin(angle))
    # Move sphere far away so it doesn't interfere
    return ClipSpec(
        scenario="wind",
        seed=seed,
        duration_s=1.0,
        initial_height_m=h,
        sphere_center_m=(0.0, -1.0, 0.0),
        sphere_radius_m=0.05,
        pinned_corner_indices=pinned,
        wind_force_n=(fx, 0.0, fz),
        log_every_substeps=10,
    )


def sample_collision_clip(rng: np.random.Generator, base: dict, seed: int,
                          gx: int, gy: int) -> ClipSpec:
    # Pin top corners; sphere sweeps through the cloth from below
    pinned = [0 * gy + 0, 0 * gy + (gy - 1)]
    cx = float(rng.uniform(0.85, 1.15))
    cz = float(rng.uniform(0.85, 1.15))
    cy = float(rng.uniform(0.2, 0.4))
    r = float(rng.uniform(0.10, 0.18))
    return ClipSpec(
        scenario="collision",
        seed=seed,
        duration_s=1.0,
        initial_height_m=1.0,
        sphere_center_m=(cx, cy, cz),
        sphere_radius_m=r,
        pinned_corner_indices=pinned,
        wind_force_n=(0.0, 0.0, 0.0),
        log_every_substeps=10,
    )


# -----------------------------------------------------------------------------
# Single-clip generation
# -----------------------------------------------------------------------------

def cfg_for_clip(base_cfg: dict, spec: ClipSpec, smoke: bool) -> dict:
    cfg = deepcopy(base_cfg)
    cfg["determinism"]["seed"] = int(spec.seed)
    cfg["backend"]["random_seed"] = int(spec.seed)
    cfg["cloth"]["initial_height_m"] = spec.initial_height_m
    cfg["contact"]["primitives"]["sphere"]["center_m"] = list(spec.sphere_center_m)
    cfg["contact"]["primitives"]["sphere"]["radius_m"] = spec.sphere_radius_m
    if smoke:
        cfg["backend"]["arch"] = "cpu"
        cfg["cloth"]["grid"] = [16, 16]
        cfg["mpm"]["grid_resolution"] = 32
        cfg["mpm"]["dx_m"] = cfg["mpm"]["domain_size_m"] / cfg["mpm"]["grid_resolution"]
        cfg["mpm"]["inv_dx"] = 1.0 / cfg["mpm"]["dx_m"]
    return cfg


def cfg_hash(cfg: dict) -> str:
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]


def run_one_clip(spec: ClipSpec, base_cfg: dict, smoke: bool) -> dict[str, np.ndarray]:
    """Run the simulator for one clip and return a dict of stacked arrays."""
    # Deferred import: instantiating MPMClothSim calls ti.init(), and we want
    # to do that exactly once per clip (Taichi global state is per-process).
    import taichi as ti
    ti.reset()
    from src.mpm_cloth import MPMClothSim
    cfg = cfg_for_clip(base_cfg, spec, smoke)
    gx, gy = cfg["cloth"]["grid"]
    n = gx * gy

    # Build pinned mask: True for corner particles that act as suspension points.
    pinned_mask = np.zeros(n, dtype=bool)
    for idx in spec.pinned_corner_indices:
        pinned_mask[idx] = True

    # wind_force_n is the total force on the cloth in Newtons; distribute evenly.
    total_wind = np.array(spec.wind_force_n, dtype=np.float32)
    f_ext = np.tile(total_wind / n, (n, 1))  # (N, 3) per-particle force

    sim = MPMClothSim(cfg)
    sim.reset(pinned_mask=pinned_mask)

    dt = float(cfg["mpm"]["dt_s"])
    n_substeps = int(round(spec.duration_s / dt))
    log_every = spec.log_every_substeps

    states_x, states_v, states_a, states_F, states_cf = [], [], [], [], []
    for s in range(n_substeps):
        sim.step(f_ext=f_ext)
        if s % log_every == 0:
            st = sim.state()
            states_x.append(st["x"].astype(np.float32))
            states_v.append(st["v"].astype(np.float32))
            states_a.append(st["a"].astype(np.float32))
            states_F.append(st["F"].astype(np.float32))
            states_cf.append(st["contact_flag"])

    return {
        "x": np.stack(states_x, axis=0),
        "v": np.stack(states_v, axis=0),
        "a": np.stack(states_a, axis=0),
        "F": np.stack(states_F, axis=0),
        "contact_flag": np.stack(states_cf, axis=0),
        "meta": np.array(
            {**spec.to_dict(), "config_hash": cfg_hash(cfg), "dt_s": dt,
             "n_substeps": n_substeps, "log_every_substeps": log_every,
             "grid": cfg["cloth"]["grid"], "smoke": smoke},
            dtype=object,
        ),
    }


def save_clip(out_dir: Path, clip: dict, spec: ClipSpec, idx_in_scenario: int) -> Path:
    sub = out_dir / spec.scenario
    sub.mkdir(parents=True, exist_ok=True)
    name = f"clip_{spec.seed:06d}_{idx_in_scenario:06d}.npz"
    path = sub / name
    np.savez_compressed(
        path,
        x=clip["x"], v=clip["v"], a=clip["a"], F=clip["F"],
        contact_flag=clip["contact_flag"], meta=clip["meta"],
    )
    return path


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate the MPM cloth trajectory dataset.")
    ap.add_argument("--config", default=str(ROOT / "configs" / "mpm.yaml"))
    ap.add_argument("--out", default=str(ROOT / "data" / "cloth_trajectories"))
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny CPU run (2 clips per scenario at 16x16) for schema verification.")
    ap.add_argument("--n-drape", type=int, default=4000)
    ap.add_argument("--n-wind", type=int, default=3000)
    ap.add_argument("--n-collision", type=int, default=3000)
    ap.add_argument("--seeds-train", type=int, default=8,
                    help="Seeds 0..N-1 reserved for train (train_split assigned via clip_idx).")
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, cap total clips (handy for testing).")
    args = ap.parse_args()

    base_cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        counts = {"drape": 2, "wind": 2, "collision": 2}
    else:
        counts = {"drape": args.n_drape, "wind": args.n_wind, "collision": args.n_collision}

    if args.limit > 0:
        scale = max(args.limit / sum(counts.values()), 0.0)
        for k in counts:
            counts[k] = max(int(counts[k] * scale), 1)

    # Scenario sampler: deterministic across runs by seeding from clip's seed.
    rows = []
    started = time.perf_counter()
    n_total = sum(counts.values())
    n_done = 0

    # Seed ranges by split:
    #   Train  : [0, 1_000_000)
    #   Val    : [1_000_000, 2_000_000)
    #   Eval   : [2_000_000, 3_000_000)
    # For the smoke run we use 0..5 so it's trivial to reproduce.
    seed_base = {"drape": 0, "wind": 100_000, "collision": 200_000}
    if args.smoke:
        seed_base = {"drape": 0, "wind": 100, "collision": 200}

    gx, gy = base_cfg["cloth"]["grid"]
    for scenario, n in counts.items():
        for i in range(n):
            seed = seed_base[scenario] + i
            rng = np.random.default_rng(seed)
            if scenario == "drape":
                spec = sample_drape_clip(rng, base_cfg, seed)
            elif scenario == "wind":
                spec = sample_wind_clip(rng, base_cfg, seed, gx, gy)
            else:
                spec = sample_collision_clip(rng, base_cfg, seed, gx, gy)

            try:
                t0 = time.perf_counter()
                clip = run_one_clip(spec, base_cfg, smoke=args.smoke)
                wall = time.perf_counter() - t0
            except Exception as e:
                rows.append({"scenario": scenario, "seed": seed, "status": "ERROR",
                             "error": str(e), "path": ""})
                n_done += 1
                continue

            path = save_clip(out, clip, spec, i)
            rel = path.relative_to(ROOT)
            T = clip["x"].shape[0]
            rows.append({
                "scenario": scenario,
                "seed": seed,
                "clip_idx": i,
                "path": str(rel),
                "n_frames": T,
                "n_particles": clip["x"].shape[1],
                "duration_s": float(spec.duration_s),
                "initial_height_m": float(spec.initial_height_m),
                "sphere_center_x": float(spec.sphere_center_m[0]),
                "sphere_center_y": float(spec.sphere_center_m[1]),
                "sphere_center_z": float(spec.sphere_center_m[2]),
                "sphere_radius_m": float(spec.sphere_radius_m),
                "wind_x": float(spec.wind_force_n[0]),
                "wind_z": float(spec.wind_force_n[2]),
                "n_pinned": len(spec.pinned_corner_indices),
                "config_hash": clip["meta"].item()["config_hash"],
                "wall_s": wall,
                "status": "OK",
            })
            n_done += 1
            if n_done % max(1, n_total // 20) == 0 or n_done == n_total:
                elapsed = time.perf_counter() - started
                rate = n_done / max(elapsed, 1e-3)
                eta = (n_total - n_done) / max(rate, 1e-3)
                print(f"[{n_done:5d}/{n_total}] {scenario:9s} {wall:.1f}s/clip "
                      f"rate={rate:.2f} clip/s eta={eta/60:.1f}min", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "index.csv", index=False)
    elapsed = time.perf_counter() - started
    print(f"\nDONE: {n_done} clips in {elapsed/60:.1f} min "
          f"-> {out / 'index.csv'}")


if __name__ == "__main__":
    main()
