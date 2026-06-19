"""Modal-powered parallel dataset generator for MPM cloth trajectories.

Fans out run_one_clip() across T4 GPU containers simultaneously, completing
the full 10k-clip dataset in ~1-2 h instead of ~27 h sequentially on a
single GPU.

Each container runs exactly one clip: ti.reset() + ti.init() + simulate.
Clips are independent, so there is no coordination overhead between containers.

Usage (run from the ClothMPM project root):
  # 6-clip smoke test — validates schema and physics, costs < $0.10
  modal run scripts/generate_dataset_modal.py --smoke

  # Full 10k-clip dataset (~1-2 h, ~$30-60 on T4)
  modal run scripts/generate_dataset_modal.py

  # Custom counts
  modal run scripts/generate_dataset_modal.py --n-drape 400 --n-wind 300 --n-collision 300

  # After completion, download clips to local data/ directory
  modal volume get cloth-mpm-trajectories /data/cloth_trajectories ./data/cloth_trajectories

GPU choice: T4 (default, cheaper) vs A10G (3-4x faster, costs ~2.5x more).
To switch, change gpu="T4" to gpu="A10G" in the @app.function decorator below.
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import modal
import numpy as np
import pandas as pd
import yaml

# Project root on the local machine (one level up from scripts/)
ROOT = Path(__file__).resolve().parents[1]

# Where the project will live inside each remote container
REMOTE_ROOT = Path("/root/ClothMPM")

# Modal Volume name — created on first run, reused on subsequent runs
VOLUME_NAME = "cloth-mpm-trajectories"

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------
# CUDA 12.1 runtime + Python 3.11 + Taichi + data deps.
# scipy is required by cloth_implicit.py which is imported when src/ loads.
# ---------------------------------------------------------------------------

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install([
        "taichi>=1.7,<2.0",
        "numpy>=1.26,<2.2",
        "scipy>=1.11",
        "pandas>=2.1",
        "pyyaml>=6.0",
    ])
)

# ---------------------------------------------------------------------------
# Persistent volume for output clips
# ---------------------------------------------------------------------------
# All containers write their .npz files here. The local entrypoint writes
# index.csv here too so a single `modal volume get` fetches everything.
# ---------------------------------------------------------------------------

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# ---------------------------------------------------------------------------
# Source mount
# ---------------------------------------------------------------------------
# Syncs src/, scripts/, and configs/ from the local machine into each
# container. Only .py and .yaml files are included; data/, .venv/, etc. are
# excluded to keep the mount fast.
# ---------------------------------------------------------------------------

def _include(p: Path) -> bool:
    """Return True for files that should be synced into the remote container."""
    excluded_dirs = {
        ".venv", "__pycache__", ".git", ".pytest_cache",
        "notebooks", "results", "data",
    }
    return (
        p.suffix in {".py", ".yaml"}
        and not any(part in excluded_dirs for part in p.parts)
    )


src_mount = modal.Mount.from_local_dir(
    ROOT,
    remote_path=str(REMOTE_ROOT),
    condition=_include,
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = modal.App("cloth-mpm-datagen")

# ---------------------------------------------------------------------------
# Remote function — one clip per invocation
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="T4",                      # change to "A10G" for ~3-4x speedup
    mounts=[src_mount],
    volumes={"/data": volume},
    timeout=600,                   # 10-min hard cap; full-res T4 finishes <2 min
    retries=1,                     # retry once on container failure
)
def generate_clip(
    spec_dict: dict[str, Any],
    smoke: bool,
    idx_in_scenario: int,
) -> dict[str, Any]:
    """Simulate one MPM cloth clip on the remote GPU and save the .npz to the volume."""
    import sys
    import time
    import traceback
    from pathlib import Path

    import yaml

    # Make the mounted project importable
    sys.path.insert(0, "/root/ClothMPM")
    sys.path.insert(0, "/root/ClothMPM/scripts")

    # generate_dataset.py is a plain script with no Modal imports — safe to import
    from generate_dataset import ClipSpec, run_one_clip, save_clip

    spec = ClipSpec(**spec_dict)
    base_cfg = yaml.safe_load(
        Path("/root/ClothMPM/configs/mpm.yaml").read_text()
    )
    out_dir = Path("/data/cloth_trajectories")

    try:
        t0 = time.perf_counter()
        clip = run_one_clip(spec, base_cfg, smoke)
        wall = time.perf_counter() - t0

        path = save_clip(out_dir, clip, spec, idx_in_scenario)
        # path is e.g. /data/cloth_trajectories/drape/clip_000000_000000.npz
        # Store relative to /data so it matches the download layout:
        #   modal volume get ... /data/cloth_trajectories ./data/cloth_trajectories
        #   → ./data/cloth_trajectories/drape/clip_000000_000000.npz
        # Prepend "data/" to match the path format in generate_dataset.py (relative to ROOT)
        rel = "data/" + str(path.relative_to(Path("/data")))

        print(f"[OK] {spec.scenario} seed={spec.seed} idx={idx_in_scenario} {wall:.1f}s")
        return {
            "scenario": spec.scenario,
            "seed": spec.seed,
            "clip_idx": idx_in_scenario,
            "path": rel,
            "n_frames": int(clip["x"].shape[0]),
            "n_particles": int(clip["x"].shape[1]),
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
        }

    except Exception as exc:
        print(f"[ERROR] {spec.scenario} seed={spec.seed}: {exc}")
        traceback.print_exc()
        return {
            "scenario": spec.scenario,
            "seed": spec.seed,
            "clip_idx": idx_in_scenario,
            "path": "",
            "n_frames": 0,
            "n_particles": 0,
            "duration_s": 0.0,
            "initial_height_m": 0.0,
            "sphere_center_x": 0.0,
            "sphere_center_y": 0.0,
            "sphere_center_z": 0.0,
            "sphere_radius_m": 0.0,
            "wind_x": 0.0,
            "wind_z": 0.0,
            "n_pinned": 0,
            "config_hash": "",
            "wall_s": 0.0,
            "status": f"ERROR: {exc}",
        }


# ---------------------------------------------------------------------------
# Manifest writer — runs remotely so index.csv lives alongside the clips
# ---------------------------------------------------------------------------

@app.function(volumes={"/data": volume})
def write_manifest_to_volume(csv_text: str) -> None:
    """Write the completed index.csv into the volume."""
    from pathlib import Path
    out = Path("/data/cloth_trajectories/index.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(csv_text)
    print(f"Manifest written → {out}  ({len(csv_text.splitlines())} rows)")


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    smoke: bool = False,
    n_drape: int = 4000,
    n_wind: int = 3000,
    n_collision: int = 3000,
) -> None:
    """Build all ClipSpecs locally, fan out to Modal, collect results."""
    # Import samplers from the local copy of generate_dataset.py
    sys.path.insert(0, str(ROOT / "scripts"))
    from generate_dataset import (
        sample_drape_clip,
        sample_wind_clip,
        sample_collision_clip,
    )

    base_cfg = yaml.safe_load((ROOT / "configs" / "mpm.yaml").read_text())
    gx, gy = base_cfg["cloth"]["grid"]   # 64, 64 for full run

    if smoke:
        counts = {"drape": 2, "wind": 2, "collision": 2}
        seed_base = {"drape": 0, "wind": 100, "collision": 200}
        print("Smoke mode: 6 clips at 16×16 grid (schema + physics check).")
    else:
        counts = {
            "drape": n_drape,
            "wind": n_wind,
            "collision": n_collision,
        }
        seed_base = {"drape": 0, "wind": 100_000, "collision": 200_000}

    # Build the full task list before dispatching anything
    tasks: list[tuple[dict[str, Any], bool, int]] = []
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
            tasks.append((asdict(spec), smoke, i))

    n_total = len(tasks)
    print(f"Dispatching {n_total} clips to Modal  "
          f"({counts['drape']} drape / {counts['wind']} wind / {counts['collision']} collision)")

    # Fan out — Modal runs up to 100 containers concurrently by default.
    # order_outputs=False lets results stream in as they complete rather than
    # waiting for the slowest container in each batch.
    rows: list[dict[str, Any]] = []
    n_ok = n_err = 0
    report_every = max(1, n_total // 20)   # log progress ~20 times

    for row in generate_clip.starmap(tasks, order_outputs=False):
        if row["status"] == "OK":
            n_ok += 1
        else:
            n_err += 1
        rows.append(row)
        done = n_ok + n_err
        if done % report_every == 0 or done == n_total:
            print(f"  {done:>5}/{n_total}  ✓ {n_ok}  ✗ {n_err}")

    # Write manifest locally (index.csv matches generate_dataset.py's format)
    local_out = ROOT / "data" / "cloth_trajectories"
    local_out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    local_manifest = local_out / "index.csv"
    df.to_csv(local_manifest, index=False)
    print(f"\nLocal manifest  → {local_manifest}")

    # Mirror index.csv into the volume so `modal volume get` fetches everything
    write_manifest_to_volume.remote(df.to_csv(index=False))

    # Summary
    print(f"\n{'='*55}")
    print(f"Results: {n_ok} OK / {n_err} errors / {n_total} total")
    if n_err:
        err_rows = df[df["status"] != "OK"][["scenario", "seed", "status"]]
        print("\nFailed clips:")
        print(err_rows.to_string(index=False))

    print(f"""
Next steps
----------
1. Download clips (may take a few minutes for large datasets):
     modal volume get {VOLUME_NAME} /data/cloth_trajectories ./data/cloth_trajectories

2. Verify physics on a sample:
     cd {ROOT}
     .venv/bin/python -m pytest tests/test_dataset.py -v

3. Re-run only failed clips (edit seed_base / counts as needed):
     modal run scripts/generate_dataset_modal.py --n-drape 0 --n-wind 0 --n-collision 0
""")
