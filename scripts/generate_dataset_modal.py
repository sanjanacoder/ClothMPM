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
    # Taichi's native lib (taichi_python.so) links libX11 + GL even for headless
    # runs; the CUDA base image doesn't ship them. Without these, `import taichi`
    # fails with "libX11.so.6: cannot open shared object file".
    .apt_install("libx11-6", "libgl1", "libglib2.0-0")
    # Sync source into each container. Modal 1.x removed modal.Mount; local files
    # now attach to the image. copy=False keeps them as runtime mounts, so local
    # edits to the simulator/config apply without an image rebuild. These must be
    # the final image layers.
    .add_local_dir(str(ROOT / "src"), f"{REMOTE_ROOT}/src",
                   ignore=["**/__pycache__", "**/*.pyc"])
    .add_local_dir(str(ROOT / "scripts"), f"{REMOTE_ROOT}/scripts",
                   ignore=["**/__pycache__", "**/*.pyc"])
    .add_local_dir(str(ROOT / "configs"), f"{REMOTE_ROOT}/configs")
)

# ---------------------------------------------------------------------------
# Persistent volume for output clips
# ---------------------------------------------------------------------------
# All containers write their .npz files here. The local entrypoint writes
# index.csv here too so a single `modal volume get` fetches everything.
# ---------------------------------------------------------------------------

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

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
    volumes={"/data": volume},
    timeout=600,                   # 10-min hard cap; full-res T4 finishes <2 min
    retries=1,                     # retry once on container failure
    max_containers=10,             # workspace GPU cap (10 on current plan); larger
                                   # batches run in waves rather than erroring
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

    import json

    spec = ClipSpec(**spec_dict)
    base_cfg = yaml.safe_load(
        Path("/root/ClothMPM/configs/mpm.yaml").read_text()
    )
    out_dir = Path("/data/cloth_trajectories")

    # Resume: skip the (expensive) sim if this clip was already generated. Seeds
    # are deterministic, so an existing .npz + .row.json sidecar is complete and
    # identical -- lets a re-run fill only the missing seeds.
    name = f"clip_{spec.seed:06d}_{idx_in_scenario:06d}"
    sub = out_dir / spec.scenario
    npz_path = sub / f"{name}.npz"
    sidecar = sub / f"{name}.row.json"
    if npz_path.exists() and sidecar.exists():
        print(f"[SKIP] {spec.scenario} seed={spec.seed} idx={idx_in_scenario} (present)")
        return json.loads(sidecar.read_text())

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

        row = {
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
        # Write a per-clip manifest-row sidecar next to the .npz and commit, so
        # the full index.csv can be rebuilt from the volume alone -- no need for
        # the client to collect .spawn() results (which are fire-and-forget).
        import json as _json
        sidecar = path.with_name(path.stem + ".row.json")
        sidecar.write_text(_json.dumps(row))
        volume.commit()

        print(f"[OK] {spec.scenario} seed={spec.seed} idx={idx_in_scenario} {wall:.1f}s")
        return row

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

@app.function(image=image, volumes={"/data": volume})
def write_manifest_to_volume(csv_text: str) -> None:
    """Write the completed index.csv into the volume."""
    from pathlib import Path
    out = Path("/data/cloth_trajectories/index.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(csv_text)
    print(f"Manifest written → {out}  ({len(csv_text.splitlines())} rows)")


@app.function(image=image, volumes={"/data": volume}, timeout=1800)
def existing_seeds() -> set[tuple[str, int]]:
    """Return the (scenario, seed) pairs already generated on the volume, from
    the per-clip *.row.json sidecars -- used to resume without re-generating."""
    from pathlib import Path
    volume.reload()
    out: set[tuple[str, int]] = set()
    for p in Path("/data/cloth_trajectories").glob("*/*.row.json"):
        # filename: clip_{seed:06d}_{idx:06d}.row.json
        try:
            seed = int(p.stem.split("_")[1])
            out.add((p.parent.name, seed))
        except (IndexError, ValueError):
            continue
    print(f"existing_seeds: {len(out)} clips already on volume")
    return out


@app.function(image=image, volumes={"/data": volume}, timeout=1800)
def rebuild_manifest_from_volume() -> dict[str, Any]:
    """Assemble index.csv from the per-clip *.row.json sidecars on the volume.

    Decouples the manifest from the client: since generation uses .spawn()
    (fire-and-forget), results aren't collected locally. Each clip left a row
    sidecar; here we concatenate them. Idempotent -- safe to re-run any time.
    """
    import json
    from pathlib import Path

    import pandas as pd

    volume.reload()
    root = Path("/data/cloth_trajectories")
    sidecars = sorted(root.glob("*/*.row.json"))
    rows = [json.loads(p.read_text()) for p in sidecars]
    if not rows:
        print("no *.row.json sidecars found")
        return {"n": 0}
    df = pd.DataFrame(rows).sort_values(["scenario", "seed"]).reset_index(drop=True)
    out = root / "index.csv"
    out.write_text(df.to_csv(index=False))
    volume.commit()
    by = df.groupby("scenario").size().to_dict()
    print(f"Manifest rebuilt → {out}  ({len(df)} clips: {by})")
    return {"n": int(len(df)), "by_scenario": {k: int(v) for k, v in by.items()}}


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    smoke: bool = False,
    n_drape: int = 5000,
    # Wind deferred (default 0) to match scripts/generate_dataset.py: at full
    # resolution the corner-pinned sheet inverts an element the explicit solver
    # cannot integrate -> container crash. This is physics-level, NOT the
    # arm64-only issue, so it would fail on CUDA too. Sampler kept: pass
    # --n-wind N to opt in. See docs/wind-deferral.md.
    n_wind: int = 0,
    n_collision: int = 5000,
    rebuild_only: bool = False,
    force_regen: bool = False,     # re-generate even seeds already on the volume
) -> None:
    """Spawn all clips fire-and-forget (robust to client disconnect), then
    (separately) rebuild index.csv from the volume. Run with `modal run
    --detach` so the app keeps processing after this entrypoint returns.

    --rebuild-only: skip generation, just assemble index.csv from the per-clip
    *.row.json sidecars already on the volume.
    """
    if rebuild_only:
        res = rebuild_manifest_from_volume.remote()
        print(f"Manifest rebuilt: {res}")
        return

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

    # Resume: skip seeds already on the volume so a re-run generates only the
    # missing clips. Filtering here (client-side) avoids even starting a GPU
    # container for clips that already exist.
    if not smoke and not force_regen:
        existing = existing_seeds.remote()
        before = len(tasks)
        tasks = [t for t in tasks
                 if (t[0]["scenario"], t[0]["seed"]) not in existing]
        print(f"Resume: {len(existing)} clips already on volume; "
              f"skipping {before - len(tasks)}, generating {len(tasks)}.")

    n_total = len(tasks)
    print(f"Spawning {n_total} clips fire-and-forget  "
          f"({counts['drape']} drape / {counts['wind']} wind / {counts['collision']} collision)")

    # .spawn() enqueues each clip server-side and returns immediately -- the
    # detached app processes them independently of this client, so a disconnect
    # during the ~8h generation is harmless (unlike .starmap(), which Modal
    # cancels when the client stops polling). The client only needs to survive
    # this enqueue loop. Each clip writes its own .npz + row sidecar to the
    # volume; the manifest is rebuilt afterward with --rebuild-only.
    for spec_dict, sm, idx in tasks:
        generate_clip.spawn(spec_dict, sm, idx)
    print(f"Spawned {n_total} clips. They run server-side (needs `modal run --detach`).")

    print(f"""
Next steps
----------
1. Monitor:  modal app list   |   modal app logs <app-id>
   (clips + row.json sidecars stream to the volume as they finish)

2. When generation is done, build the manifest from the volume:
     modal run scripts/generate_dataset_modal.py --rebuild-only

3. Download a sample if needed:
     modal volume get {VOLUME_NAME} cloth_trajectories/index.csv ./index.csv
""")
