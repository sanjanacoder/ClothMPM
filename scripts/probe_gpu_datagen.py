"""One-off GPU probe: measure full-res MPM clip generation time on T4 vs A10G.

Datagen is pure GPU simulation (no dataloader), so unlike training it is
compute-bound and a faster GPU should pay off. This times run_one_clip on the
SAME clips on both GPUs and does NOT write to the volume (no save_clip, no
manifest) -- purely a timing measurement to size the 10k run.

  modal run scripts/probe_gpu_datagen.py
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import modal
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/ClothMPM")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-runtime-ubuntu22.04", add_python="3.11",
    )
    .pip_install([
        "taichi>=1.7,<2.0", "numpy>=1.26,<2.2", "scipy>=1.11",
        "pandas>=2.1", "pyyaml>=6.0",
    ])
    .apt_install("libx11-6", "libgl1", "libglib2.0-0")
    .add_local_dir(str(ROOT / "src"), f"{REMOTE_ROOT}/src",
                   ignore=["**/__pycache__", "**/*.pyc"])
    .add_local_dir(str(ROOT / "scripts"), f"{REMOTE_ROOT}/scripts",
                   ignore=["**/__pycache__", "**/*.pyc"])
    .add_local_dir(str(ROOT / "configs"), f"{REMOTE_ROOT}/configs")
)

app = modal.App("clothmpm-gpu-probe")


@app.function(image=image, gpu="T4", timeout=900)
def probe_clip(spec_dict: dict, gpu_label: str) -> dict:
    """Simulate one full-res clip and time it; no persistence."""
    import subprocess
    import time

    sys.path.insert(0, "/root/ClothMPM")
    sys.path.insert(0, "/root/ClothMPM/scripts")
    from generate_dataset import ClipSpec, run_one_clip

    base_cfg = yaml.safe_load(
        Path("/root/ClothMPM/configs/mpm.yaml").read_text())
    spec = ClipSpec(**spec_dict)

    try:
        dev = subprocess.run(["nvidia-smi", "--query-gpu=name",
                              "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
    except Exception:
        dev = gpu_label
    t0 = time.perf_counter()
    clip = run_one_clip(spec, base_cfg, False)   # full-res
    wall = time.perf_counter() - t0
    finite = bool(np.isfinite(clip["x"]).all())
    print(f"[{gpu_label}] {dev} seed={spec.seed} {wall:.1f}s "
          f"frames={clip['x'].shape[0]} finite={finite}")
    return {"gpu": gpu_label, "device": dev, "seed": spec.seed,
            "wall_s": wall, "finite": finite}


@app.local_entrypoint()
def main(n: int = 2):
    sys.path.insert(0, str(ROOT / "scripts"))
    from generate_dataset import sample_drape_clip

    base_cfg = yaml.safe_load((ROOT / "configs" / "mpm.yaml").read_text())
    specs = [asdict(sample_drape_clip(np.random.default_rng(s), base_cfg, s))
             for s in range(n)]

    fns = {"T4": probe_clip, "A10G": probe_clip.with_options(gpu="A10G")}
    results = []
    for label, fn in fns.items():
        for sd in specs:
            results.append(fn.remote(sd, label))

    print("\n=== per-clip wall (full-res, 4096 particles) ===")
    by_gpu: dict[str, list[float]] = {}
    for r in results:
        by_gpu.setdefault(r["gpu"], []).append(r["wall_s"])
        print(f"  {r['gpu']:>5}  seed={r['seed']}  {r['wall_s']:.1f}s  finite={r['finite']}")
    print("\n=== mean per-clip ===")
    means = {g: sum(v) / len(v) for g, v in by_gpu.items()}
    for g, m in means.items():
        print(f"  {g:>5}: {m:.1f}s")
    if "T4" in means and "A10G" in means and means["A10G"] > 0:
        print(f"\nA10G speedup over T4: {means['T4'] / means['A10G']:.2f}x")
