"""Modal training entrypoint for the neural grid-update (MLP / GNN).

Runs one training job on a Modal GPU against the full-res cloth batch that
already lives on the `cloth-mpm-trajectories` volume (written by ClothMPM's
datagen). Two steps:

  1. prepare_mmap (CPU): one-time convert the volume's .npz clips to the
     per-array .npy mmap store (scripts/npz_to_mmap.py) so the GPU run is not
     dataloader-bound. Persisted on the volume at /data/fullres100_mmap.
  2. train (GPU): run src.train from the mmap store; checkpoints + log land on
     the volume under /data/results so they survive the container.

Usage (from the parent repo root):
  # smoke: few frames / epochs, cheap, proves the path end-to-end
  modal run scripts/train_modal.py --smoke

  # the F-out-of-scope gate: full batch, x,v -> a
  modal run scripts/train_modal.py --epochs 30 --name mlp_noF_100clip

  # fetch results locally
  modal volume get cloth-mpm-trajectories results ./results_modal
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/root/mpm"
VOLUME_NAME = "cloth-mpm-trajectories"

# torch (CUDA wheel) + the handful of deps src.train needs. No taichi/scipy:
# src/__init__.py is inert and the training path only imports torch, numpy,
# pandas, yaml.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-runtime-ubuntu22.04", add_python="3.11"
    )
    .pip_install([
        "torch>=2.6",
        "numpy>=1.26,<2.2",
        "pandas>=2.1",
        "pyyaml>=6.0",
    ])
    .add_local_dir(str(ROOT / "src"), f"{REMOTE_ROOT}/src",
                   ignore=["**/__pycache__", "**/*.pyc"])
    .add_local_dir(str(ROOT / "scripts"), f"{REMOTE_ROOT}/scripts",
                   ignore=["**/__pycache__", "**/*.pyc"])
    .add_local_dir(str(ROOT / "configs"), f"{REMOTE_ROOT}/configs")
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App("cloth-mpm-train")

# Volume layout (mounted at /data in every container):
#   /data/cloth_trajectories/{drape,collision}/*.npz + index.csv   (source)
#   /data/fullres100_mmap/...                                       (built here)
#   /data/results/...                                              (checkpoints)
MMAP_DIR = "/data/fullres100_mmap"
MMAP_MANIFEST = f"{MMAP_DIR}/index.csv"


@app.function(image=image, volumes={"/data": volume}, timeout=3600)
def prepare_mmap(force: bool = False) -> str:
    """Convert the volume's .npz clips to the per-array .npy mmap store."""
    import pandas as pd

    sys.path.insert(0, REMOTE_ROOT)
    sys.path.insert(0, f"{REMOTE_ROOT}/scripts")
    import npz_to_mmap

    if Path(MMAP_MANIFEST).exists() and not force:
        n = len(pd.read_csv(MMAP_MANIFEST))
        print(f"mmap store already present ({n} clips) -> {MMAP_DIR}")
        return MMAP_MANIFEST

    # The volume manifest paths are "data/cloth_trajectories/..."; in-container
    # the files sit at /data/cloth_trajectories/... Rewrite to absolute so the
    # converter (and the trainer) resolve them regardless of ROOT.
    src_manifest = Path("/data/cloth_trajectories/index.csv")
    df = pd.read_csv(src_manifest)
    df["path"] = df["path"].map(lambda p: "/" + p if not p.startswith("/") else p)
    abs_manifest = Path("/tmp/index_abs.csv")
    df.to_csv(abs_manifest, index=False)

    # npz_to_mmap resolves ROOT/path; absolute paths make ROOT a no-op.
    npz_to_mmap.ROOT = Path("/")
    npz_to_mmap.convert(abs_manifest, Path(MMAP_DIR), include_F=False,
                        scenarios=None)
    volume.commit()
    print(f"mmap store built -> {MMAP_DIR}")
    return MMAP_MANIFEST


@app.function(image=image, gpu="T4", cpu=16.0, volumes={"/data": volume},
              timeout=28800)   # 8h: GNN 10 epochs ~6h; headroom for bigger runs
def train(
    config: str = "configs/mlp.yaml",
    scenarios: list[str] | None = None,
    epochs: int = 30,
    name: str = "mlp_noF_100clip",
    smoke: bool = False,
    num_workers: int = 4,
    batch_size: int | None = None,
) -> str:
    """Run src.train from the mmap store on the GPU; persist checkpoints."""
    import os

    import torch

    print(f"CUDA available: {torch.cuda.is_available()}  "
          f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}"
          f"  cpus: {os.cpu_count()}")

    cmd = [
        sys.executable, "-u", "-m", "src.train",   # -u: unbuffered -> live logs
        "--config", config,
        "--manifest", MMAP_MANIFEST,
        "--no-F",
        "--out-root", "/data/results",
        "--name", name,
        "--epochs", str(epochs),
        "--num-workers", str(num_workers),
    ]
    if batch_size is not None:
        # Bigger batches saturate the GPU: the MLP run was compute-bound with
        # ~11k tiny steps/epoch at batch 8. 64+ means far fewer, larger steps.
        cmd += ["--batch-size", str(batch_size)]
    if scenarios:
        cmd += ["--scenarios", *scenarios]
    if smoke:
        cmd += ["--max-train-frames", "800"]

    print("running:", " ".join(cmd), flush=True)
    # Stream line-by-line (stderr merged) so `modal app logs` shows live epoch
    # progress and any error -- capturing hides everything until exit and loses
    # it if the container stops. Keep the tail to return to the caller.
    lines: list[str] = []
    proc = subprocess.Popen(cmd, cwd=REMOTE_ROOT, text=True, bufsize=1,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"training failed (exit {rc}); see logs above")
    volume.commit()

    tail = "".join(lines).strip().splitlines()[-12:]
    return "\n".join(tail)


@app.local_entrypoint()
def main(
    config: str = "configs/mlp.yaml",
    epochs: int = 30,
    name: str = "mlp_noF_100clip",
    scenarios: str = "",           # comma-separated, e.g. "drape,collision"
    smoke: bool = False,
    force_prepare: bool = False,
    num_workers: int = 12,         # feature assembly is the bottleneck; feed the
                                   # GPU from many cores (train fn reserves 16)
    batch_size: int = 64,          # bigger than the config's 8 -> fewer GPU steps
    gpu: str = "T4",               # "A10G"/"L4"/... for more compute
):
    scen = [s for s in scenarios.split(",") if s] or None
    if smoke:
        epochs = min(epochs, 3)
        name = name if name != "mlp_noF_100clip" else "mlp_noF_smoke"

    print("== preparing mmap store ==")
    manifest = prepare_mmap.remote(force=force_prepare)
    print(f"manifest: {manifest}")

    print(f"== training ({'smoke' if smoke else 'full'}, epochs={epochs}, "
          f"batch={batch_size}, gpu={gpu}) ==")
    # with_options lets us pick the GPU class at call time without redefining
    # the function (decorator default is T4).
    train_fn = train.with_options(gpu=gpu) if gpu != "T4" else train
    kw = dict(config=config, scenarios=scen, epochs=epochs, name=name,
              smoke=smoke, num_workers=num_workers, batch_size=batch_size)

    if smoke:
        # Short run: block and print the loss tail for immediate feedback.
        print(train_fn.remote(**kw))
    else:
        # Long run: fire-and-forget. .remote() blocks the local client for
        # hours and gets cancelled on any disconnect; .spawn() runs fully
        # server-side (pair with `modal run --detach`). Monitor via app logs;
        # checkpoints land on the volume as val improves.
        call = train_fn.spawn(**kw)
        print(f"spawned training (call {call.object_id}) -- runs server-side.")
        print(f"  monitor:  modal app logs <app-id>   (modal app list to find it)")
        print(f"  checkpoints stream to the volume at "
              f"results/checkpoints/{name}/best.pt")
    print(f"\nfetch: modal volume get {VOLUME_NAME} results ./results_modal")
