"""Modal A10G runner for the H3 GPU speed benchmark."""
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/root/mpm"
SCRATCH = str(Path(__file__).resolve().parent / "manifests")

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-runtime-ubuntu22.04", add_python="3.11")
    .pip_install(["taichi>=1.7,<2.0", "torch>=2.6", "numpy>=1.26,<2.2",
                  "scipy>=1.11", "pandas>=2.1", "pyyaml>=6.0"])
    .apt_install("libx11-6", "libgl1", "libglib2.0-0")
    .add_local_dir(str(ROOT / "src"), f"{REMOTE}/src", ignore=["**/__pycache__", "**/*.pyc"])
    .add_local_dir(str(ROOT / "scripts"), f"{REMOTE}/scripts", ignore=["**/__pycache__", "**/*.pyc"])
    .add_local_dir(str(ROOT / "configs"), f"{REMOTE}/configs")
    .add_local_file(f"{SCRATCH}/index_heldout_npz.csv", f"{REMOTE}/index_heldout_npz.csv")
)
vol = modal.Volume.from_name("cloth-mpm-trajectories")
app = modal.App("cloth-mpm-benchmark")
OUT = "/data/results/benchmark"


@app.function(image=image, gpu="A10G", cpu=8.0, volumes={"/data": vol}, timeout=3600)
def run(ckpt_rel: str) -> None:
    import sys, io, contextlib
    sys.path.insert(0, REMOTE)
    from pathlib import Path
    import numpy as np, pandas as pd, torch
    import scripts.benchmark_gpu as bg
    from src.mpm_cloth import load_mpm_config

    print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
    df = pd.read_csv(f"{REMOTE}/index_heldout_npz.csv")
    df = df[df["status"] == "OK"].reset_index(drop=True)
    # collision clip -> realistic sphere scene for MPM/fallback
    r = df[df.scenario == "collision"].iloc[0]
    clip = np.load(r["path"], allow_pickle=True)
    meta = clip["meta"].item()
    gx, gy = meta["grid"]
    N = int(gx) * int(gy)
    pinned = np.zeros(N, dtype=bool)
    for i in meta.get("pinned_corner_indices", []):
        pinned[int(i)] = True

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        # neural (torch/cuda) first, then fallback (cpu), then MPM (taichi/cuda)
        t_neural = bg.time_neural_forward(Path(f"/data/{ckpt_rel}"), clip, meta, device="cuda")
        print(f"[timed] neural forward: {t_neural*1e3:.3f} ms")
        t_fb = bg.time_fallback_step(meta, clip)
        print(f"[timed] implicit fallback: {t_fb*1e3:.3f} ms")
        cfg = load_mpm_config(f"{REMOTE}/configs/mpm.yaml")
        log_every = int(cfg["mpm"]["substeps_per_log"])
        t_mpm_sub = bg.time_mpm_substep(cfg, pinned)
        print(f"[timed] MPM substep: {t_mpm_sub*1e3:.3f} ms  (log_every={log_every})")
        bg.report(t_mpm_sub, log_every, t_neural, t_fb,
                  fallback_fracs={"drape": 0.35, "collision": 0.0})
    out = buf.getvalue()
    print(out, flush=True)
    Path(OUT).mkdir(parents=True, exist_ok=True)
    Path(f"{OUT}/benchmark.txt").write_text(out)
    vol.commit()
    print("== wrote /data/results/benchmark/benchmark.txt ==", flush=True)


@app.local_entrypoint()
def main(ckpt_rel: str = "results/checkpoints/gnn_noF_b3k_pf5/best.pt"):
    call = run.spawn(ckpt_rel)
    print(f"spawned GPU benchmark (call {call.object_id}); result -> /data/results/benchmark/")
