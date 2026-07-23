"""Scale-sensitivity probe on A10G: does the neural surrogate beat Taichi MPM at
higher cloth resolution? Scales cloth particles AND the MPM background grid
together (constant particle density = the physically consistent higher-fidelity
regime). Neural/fallback states are synthesized (timing is state-insensitive)."""
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
)
vol = modal.Volume.from_name("cloth-mpm-trajectories")
app = modal.App("cloth-mpm-benchscale")
OUT = "/data/results/benchmark"


def _synth_meta(cr, gr):
    return {"grid": [cr, cr], "dt_s": 1e-4, "log_every_substeps": 10,
            "pinned_corner_indices": [0, cr - 1], "initial_height_m": 1.0,
            "sphere_center_m": [1.0, 0.4, 1.0], "sphere_radius_m": 0.2}


def _synth_clip(cr, n_frames=12):
    import numpy as np
    ii, jj = np.meshgrid(np.arange(cr), np.arange(cr), indexing="ij")
    flat = np.stack([ii.ravel() / cr, np.ones(cr * cr), jj.ravel() / cr], -1).astype(np.float32)
    x = np.tile(flat[None], (n_frames, 1, 1))
    v = (0.01 * np.random.default_rng(0).standard_normal((n_frames, cr * cr, 3))).astype(np.float32)
    return {"x": x, "v": v}


@app.function(image=image, gpu="A10G", cpu=8.0, volumes={"/data": vol}, timeout=5400)
def bench_one(cr: int, gr: int, ckpt_rel: str) -> None:
    """One cloth/grid size in a fresh process (fresh ti.init -- Taichi can't be
    re-inited for a new grid in the same process). Times neural + MPM, writes a
    one-line result to the volume."""
    import sys, copy
    sys.path.insert(0, REMOTE)
    from pathlib import Path
    import numpy as np, torch
    import scripts.benchmark_gpu as bg
    from src.mpm_cloth import load_mpm_config

    N = cr * cr
    meta = _synth_meta(cr, gr)
    clip = _synth_clip(cr)
    t_neu = bg.time_neural_forward(Path(f"/data/{ckpt_rel}"), clip, meta,
                                   n_warm=20, n_time=100, device="cuda")
    cfg = load_mpm_config(f"{REMOTE}/configs/mpm.yaml")
    cfg = copy.deepcopy(cfg)
    cfg["cloth"]["grid"] = [cr, cr]
    cfg["mpm"]["grid_resolution"] = gr
    dom = float(cfg["mpm"]["domain_size_m"])
    cfg["mpm"]["dx_m"] = dom / gr
    cfg["mpm"]["inv_dx"] = gr / dom
    pinned = np.zeros(N, dtype=bool); pinned[0] = True; pinned[cr - 1] = True
    t_mpm = bg.time_mpm_substep(cfg, pinned, n_warm=20, n_time=100) * 10
    line = (f"{cr}x{cr} {N} {gr} {t_mpm*1e3:.2f} {t_neu*1e3:.2f} "
            f"{t_mpm/t_neu:.2f}\n")
    print("RESULT", line, flush=True)
    Path(OUT).mkdir(parents=True, exist_ok=True)
    Path(f"{OUT}/scale_{cr}.txt").write_text(line)
    vol.commit()


@app.local_entrypoint()
def main(ckpt_rel: str = "results/checkpoints/gnn_noF_b3k_pf5/best.pt"):
    # (cloth_res, mpm_grid_res): particle density held ~constant (grid = 2x cloth)
    sizes = [(64, 128), (96, 192), (128, 256), (160, 320)]
    for cr, gr in sizes:
        bench_one.spawn(cr, gr, ckpt_rel)
    print(f"spawned {len(sizes)} per-size benches -> /data/results/benchmark/scale_<cr>.txt")
