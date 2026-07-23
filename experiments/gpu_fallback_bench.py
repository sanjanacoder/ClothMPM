"""Time the GPU implicit fallback on A10G and recompute the hybrid speed numbers."""
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/root/mpm"
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-runtime-ubuntu22.04", add_python="3.11")
    .apt_install("gcc", "g++")
    .pip_install(["torch>=2.6", "numpy>=1.26,<2.2", "scipy>=1.11", "pandas>=2.1", "pyyaml>=6.0"])
    .add_local_dir(str(ROOT / "src"), f"{REMOTE}/src", ignore=["**/__pycache__", "**/*.pyc"])
    .add_local_dir(str(ROOT / "configs"), f"{REMOTE}/configs")
)
vol = modal.Volume.from_name("cloth-mpm-trajectories")
app = modal.App("cloth-mpm-gpufb")
OUT = "/data/results/benchmark"


@app.function(image=image, gpu="A10G", cpu=8.0, volumes={"/data": vol}, timeout=3600)
def run() -> None:
    import sys, io, contextlib, time, copy
    sys.path.insert(0, REMOTE)
    from pathlib import Path
    import numpy as np, torch
    from src.cloth_implicit import load_implicit_config
    from src.cloth_implicit_gpu import GPUImplicitClothSim

    dev = "cuda"

    def synth(cr):
        ii, jj = np.meshgrid(np.arange(cr), np.arange(cr), indexing="ij")
        return np.stack([(ii.ravel() + .5) / cr, np.full(cr * cr, 1.0),
                         (jj.ravel() + .5) / cr], -1).astype(np.float32)

    def timeit(fn, n_warm=10, n_time=50):
        for _ in range(n_warm):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_time):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_time * 1e3

    # reference numbers from prior benchmarks (A10G)
    MPM = {64: 5.6, 128: 16.5}
    NEURAL_OPT = {64: 2.11, 128: 8.02}   # fp16 + torch.compile
    FRAC = {"drape": 0.35, "collision": 0.0}

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print("CUDA:", torch.cuda.get_device_name(0))
        for cr in (64, 128):
            cfg = copy.deepcopy(load_implicit_config(f"{REMOTE}/configs/mpm.yaml"))
            cfg["cloth"]["grid"] = [cr, cr]
            sim = GPUImplicitClothSim(cfg, device=dev)
            x0 = synth(cr)
            sim.reset(x0=x0, pinned=[0, cr - 1])
            for _ in range(20):   # settle to a realistic deforming state
                sim.step(f_ext=None, dt=1e-3, cg_max_iters=50, cg_tol=1e-4)
            n = cr * cr
            # (a) eager, adaptive 50-iter / tol 1e-4 (validated path)
            t_eager = timeit(lambda: sim.step(f_ext=None, dt=1e-3, cg_max_iters=50, cg_tol=1e-4))
            # (b) eager, fixed 30 iters
            t_fixed = timeit(lambda: sim.step_fixed(n_iters=30, dt=1e-3))
            # (c) fixed 30 iters + torch.compile (CUDA graphs)
            try:
                solver = torch.compile(sim.solve_dv, mode="reduce-overhead")
                t_comp = timeit(lambda: sim.step_fixed(n_iters=30, dt=1e-3, solver=solver), n_warm=15)
            except Exception as ex:
                t_comp = float("nan"); print("compile failed:", repr(ex)[:140])
            best = min(x for x in (t_fixed, t_comp) if x == x)
            print(f"\n[cloth {cr}x{cr} = {n} particles]")
            print(f"  GPU fallback eager (50it/1e-4): {t_eager:7.2f} ms   (CPU was 119 ms @64)")
            print(f"  GPU fallback fixed-30it eager : {t_fixed:7.2f} ms")
            print(f"  GPU fallback fixed-30it +compile:{t_comp:7.2f} ms")
            print(f"  full-MPM (Taichi)     : {MPM[cr]:7.2f} ms")
            print(f"  neural (fp16+compile) : {NEURAL_OPT[cr]:7.2f} ms")
            print("  --- hybrid effective per-frame + speedup vs MPM (best fallback) ---")
            for scen, f in FRAC.items():
                t_hy = (1 - f) * NEURAL_OPT[cr] + f * best
                print(f"    {scen:>9} (f={f:.2f}): {t_hy:6.2f} ms  ->  "
                      f"{MPM[cr] / t_hy:.2f}x vs MPM")
    out = buf.getvalue()
    print(out, flush=True)
    Path(OUT).mkdir(parents=True, exist_ok=True)
    Path(f"{OUT}/gpu_fallback.txt").write_text(out)
    vol.commit()
    print("== wrote gpu_fallback.txt ==", flush=True)


@app.local_entrypoint()
def main():
    call = run.spawn()
    print(f"spawned gpu-fallback bench (call {call.object_id})")
