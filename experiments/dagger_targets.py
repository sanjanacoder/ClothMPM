"""DAgger stage 2 (revised): label collected off-manifold states with the physics
FALLBACK's own next-step acceleration (unconditionally-stable implicit solve, so
targets are always physical -- unlike an explicit MPM step from a state with no
F/C history). Target = (v_after_fallback_step - v)/dt. torch only."""
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
app = modal.App("cloth-mpm-dagger2fb")
DAG = "/data/results/dagger"


@app.function(image=image, gpu="A10G", cpu=8.0, volumes={"/data": vol}, timeout=7200)
def run() -> None:
    import sys, copy
    sys.path.insert(0, REMOTE)
    import numpy as np, torch
    from src.cloth_implicit import load_implicit_config
    from src.cloth_implicit_gpu import GPUImplicitClothSim

    dev = "cuda"
    d = np.load(f"{DAG}/states.npz")
    X, Vh = d["x"], d["vhist"]; dt = float(d["dt"]); M, N, _ = X.shape
    cfg = copy.deepcopy(load_implicit_config(f"{REMOTE}/configs/mpm.yaml"))
    sim = GPUImplicitClothSim(cfg, device=dev); sim.reset(x0=np.zeros((N, 3), np.float32), pinned=[])
    sim.sphere_enabled = False
    solver = torch.compile(sim.solve_dv, mode="reduce-overhead")

    targets = np.zeros((M, N, 3), np.float32); mags = []
    with torch.no_grad():
        for m in range(M):
            v0 = torch.as_tensor(Vh[m][:, -1], dtype=torch.float32, device=dev)
            sim.x = torch.as_tensor(X[m], dtype=torch.float32, device=dev)
            sim.v = v0.clone()
            sim.step_fixed(n_iters=30, dt=dt, solver=solver)
            a = (sim.v - v0) / dt
            targets[m] = a.cpu().numpy()
            if m % 500 == 0:
                mm = float(a.norm(dim=-1).mean()); mags.append(mm)
                print(f"  [{m}] |a| mean={mm:.2f} max={float(a.norm(dim=-1).max()):.2f}", flush=True)
    print(f"finite: {int(np.isfinite(targets).all())}  overall |a| mean~{np.mean(mags):.2f}", flush=True)
    np.savez(f"{DAG}/targets.npz", a=targets)
    vol.commit()
    print("== wrote fallback targets -> /data/results/dagger/targets.npz ==", flush=True)


@app.local_entrypoint()
def main():
    call = run.spawn()
    print(f"spawned dagger fallback-targets (call {call.object_id})")
