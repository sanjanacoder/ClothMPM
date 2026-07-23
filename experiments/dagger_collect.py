"""DAgger stage 1: collect off-manifold (handoff + fallback) states from hysteresis
hybrid rollouts on drape TRAINING clips. Saves (x, v_history) per collected frame
to the volume for MPM target labelling (stage 2)."""
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/root/mpm"
SCRATCH = str(Path(__file__).resolve().parent / "manifests")
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-runtime-ubuntu22.04", add_python="3.11")
    .apt_install("gcc", "g++")
    .pip_install(["torch>=2.6", "numpy>=1.26,<2.2", "scipy>=1.11", "pandas>=2.1", "pyyaml>=6.0"])
    .add_local_dir(str(ROOT / "src"), f"{REMOTE}/src", ignore=["**/__pycache__", "**/*.pyc"])
    .add_local_dir(str(ROOT / "scripts"), f"{REMOTE}/scripts", ignore=["**/__pycache__", "**/*.pyc"])
    .add_local_dir(str(ROOT / "configs"), f"{REMOTE}/configs")
    .add_local_file(f"{SCRATCH}/index_dagger_npz.csv", f"{REMOTE}/index_dagger_npz.csv")
)
vol = modal.Volume.from_name("cloth-mpm-trajectories")
app = modal.App("cloth-mpm-dagger1")
OUT = "/data/results/dagger"


@app.function(image=image, gpu="A10G", cpu=8.0, volumes={"/data": vol}, timeout=10800)
def run(ckpt_rel: str, n_steps: int, per_clip: int) -> None:
    import sys, copy
    sys.path.insert(0, REMOTE)
    from pathlib import Path
    import numpy as np, pandas as pd, torch
    from src.data import (assemble_edge_features, assemble_node_features_for_gnn,
                          build_kring_index, build_mesh_edge_index)
    from src.cloth_implicit import load_implicit_config
    from src.cloth_implicit_gpu import GPUImplicitClothSim
    from scripts.eval_rollout import _load_ckpt

    dev = "cuda"
    model, stats, mk, incF, C = _load_ckpt(Path(f"/data/{ckpt_rel}"))
    model = model.to(dev).eval()
    fm = stats.feat_mean.to(dev); fs = stats.feat_std.to(dev)
    em = stats.edge_mean.to(dev) if stats.edge_mean is not None else None
    es = stats.edge_std.to(dev) if stats.edge_std is not None else None
    tm = stats.target_mean.to(dev); ts = stats.target_std.to(dev)

    def strain_pre(x0, ei):
        N = x0.shape[0]; src, dst = ei[0].tolist(), ei[1].tolist()
        nl = [[] for _ in range(N)]
        for s, d in zip(src, dst):
            nl[s].append(d)
        K = max(len(a) for a in nl)
        nbr = torch.zeros(N, K, dtype=torch.long, device=dev); msk = torch.zeros(N, K, 1, device=dev)
        for i, a in enumerate(nl):
            for j, dd in enumerate(a):
                nbr[i, j] = dd; msk[i, j, 0] = 1.0
        dX = (x0[nbr] - x0[:, None, :]) * msk
        A = torch.einsum("nki,nkj->nij", dX, dX) + 1e-6 * torch.eye(3, device=dev)
        return nbr, msk, dX, torch.linalg.inv(A)

    def Fnow(x, nbr, msk, dX, Ai):
        dx = (x[nbr] - x[:, None, :]) * msk
        return torch.einsum("nki,nkj->nij", dx, dX) @ Ai

    @torch.no_grad()
    def a_neural(x, vh, F_id, f_ext, ei):
        node = assemble_node_features_for_gnn(x, vh, F_id, f_ext, mean_center=True, include_F=incF)
        edge = assemble_edge_features(x, ei)
        nn = (node - fm) / fs
        ee = ((edge - em) / es) if em is not None else edge
        return model(nn, ee, ei) * ts + tm

    cfg = copy.deepcopy(load_implicit_config(f"{REMOTE}/configs/mpm.yaml"))
    sim = GPUImplicitClothSim(cfg, device=dev); sim.reset(x0=np.zeros((64 * 64, 3), np.float32), pinned=[])
    sim.sphere_enabled = False
    solver = torch.compile(sim.solve_dv, mode="reduce-overhead")

    df = pd.read_csv(f"{REMOTE}/index_dagger_npz.csv")
    df = df[df.status == "OK"]
    Xs, Vhs = [], []      # collected (N,3) and (N,C,3)
    thr_pct = 50; exit_ratio = 0.3; warm = 20

    @torch.no_grad()
    def collect(clip, meta):
        gx, gy = meta["grid"]; N = int(gx) * int(gy)
        dt = float(meta["dt_s"]) * int(meta["log_every_substeps"])
        xref = torch.as_tensor(clip["x"], dtype=torch.float32, device=dev)
        vref = torch.as_tensor(clip["v"], dtype=torch.float32, device=dev)
        s0 = C - 1; steps = min(n_steps, xref.shape[0] - 1 - s0)
        kring = build_kring_index(int(gx), int(gy)).to(dev)
        ei = build_mesh_edge_index(int(gx), int(gy)).to(dev)
        F_id = torch.eye(3, device=dev).expand(N, 3, 3).contiguous(); f_ext = torch.zeros(N, 3, device=dev)
        nbr, msk, dX, Ai = strain_pre(xref[0], ei)
        # per-clip threshold from this clip's neural-rollout strain
        pool = []
        x = xref[s0].clone(); v = vref[s0].clone()
        vhist = [vref[s0 - C + 1 + i].clone() for i in range(C)]
        Fprev = Fnow(x, nbr, msk, dX, Ai)
        for k in range(steps):
            F = Fnow(x, nbr, msk, dX, Ai)
            pool.append(float((F - Fprev).pow(2).sum((-1, -2)).sqrt().max())); Fprev = F
            a = a_neural(x, torch.stack(vhist, dim=1), F_id, f_ext, ei)
            v = v + dt * a; x = x + dt * v; vhist = vhist[1:] + [v]
        thr = float(np.quantile(np.array(pool), thr_pct / 100.0))
        # hybrid rollout, collecting handoff-region + fallback states
        x = xref[s0].clone(); v = vref[s0].clone()
        vhist = [vref[s0 - C + 1 + i].clone() for i in range(C)]
        Fprev = Fnow(x, nbr, msk, dX, Ai); fired = False; since = 999; got = 0
        for k in range(steps):
            F = Fnow(x, nbr, msk, dX, Ai)
            sc = float((F - Fprev).pow(2).sum((-1, -2)).sqrt().max()); Fprev = F
            if k >= warm:
                fire = sc > (thr * exit_ratio) if fired else sc > thr
            else:
                fire = False
            if k >= warm and fire != fired:
                since = 0
            # collect: post-handoff region (<=12 frames after a switch) or fallback frames
            if k >= warm and got < per_clip and (since <= 12 or fire):
                Xs.append(x.detach().cpu().numpy().astype(np.float32))
                Vhs.append(torch.stack(vhist, dim=1).detach().cpu().numpy().astype(np.float32))
                got += 1
            if fire:
                sim.x = x; sim.v = v
                sim.step_fixed(n_iters=30, dt=dt, solver=solver)
                x, v = sim.x, sim.v
            else:
                a = a_neural(x, torch.stack(vhist, dim=1), F_id, f_ext, ei)
                v = v + dt * a; x = x + dt * v
            vhist = vhist[1:] + [v]; fired = fire; since += 1

    for _, r in df.iterrows():
        clip = np.load(r["path"], allow_pickle=True)
        collect(clip, clip["meta"].item())

    X = np.stack(Xs); Vh = np.stack(Vhs)
    print(f"collected {X.shape[0]} frames  x{X.shape}  vh{Vh.shape}", flush=True)
    Path(OUT).mkdir(parents=True, exist_ok=True)
    np.savez(f"{OUT}/states.npz", x=X, vhist=Vh, dt=np.float32(1e-3), grid=np.int32([64, 64]))
    vol.commit()
    print("== wrote /data/results/dagger/states.npz ==", flush=True)


@app.local_entrypoint()
def main(ckpt_rel: str = "results/checkpoints/gnn_noF_b3k_pf5/best.pt",
         n_steps: int = 400, per_clip: int = 35):
    call = run.spawn(ckpt_rel, n_steps, per_clip)
    print(f"spawned dagger collect (call {call.object_id})")
