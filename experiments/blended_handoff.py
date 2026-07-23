"""Option (b): blended / ramped handoff. Instead of a hard neural<->fallback swap,
a fallback weight w ramps toward the detector's target (0 or 1) over blend_K frames;
during the ramp the next-state is a blend (1-w)*neural + w*fallback. This smooths the
discontinuity without ever feeding the neural a raw off-manifold state. Compares
hard vs blended handoff at matched budget on held-out drape."""
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
    .add_local_file(f"{SCRATCH}/index_heldout_large.csv", f"{REMOTE}/index_heldout_large.csv")
)
vol = modal.Volume.from_name("cloth-mpm-trajectories")
app = modal.App("cloth-mpm-blend")
OUT = "/data/results/benchmark"

# (name, exit_ratio, blend_K).  blend_K=1 -> hard switch; >1 -> ramp over K frames
POLICIES = [("naive-hard", 1.0, 1), ("hyst-hard", 0.3, 1),
            ("naive-blend8", 1.0, 8), ("hyst-blend8", 0.3, 8), ("hyst-blend15", 0.3, 15)]


@app.function(image=image, gpu="A10G", cpu=8.0, volumes={"/data": vol}, timeout=14400)
def run(ckpt_rel: str, n_steps: int, thr_pct: int) -> None:
    import sys, io, contextlib, copy
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

    @torch.no_grad()
    def rollout(clip, meta, thr, exit_ratio, blend_K, n_steps, warm=20, use_fb=True):
        gx, gy = meta["grid"]; N = int(gx) * int(gy)
        dt = float(meta["dt_s"]) * int(meta["log_every_substeps"])
        xref = torch.as_tensor(clip["x"], dtype=torch.float32, device=dev)
        vref = torch.as_tensor(clip["v"], dtype=torch.float32, device=dev)
        s0 = C - 1; steps = min(n_steps, xref.shape[0] - 1 - s0)
        ei = build_mesh_edge_index(int(gx), int(gy)).to(dev)
        F_id = torch.eye(3, device=dev).expand(N, 3, 3).contiguous(); f_ext = torch.zeros(N, 3, device=dev)
        nbr, msk, dX, Ai = strain_pre(xref[0], ei)
        x = xref[s0].clone(); v = vref[s0].clone()
        vhist = [vref[s0 - C + 1 + i].clone() for i in range(C)]
        Fprev = Fnow(x, nbr, msk, dX, Ai); fired = False; w = 0.0; wsum = 0.0; active = 0
        step_w = 1.0 / blend_K
        for k in range(steps):
            F = Fnow(x, nbr, msk, dX, Ai)
            sc = float((F - Fprev).pow(2).sum((-1, -2)).sqrt().max()); Fprev = F
            if use_fb and k >= warm:
                fire = sc > (thr * exit_ratio) if fired else sc > thr
            else:
                fire = False
            fired = fire
            target = 1.0 if fire else 0.0
            w = min(w + step_w, target) if target > w else max(w - step_w, target)
            wsum += w; active += int(w > 1e-6)
            vh = torch.stack(vhist, dim=1)
            if w <= 1e-6:                              # pure neural
                a = a_neural(x, vh, F_id, f_ext, ei); v = v + dt * a; x = x + dt * v
            elif w >= 1 - 1e-6:                        # pure fallback
                sim.x = x; sim.v = v; sim.step_fixed(n_iters=30, dt=dt, solver=solver)
                x, v = sim.x, sim.v
            else:                                      # blended next-state
                a = a_neural(x, vh, F_id, f_ext, ei); vn = v + dt * a; xn = x + dt * vn
                sim.x = x; sim.v = v; sim.step_fixed(n_iters=30, dt=dt, solver=solver)
                xf, vf = sim.x, sim.v
                x = (1 - w) * xn + w * xf; v = (1 - w) * vn + w * vf
            vhist = vhist[1:] + [v]
        l2 = float((x - xref[s0 + steps]).norm(dim=-1).mean())
        return l2, wsum / steps, active / steps

    df = pd.read_csv(f"{REMOTE}/index_heldout_large.csv")
    dr = df[(df.scenario == "drape") & (df.status == "OK")].sort_values("seed")
    clips = [(np.load(r["path"], allow_pickle=True),) for _, r in dr.iterrows()]
    clips = [(c[0], c[0]["meta"].item()) for c in clips]

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(f"CUDA {torch.cuda.get_device_name(0)}  drape clips {len(clips)}  ckpt {ckpt_rel}")
        # neural baseline + strain pool (from the neural rollout), and fallback baseline
        neu = []; allstr = []
        for c, m in clips:
            gx, gy = m["grid"]; N = int(gx) * int(gy)
            dtf = float(m["dt_s"]) * int(m["log_every_substeps"])
            xref = torch.as_tensor(c["x"], dtype=torch.float32, device=dev)
            vref = torch.as_tensor(c["v"], dtype=torch.float32, device=dev)
            ei = build_mesh_edge_index(int(gx), int(gy)).to(dev)
            F_id = torch.eye(3, device=dev).expand(N, 3, 3).contiguous(); f_ext = torch.zeros(N, 3, device=dev)
            nbr, msk, dX, Ai = strain_pre(xref[0], ei)
            s0 = C - 1; steps = min(n_steps, xref.shape[0] - 1 - s0)
            x = xref[s0].clone(); v = vref[s0].clone(); vhist = [vref[s0 - C + 1 + i].clone() for i in range(C)]
            Fprev = Fnow(x, nbr, msk, dX, Ai)
            for k in range(steps):
                Fc = Fnow(x, nbr, msk, dX, Ai)
                allstr.append(float((Fc - Fprev).pow(2).sum((-1, -2)).sqrt().max())); Fprev = Fc
                a = a_neural(x, torch.stack(vhist, dim=1), F_id, f_ext, ei)
                v = v + dtf * a; x = x + dtf * v; vhist = vhist[1:] + [v]
            neu.append(float((x - xref[s0 + steps]).norm(dim=-1).mean()))
        neu = float(np.mean(neu))
        fb = float(np.mean([rollout(c, m, -1e9, 1.0, 1, n_steps)[0] for c, m in clips]))
        thr = float(np.quantile(np.array(allstr), thr_pct / 100.0))
        print(f"\nbaselines: neural l2={neu:.4f}  fallback-only l2={fb:.4f} ({100*(1-fb/neu):+.1f}%)  thr_pct={thr_pct}")

        print(f"\n=== hard vs blended handoff (matched thr_pct={thr_pct}) ===")
        print(f"{'policy':>13} {'fb_budget':>9} {'active_fr':>9} {'l2_final':>9} {'vs_neu':>8}")
        for name, er, bK in POLICIES:
            l2s, wb, af = [], [], []
            for c, m in clips:
                l2, wbud, afr = rollout(c, m, thr, er, bK, n_steps)
                l2s.append(l2); wb.append(wbud); af.append(afr)
            print(f"{name:>13} {np.mean(wb):>9.2f} {np.mean(af):>9.2f} {np.mean(l2s):>9.4f} {100*(1-np.mean(l2s)/neu):>+7.1f}%")
    out = buf.getvalue(); print(out, flush=True)
    Path(OUT).mkdir(parents=True, exist_ok=True)
    Path(f"{OUT}/blended_handoff.txt").write_text(out); vol.commit()
    print("== wrote blended_handoff.txt ==", flush=True)


@app.local_entrypoint()
def main(ckpt_rel: str = "results/checkpoints/gnn_noF_b3k_pf5/best.pt",
         n_steps: int = 400, thr_pct: int = 50):
    call = run.spawn(ckpt_rel, n_steps, thr_pct)
    print(f"spawned blended handoff (call {call.object_id})")
