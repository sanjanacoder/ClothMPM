"""Drape hysteresis x threshold sweep on A10G: fallback-fraction vs accuracy
tradeoff (Fig-2) + best operating point. fp32 accuracy. Collision needs no sweep
(detector fires 0%)."""
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
app = modal.App("cloth-mpm-sweep")
OUT = "/data/results/benchmark"


@app.function(image=image, gpu="A10G", cpu=8.0, volumes={"/data": vol}, timeout=10800)
def run(ckpt_rel: str, n_steps: int) -> None:
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
        N = x0.shape[0]
        src, dst = ei[0].tolist(), ei[1].tolist()
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
    def accel(x, vh, F_id, f_ext, ei):
        node = assemble_node_features_for_gnn(x, vh, F_id, f_ext, mean_center=True, include_F=incF)
        edge = assemble_edge_features(x, ei)
        nn = (node - fm) / fs
        ee = ((edge - em) / es) if em is not None else edge
        return model(nn, ee, ei) * ts + tm

    @torch.no_grad()
    def rollout(clip, meta, thr, exit_ratio, fb_sim, fb_solver, n_steps, warm=20,
                record=False, use_fb=True):
        gx, gy = meta["grid"]; N = int(gx) * int(gy)
        dt = float(meta["dt_s"]) * int(meta["log_every_substeps"])
        xref = torch.as_tensor(clip["x"], dtype=torch.float32, device=dev)
        vref = torch.as_tensor(clip["v"], dtype=torch.float32, device=dev)
        s0 = C - 1; steps = min(n_steps, xref.shape[0] - 1 - s0)
        kring = build_kring_index(int(gx), int(gy)).to(dev)
        ei = build_mesh_edge_index(int(gx), int(gy)).to(dev)
        F_id = torch.eye(3, device=dev).expand(N, 3, 3).contiguous(); f_ext = torch.zeros(N, 3, device=dev)
        nbr, msk, dX, Ai = strain_pre(xref[0], ei)
        x = xref[s0].clone(); v = vref[s0].clone()
        vhist = [vref[s0 - C + 1 + i].clone() for i in range(C)]
        Fprev = Fnow(x, nbr, msk, dX, Ai)
        strains = []; nfb = 0; fired = False
        for k in range(steps):
            F = Fnow(x, nbr, msk, dX, Ai)
            sc = float((F - Fprev).pow(2).sum((-1, -2)).sqrt().max()); Fprev = F
            if use_fb and k >= warm:
                fire = sc > (thr * exit_ratio) if fired else sc > thr
            else:
                fire = False
            fired = fire
            if fire:
                fb_sim.x = x; fb_sim.v = v
                fb_sim.step_fixed(n_iters=30, dt=dt, solver=fb_solver)
                x, v = fb_sim.x, fb_sim.v; nfb += 1
            else:
                a = accel(x, torch.stack(vhist, dim=1), F_id, f_ext, ei)
                v = v + dt * a; x = x + dt * v
            vhist = vhist[1:] + [v]
            if record:
                strains.append(sc)
        l2 = float((x - xref[s0 + steps]).norm(dim=-1).mean())
        if record:
            return l2, strains
        return l2, nfb / steps

    # drape fallback (no pins), compiled solve
    cfg = copy.deepcopy(load_implicit_config(f"{REMOTE}/configs/mpm.yaml"))
    sim = GPUImplicitClothSim(cfg, device=dev); sim.reset(x0=np.zeros((64 * 64, 3), np.float32), pinned=[])
    sim.sphere_enabled = False
    solver = torch.compile(sim.solve_dv, mode="reduce-overhead")

    df = pd.read_csv(f"{REMOTE}/index_heldout_large.csv")
    dr = df[(df.scenario == "drape") & (df.status == "OK")].sort_values("seed")
    clips = [np.load(r["path"], allow_pickle=True) for _, r in dr.iterrows()]
    metas = [c["meta"].item() for c in clips]
    nt = len(clips) // 2
    Ct, Mt = clips[:nt], metas[:nt]           # TUNE half (lower seeds)
    Ce, Me = clips[nt:], metas[nt:]           # HELD-OUT EVAL half (disjoint)

    def m_neural(C, M):
        out, ss = [], []
        for c, m in zip(C, M):
            l2, s = rollout(c, m, 0, 1.0, sim, solver, n_steps, record=True, use_fb=False)
            out.append(l2); ss += s
        return float(np.mean(out)), np.array(ss)

    def m_hybrid(C, M, thr, er):
        l2s, fr = [], []
        for c, m in zip(C, M):
            l2, f = rollout(c, m, thr, er, sim, solver, n_steps)
            l2s.append(l2); fr.append(f)
        return float(np.mean(l2s)), float(np.mean(fr))

    def m_fb(C, M):
        return float(np.mean([rollout(c, m, -1e9, 1.0, sim, solver, n_steps)[0] for c, m in zip(C, M)]))

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(f"CUDA: {torch.cuda.get_device_name(0)}   TUNE clips={len(Ct)}  HELD-OUT clips={len(Ce)}")
        neu_t, str_t = m_neural(Ct, Mt)
        print(f"\n=== TUNE sweep (n={len(Ct)}, neural l2={neu_t:.4f}): thr_pctile x exit_ratio ===")
        print(f"{'thr_pct':>7} {'exit_r':>7} {'fb_frac':>8} {'l2':>8} {'vs_neu':>8}")
        best = (1e9, None)
        for pct in (50, 65, 80):
            thr = float(np.quantile(str_t, pct / 100.0))
            for er in (1.0, 0.5, 0.3, 0.15):
                ml2, mfr = m_hybrid(Ct, Mt, thr, er)
                print(f"{pct:>7} {er:>7.2f} {mfr:>8.2f} {ml2:>8.4f} {100*(1-ml2/neu_t):>+7.1f}%")
                if ml2 < best[0]:
                    best = (ml2, (pct, er, thr, mfr))
        bpct, ber, bthr, bfr = best[1]
        print(f"\nSELECTED on TUNE: thr_pct={bpct} exit_ratio={ber} "
              f"(abs_thr={bthr:.4f}, fb_frac={bfr:.2f}, tune l2={best[0]:.4f})")

        # ---- HELD-OUT EVAL with the tune-selected config (abs threshold) ----
        neu_e, _ = m_neural(Ce, Me)
        fb_e = m_fb(Ce, Me)
        hyb_e, fr_e = m_hybrid(Ce, Me, bthr, ber)
        print(f"\n=== HELD-OUT EVAL (n={len(Ce)}, disjoint clips) with tune-selected config ===")
        print(f"  neural           l2={neu_e:.4f}")
        print(f"  fallback-only    l2={fb_e:.4f}   ({100*(1-fb_e/neu_e):+.1f}% vs neural)")
        print(f"  hybrid (tuned)   l2={hyb_e:.4f}   ({100*(1-hyb_e/neu_e):+.1f}% vs neural)  fb_frac={fr_e:.2f}")
    out = buf.getvalue(); print(out, flush=True)
    Path(OUT).mkdir(parents=True, exist_ok=True)
    Path(f"{OUT}/hybrid_sweep.txt").write_text(out); vol.commit()
    print("== wrote hybrid_sweep.txt ==", flush=True)


@app.local_entrypoint()
def main(ckpt_rel: str = "results/checkpoints/gnn_noF_b3k_pf5/best.pt", n_steps: int = 400):
    call = run.spawn(ckpt_rel, n_steps)
    print(f"spawned drape sweep (call {call.object_id})")
