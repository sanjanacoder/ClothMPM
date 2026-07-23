"""Handoff ablation (mentor-requested). Under a ~matched fallback budget, vary the
NUMBER of solver transitions (naive per-frame switching vs hysteresis vs fixed
minimum fallback-burst lengths) and test whether fragmented switching is worse than
contiguous fallback. Reports: #switches, fallback fraction, rollout error over time,
final error, and neural-vs-fallback acceleration disagreement around transitions.
Drape held-out clips, fp32 accuracy, GPU fallback."""
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
app = modal.App("cloth-mpm-ablation")
OUT = "/data/results/benchmark"

# policies: (name, exit_ratio, min_burst).  naive = flip per frame (max fragmentation)
POLICIES = [("naive", 1.0, 1), ("hyst0.3", 0.3, 1),
            ("burst5", 1.0, 5), ("burst20", 1.0, 20), ("burst50", 1.0, 50)]
CKPTS = [25, 50, 100, 150, 200, 300, 400]


@app.function(image=image, gpu="A10G", cpu=8.0, volumes={"/data": vol}, timeout=14400)
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

    # drape fallback (no pins), compiled
    cfg = copy.deepcopy(load_implicit_config(f"{REMOTE}/configs/mpm.yaml"))
    sim = GPUImplicitClothSim(cfg, device=dev); sim.reset(x0=np.zeros((64 * 64, 3), np.float32), pinned=[])
    sim.sphere_enabled = False
    solver = torch.compile(sim.solve_dv, mode="reduce-overhead")

    @torch.no_grad()
    def rollout(clip, meta, thr, exit_ratio, min_burst, n_steps, warm=20,
                record_curve=False, record_dis=False, use_fb=True):
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
        nfb = 0; ntrans = 0; fired = False; burst = 0
        curve = {}; dis_tr = []; dis_st = []; dis_since = []; since = 999
        strain_pool = []
        for k in range(steps):
            F = Fnow(x, nbr, msk, dX, Ai)
            sc = float((F - Fprev).pow(2).sum((-1, -2)).sqrt().max()); Fprev = F
            strain_pool.append(sc)
            # unified policy fire decision
            if use_fb and k >= warm:
                if burst > 0:
                    fire = True; burst -= 1
                else:
                    thr_eff = thr * exit_ratio if fired else thr
                    fire = sc > thr_eff
                    if fire and min_burst > 1:
                        burst = min_burst - 1
            else:
                fire = False
            is_trans = (k >= warm) and (fire != fired)
            if is_trans:
                ntrans += 1; since = 0
            # disagreement: both solvers evaluated on the CURRENT state (non-commit)
            if record_dis and k >= warm:
                an = a_neural(x, torch.stack(vhist, dim=1), F_id, f_ext, ei)
                sim.x = x.clone(); sim.v = v.clone()
                sim.step_fixed(n_iters=30, dt=dt, solver=solver)
                af = (sim.v - v) / dt
                d = float((an - af).norm(dim=-1).mean())
                (dis_tr if is_trans else dis_st).append(d)
                dis_since.append((min(since, 30), d))
            # apply the policy-selected solver
            if fire:
                sim.x = x; sim.v = v
                sim.step_fixed(n_iters=30, dt=dt, solver=solver)
                x, v = sim.x, sim.v; nfb += 1
            else:
                a = a_neural(x, torch.stack(vhist, dim=1), F_id, f_ext, ei)
                v = v + dt * a; x = x + dt * v
            vhist = vhist[1:] + [v]
            fired = fire; since += 1
            if record_curve and (k + 1) in CKPTS:
                curve[k + 1] = float((x - xref[s0 + 1 + k]).norm(dim=-1).mean())
        l2 = float((x - xref[s0 + steps]).norm(dim=-1).mean())
        out = {"l2": l2, "fb_frac": nfb / steps, "ntrans": ntrans, "strain": strain_pool}
        if record_curve:
            out["curve"] = curve
        if record_dis:
            out["dis_tr"] = dis_tr; out["dis_st"] = dis_st; out["dis_since"] = dis_since
        return out

    df = pd.read_csv(f"{REMOTE}/index_heldout_large.csv")
    dr = df[(df.scenario == "drape") & (df.status == "OK")].sort_values("seed")
    clips = [(np.load(r["path"], allow_pickle=True), None) for _, r in dr.iterrows()]
    clips = [(c, c["meta"].item()) for c, _ in clips]

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(f"CUDA {torch.cuda.get_device_name(0)}   drape clips: {len(clips)}")
        # neural baseline + strain pool
        neu = []; pool = []
        for c, m in clips:
            r = rollout(c, m, 0, 1.0, 1, n_steps, use_fb=False)
            neu.append(r["l2"]); pool += r["strain"]
        neu = float(np.mean(neu)); pool = np.array(pool)
        print(f"neural baseline l2={neu:.4f}\n")

        # PART A: policies x thresholds -> (fb_frac, ntrans, final l2)
        print("=== PART A: transitions vs error across policies x thresholds ===")
        print(f"{'policy':>9} {'thr_pct':>7} {'fb_frac':>8} {'n_switch':>9} {'l2_final':>9} {'vs_neu':>8}")
        gridA = {}
        for name, er, mb in POLICIES:
            for pct in (50, 68, 85):
                thr = float(np.quantile(pool, pct / 100.0))
                l2s, fr, tr = [], [], []
                for c, m in clips:
                    r = rollout(c, m, thr, er, mb, n_steps)
                    l2s.append(r["l2"]); fr.append(r["fb_frac"]); tr.append(r["ntrans"])
                ml2, mfr, mtr = np.mean(l2s), np.mean(fr), np.mean(tr)
                gridA[(name, pct)] = (mfr, mtr, ml2)
                print(f"{name:>9} {pct:>7} {mfr:>8.2f} {mtr:>9.1f} {ml2:>9.4f} {100*(1-ml2/neu):>+7.1f}%")

        # PART B: pick each policy's config nearest ~0.40 fallback; matched-budget compare
        TARGET = 0.40
        print(f"\n=== PART B: matched fallback ~{TARGET:.2f} -- error-over-time + disagreement ===")
        sub = clips[:12]
        print(f"{'policy':>9} {'fb_frac':>8} {'n_switch':>9} {'l2_final':>9} "
              f"{'dis@trans':>10} {'dis@steady':>11} {'ratio':>6}")
        curves = {}
        for name, er, mb in POLICIES:
            # choose threshold percentile giving fb_frac closest to TARGET (from Part A on full set)
            pct = min((50, 68, 85), key=lambda p: abs(gridA[(name, p)][0] - TARGET))
            thr = float(np.quantile(pool, pct / 100.0))
            l2s, frs, trs, dtr, dst, since_pairs, cur = [], [], [], [], [], [], {ck: [] for ck in CKPTS}
            for c, m in sub:
                r = rollout(c, m, thr, er, mb, n_steps, record_curve=True, record_dis=True)
                l2s.append(r["l2"]); frs.append(r["fb_frac"]); trs.append(r["ntrans"])
                dtr += r["dis_tr"]; dst += r["dis_st"]; since_pairs += r["dis_since"]
                for ck in CKPTS:
                    if ck in r["curve"]:
                        cur[ck].append(r["curve"][ck])
            mdtr = float(np.mean(dtr)) if dtr else float("nan")
            mdst = float(np.mean(dst)) if dst else float("nan")
            curves[name] = (np.mean(frs), np.mean(trs), np.mean(l2s),
                            {ck: (float(np.mean(cur[ck])) if cur[ck] else float("nan")) for ck in CKPTS},
                            since_pairs)
            print(f"{name:>9} {np.mean(frs):>8.2f} {np.mean(trs):>9.1f} {np.mean(l2s):>9.4f} "
                  f"{mdtr:>10.3f} {mdst:>11.3f} {mdtr/mdst:>6.2f}")

        print("\n=== error over time L2(t) at matched budget ===")
        print("policy   " + "".join(f"{('t'+str(ck)):>9}" for ck in CKPTS))
        for name, _, _ in POLICIES:
            cc = curves[name][3]
            print(f"{name:>8} " + "".join(f"{cc[ck]:>9.4f}" for ck in CKPTS))

        print("\n=== disagreement vs frames-since-last-switch (naive vs burst20) ===")
        for name in ("naive", "burst20"):
            pairs = curves[name][4]
            print(f"  {name}:")
            for lo, hi in [(0, 0), (1, 2), (3, 5), (6, 10), (11, 30)]:
                ds = [d for s, d in pairs if lo <= s <= hi]
                if ds:
                    print(f"    since_switch {lo}-{hi}: mean_disagreement={np.mean(ds):.3f}  (n={len(ds)})")
    out = buf.getvalue(); print(out, flush=True)
    Path(OUT).mkdir(parents=True, exist_ok=True)
    Path(f"{OUT}/handoff_ablation.txt").write_text(out); vol.commit()
    print("== wrote handoff_ablation.txt ==", flush=True)


@app.local_entrypoint()
def main(ckpt_rel: str = "results/checkpoints/gnn_noF_b3k_pf5/best.pt", n_steps: int = 400):
    call = run.spawn(ckpt_rel, n_steps)
    print(f"spawned handoff ablation (call {call.object_id})")
