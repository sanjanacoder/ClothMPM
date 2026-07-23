"""Neural inference speed-optimization probe (A10G, torch only -- no taichi).

Breaks the neural per-step cost into feature-assembly vs model-forward, then tests
optimizations on the model forward: torch.compile (CUDA graphs) and fp16. Times at
4096 and 16384 particles."""
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/root/mpm"
SCRATCH = str(Path(__file__).resolve().parent / "manifests")
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-runtime-ubuntu22.04", add_python="3.11")
    .apt_install("gcc", "g++")   # torch.compile / triton need a C compiler
    .pip_install(["torch>=2.6", "numpy>=1.26,<2.2", "pandas>=2.1", "pyyaml>=6.0"])
    .add_local_dir(str(ROOT / "src"), f"{REMOTE}/src", ignore=["**/__pycache__", "**/*.pyc"])
    .add_local_dir(str(ROOT / "scripts"), f"{REMOTE}/scripts", ignore=["**/__pycache__", "**/*.pyc"])
    .add_local_dir(str(ROOT / "configs"), f"{REMOTE}/configs")
    .add_local_file(f"{SCRATCH}/index_heldout_npz.csv", f"{REMOTE}/index_heldout_npz.csv")
)
vol = modal.Volume.from_name("cloth-mpm-trajectories")
app = modal.App("cloth-mpm-neuralopt")
OUT = "/data/results/benchmark"


def _synth_state(cr):
    import numpy as np
    ii, jj = np.meshgrid(np.arange(cr), np.arange(cr), indexing="ij")
    flat = np.stack([ii.ravel() / cr, np.ones(cr * cr), jj.ravel() / cr], -1).astype(np.float32)
    x = np.tile(flat[None], (12, 1, 1))
    v = (0.01 * np.random.default_rng(0).standard_normal((12, cr * cr, 3))).astype(np.float32)
    return x, v


@app.function(image=image, gpu="A10G", cpu=8.0, volumes={"/data": vol}, timeout=3600)
def run(ckpt_rel: str) -> None:
    import sys, io, contextlib, time
    sys.path.insert(0, REMOTE)
    from pathlib import Path
    import torch
    from src.data import assemble_edge_features, assemble_node_features_for_gnn
    from src.hybrid import HybridRollout
    from scripts.eval_rollout import _load_ckpt

    dev = "cuda"
    torch.backends.cudnn.benchmark = True

    def timeit(fn, n_warm=30, n_time=200):
        with torch.no_grad():
            for _ in range(n_warm):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_time):
                fn()
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_time * 1e3  # ms

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print("CUDA:", torch.cuda.get_device_name(0))
        model, stats, mk, incF, C = _load_ckpt(Path(f"/data/{ckpt_rel}"))
        model = model.to(dev).eval()
        fm = stats.feat_mean.to(dev); fs = stats.feat_std.to(dev)
        em = stats.edge_mean.to(dev) if stats.edge_mean is not None else None
        es = stats.edge_std.to(dev) if stats.edge_std is not None else None
        for cr in (64, 128):
            x, v = _synth_state(cr)
            ro = HybridRollout(model, mk, stats, dt=1e-3, detector=None,
                               threshold=float("inf"), fallback_sim=None,
                               history_C=C, include_F=incF, device=dev)
            ro.reset(x[C - 1], v[C - 1], grid_x=cr, grid_y=cr,
                     v_history=v[0:C])
            # (a) full _neural_accel (assembly + model + integration bits)
            t_full = timeit(ro._neural_accel)
            # build model inputs once (stats cached on device)
            vh = torch.stack(list(ro._v_history), dim=1)
            node = assemble_node_features_for_gnn(ro.x, vh, ro.F, ro.f_ext,
                                                  mean_center=True, include_F=incF)
            edge = assemble_edge_features(ro.x, ro.edge_index)
            node_n = ((node - fm) / fs).contiguous()
            edge_n = ((edge - em) / es).contiguous() if em is not None else edge.contiguous()
            ei = ro.edge_index
            # (b) model forward only (fixed inputs)
            t_model = timeit(lambda: model(node_n, edge_n, ei))
            # (c) torch.compile (CUDA graphs)
            try:
                cmodel = torch.compile(model, mode="reduce-overhead")
                t_compile = timeit(lambda: cmodel(node_n, edge_n, ei), n_warm=40)
            except Exception as e:
                t_compile = float("nan"); print("compile failed:", repr(e)[:120])
            # (d) fp16 model forward
            try:
                mh = model.half()
                nh, eh = node_n.half(), edge_n.half()
                t_half = timeit(lambda: mh(nh, eh, ei))
            except Exception as e:
                t_half = float("nan"); print("half failed:", repr(e)[:120])
            # (e) fp16 + torch.compile
            try:
                cmh = torch.compile(mh, mode="reduce-overhead")
                t_half_compile = timeit(lambda: cmh(nh, eh, ei), n_warm=40)
            except Exception as e:
                t_half_compile = float("nan"); print("half+compile failed:", repr(e)[:120])
            model.float()  # restore for next size
            print(f"\n[cloth {cr}x{cr} = {cr*cr} particles]")
            print(f"  full _neural_accel : {t_full:6.2f} ms")
            print(f"  model forward only : {t_model:6.2f} ms  (assembly overhead = {t_full-t_model:.2f} ms)")
            print(f"  + torch.compile    : {t_compile:6.2f} ms")
            print(f"  + fp16             : {t_half:6.2f} ms")
            print(f"  + fp16 + compile   : {t_half_compile:6.2f} ms")

        # ---- fp16 inference accuracy vs fp32 on real held-out states ----
        import numpy as np, pandas as pd
        df = pd.read_csv(f"{REMOTE}/index_heldout_npz.csv")
        df = df[df["status"] == "OK"]
        tm = stats.target_mean.to(dev); ts = stats.target_std.to(dev)
        print("\n=== fp16 vs fp32 accel relative error (real held-out states) ===")
        model.float()
        for scen in ("drape", "collision"):
            r = df[df.scenario == scen].iloc[0]
            clip = np.load(r["path"], allow_pickle=True)
            meta = clip["meta"].item(); gx, gy = meta["grid"]
            ro = HybridRollout(model, mk, stats, dt=1e-3, detector=None,
                               threshold=float("inf"), fallback_sim=None,
                               history_C=C, include_F=incF, device=dev)
            xr, vr = clip["x"], clip["v"]; s0 = C - 1
            rel_errs = []
            with torch.no_grad():
                for f in range(s0, min(s0 + 60, xr.shape[0] - 1), 3):
                    ro.reset(xr[f], vr[f], grid_x=int(gx), grid_y=int(gy),
                             v_history=vr[f - C + 1:f + 1])
                    vh = torch.stack(list(ro._v_history), dim=1)
                    nd = assemble_node_features_for_gnn(ro.x, vh, ro.F, ro.f_ext,
                                                        mean_center=True, include_F=incF)
                    ed = assemble_edge_features(ro.x, ro.edge_index)
                    nn = ((nd - fm) / fs); ee = ((ed - em) / es) if em is not None else ed
                    ei2 = ro.edge_index
                    a32 = model(nn, ee, ei2) * ts + tm
                    mh = model.half()
                    a16 = (mh(nn.half(), ee.half(), ei2).float()) * ts + tm
                    model.float()
                    rel_errs.append(float((a16 - a32).norm() / (a32.norm() + 1e-8)))
            print(f"  {scen:>9}: mean fp16 accel rel-err = {np.mean(rel_errs)*100:.3f}%  "
                  f"max = {np.max(rel_errs)*100:.3f}%")
    out = buf.getvalue()
    print(out, flush=True)
    Path(OUT).mkdir(parents=True, exist_ok=True)
    Path(f"{OUT}/neural_opt.txt").write_text(out)
    vol.commit()
    print("== wrote neural_opt.txt ==", flush=True)


@app.local_entrypoint()
def main(ckpt_rel: str = "results/checkpoints/gnn_noF_b3k_pf5/best.pt"):
    call = run.spawn(ckpt_rel)
    print(f"spawned neural-opt probe (call {call.object_id})")
