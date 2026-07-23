"""DAgger stage 3: fine-tune the surrogate on the off-manifold (handoff/fallback)
states with their MPM targets, mixed with original training frames (to avoid
forgetting). One-step normalized-accel MSE. Saves a new checkpoint."""
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/root/mpm"
SCRATCH = str(Path(__file__).resolve().parent / "manifests")
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-runtime-ubuntu22.04", add_python="3.11")
    .pip_install(["torch>=2.6", "numpy>=1.26,<2.2", "pandas>=2.1", "pyyaml>=6.0"])
    .add_local_dir(str(ROOT / "src"), f"{REMOTE}/src", ignore=["**/__pycache__", "**/*.pyc"])
    .add_local_dir(str(ROOT / "scripts"), f"{REMOTE}/scripts", ignore=["**/__pycache__", "**/*.pyc"])
    .add_local_dir(str(ROOT / "configs"), f"{REMOTE}/configs")
    .add_local_file(f"{SCRATCH}/index_ft_orig_npz.csv", f"{REMOTE}/index_ft_orig_npz.csv")
)
vol = modal.Volume.from_name("cloth-mpm-trajectories")
app = modal.App("cloth-mpm-dagger3")
DAG = "/data/results/dagger"


@app.function(image=image, gpu="A10G", cpu=8.0, volumes={"/data": vol}, timeout=10800)
def run(ckpt_rel: str, out_name: str, epochs: int, lr: float, orig_per_clip: int) -> None:
    import sys
    sys.path.insert(0, REMOTE)
    from pathlib import Path
    import numpy as np, pandas as pd, torch
    from torch import nn
    from src.data import (assemble_edge_features, assemble_node_features_for_gnn,
                          build_mesh_edge_index, NormalizationStats)
    from src.neural_solver import build_solver

    dev = "cuda"
    ckpt = torch.load(f"/data/{ckpt_rel}", map_location="cpu", weights_only=False)
    stats = NormalizationStats.from_dict(ckpt["stats"])
    model = build_solver(ckpt["model_cfg"]).to(dev).train()
    model.load_state_dict(ckpt["model"])
    incF = bool(ckpt.get("include_F", True)); C = 5
    fm = stats.feat_mean.to(dev); fs = stats.feat_std.to(dev)
    em = stats.edge_mean.to(dev) if stats.edge_mean is not None else None
    es = stats.edge_std.to(dev) if stats.edge_std is not None else None
    tm = stats.target_mean.to(dev); ts = stats.target_std.to(dev)
    ei = build_mesh_edge_index(64, 64).to(dev)
    N = 64 * 64
    F_id = torch.eye(3, device=dev).expand(N, 3, 3).contiguous()
    f_ext = torch.zeros(N, 3, device=dev)

    # augmentation samples (off-manifold states + fallback targets).
    # NOTE: read each npz array ONCE into a local -- indexing npz["x"][i] inside a
    # loop re-decompresses the whole array every time (a ~750MB stall).
    s = np.load(f"{DAG}/states.npz"); t = np.load(f"{DAG}/targets.npz")
    SX = np.asarray(s["x"]); SVH = np.asarray(s["vhist"]); TA = np.asarray(t["a"])
    aug = [("aug", SX[i], SVH[i], TA[i]) for i in range(SX.shape[0])]
    # original samples
    df = pd.read_csv(f"{REMOTE}/index_ft_orig_npz.csv"); df = df[df.status == "OK"]
    rng = np.random.default_rng(0); orig = []
    for _, r in df.iterrows():
        c = np.load(r["path"], allow_pickle=True)
        x = np.asarray(c["x"]); v = np.asarray(c["v"]); a = np.asarray(c["a"]); T = x.shape[0]
        for ti in rng.integers(C - 1, T - 1, size=orig_per_clip):
            vh = np.transpose(v[ti - C + 1:ti + 1], (1, 0, 2))   # (N,C,3)
            orig.append(("orig", x[ti], vh, a[ti]))
    print(f"aug={len(aug)} orig={len(orig)}", flush=True)
    data = aug + orig

    def loss_on(x_np, vh_np, a_np):
        x = torch.as_tensor(x_np, dtype=torch.float32, device=dev)
        vh = torch.as_tensor(vh_np, dtype=torch.float32, device=dev)
        node = assemble_node_features_for_gnn(x, vh, F_id, f_ext, mean_center=True, include_F=incF)
        edge = assemble_edge_features(x, ei)
        nn_ = (node - fm) / fs
        ee = ((edge - em) / es) if em is not None else edge
        pred = model(nn_, ee, ei)
        tgt = (torch.as_tensor(a_np, dtype=torch.float32, device=dev) - tm) / ts
        return nn.functional.mse_loss(pred, tgt)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    accum = 8
    for ep in range(epochs):
        rng.shuffle(data)
        losses = []; opt.zero_grad()
        for i, (_, x_np, vh_np, a_np) in enumerate(data):
            l = loss_on(x_np, vh_np, a_np) / accum
            l.backward(); losses.append(float(l) * accum)
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
        # quick val: mean loss on aug vs orig separately
        model.eval()
        with torch.no_grad():
            la = np.mean([float(loss_on(x, vh, a)) for _, x, vh, a in aug[:200]])
            lo = np.mean([float(loss_on(x, vh, a)) for _, x, vh, a in orig[:200]])
        model.train()
        print(f"[ep {ep}] train_mse={np.mean(losses):.4f}  aug_mse={la:.4f}  orig_mse={lo:.4f}", flush=True)

    out = {"model": model.state_dict(), "stats": ckpt["stats"],
           "model_cfg": ckpt["model_cfg"], "include_F": incF}
    p = f"/data/results/checkpoints/{out_name}/best.pt"
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, p); vol.commit()
    print(f"== saved fine-tuned checkpoint -> {p} ==", flush=True)


@app.local_entrypoint()
def main(ckpt_rel: str = "results/checkpoints/gnn_noF_b3k_pf5/best.pt",
         out_name: str = "gnn_noF_b3k_pf5_ft", epochs: int = 8, lr: float = 5e-5,
         orig_per_clip: int = 30):
    call = run.spawn(ckpt_rel, out_name, epochs, lr, orig_per_clip)
    print(f"spawned dagger finetune (call {call.object_id})")
