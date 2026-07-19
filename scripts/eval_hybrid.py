"""M4/RQ2 (H3) hybrid rollout eval: neural-only vs fallback-only vs hybrid.

The pivotal question first: does the implicit mass-spring fallback (F2) track the
MPM reference better than the pure-neural rollout on the frames where neural
drifts? If not, routing hard frames to it cannot help. For each clip we run:
  - neural-only   (threshold = +inf; fallback never fires)
  - fallback-only (threshold = -inf; fallback always fires post-warmup)
  - hybrid        (a per-scenario strain-proxy threshold)
and report per-frame L2 vs the MPM reference, fallback fraction, and mean
per-step wall-time for each path.

The fallback sim is configured per clip from the clip meta (grid, sphere for
collision, pinned corners, initial height).
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cloth_implicit import ImplicitClothSim, load_implicit_config
from src.detector import strain_rate_from_positions
from src.eval import per_particle_l2
from src.hybrid import HybridRollout
from scripts.eval_rollout import _load_ckpt


class _StrainProxyDetector:
    """Fires on the deployment strain proxy computed from the rollout's own
    recent positions (rest = the first logged frame). score() is called by
    HybridRollout with the (1,4) feature row, which we ignore; we pull positions
    from the bound rollout instead."""
    def __init__(self, rollout: HybridRollout):
        self.ro = rollout
        self._rest = None

    def bind_rest(self, rest: np.ndarray):
        self._rest = np.asarray(rest, dtype=np.float64)

    def score(self, feat_row: np.ndarray) -> np.ndarray:
        xh = self.ro._x_history
        if self._rest is None or len(xh) < 2:
            return np.zeros(1, dtype=np.float32)
        x_prev = xh[-2].detach().cpu().numpy()
        x_now = xh[-1].detach().cpu().numpy()
        seq = np.stack([self._rest, x_prev, x_now])  # rest as frame0 anchor
        ei = self.ro.edge_index.cpu().numpy()
        sp = strain_rate_from_positions(seq, ei, rest=self._rest)
        return np.array([sp[-1]], dtype=np.float32)


def _fallback_for_clip(meta: dict) -> ImplicitClothSim:
    cfg = copy.deepcopy(load_implicit_config())
    gx, gy = meta["grid"]
    cfg["cloth"]["grid"] = [int(gx), int(gy)]
    cfg["cloth"]["initial_height_m"] = float(meta.get("initial_height_m", 1.0))
    sphere = cfg["contact"]["primitives"]["sphere"]
    sc = meta.get("sphere_center_m")
    if sc is not None and float(meta.get("sphere_radius_m", 0.0)) > 0:
        sphere["enabled"] = True
        sphere["center_m"] = [float(v) for v in sc]
        sphere["radius_m"] = float(meta["sphere_radius_m"])
    else:
        sphere["enabled"] = False
    sim = ImplicitClothSim(cfg)
    sim.reset(pinned=[int(i) for i in meta.get("pinned_corner_indices", [])])
    return sim


@torch.no_grad()
def _run(ckpt_path: Path, clip, meta, mode: str, thr: float, n_steps: int):
    model, stats, model_kind, include_F, C = _load_ckpt(ckpt_path)
    gx, gy = meta["grid"]
    dt = float(meta["dt_s"]) * int(meta["log_every_substeps"])
    xref, vref = clip["x"], clip["v"]
    s0 = C - 1
    steps = min(n_steps, xref.shape[0] - 1 - s0)
    fb = _fallback_for_clip(meta) if mode != "neural" else None
    ro = HybridRollout(model, model_kind, stats, dt=dt,
                       detector=None, threshold=thr, fallback_sim=fb,
                       history_C=C, include_F=include_F, device="cpu")
    det = _StrainProxyDetector(ro)
    ro.detector = det
    ro.reset(xref[s0], vref[s0], grid_x=int(gx), grid_y=int(gy),
             v_history=vref[s0 - C + 1: s0 + 1])
    det.bind_rest(xref[0])
    t0 = time.perf_counter()
    out = ro.rollout(steps, log_every=1)
    wall = time.perf_counter() - t0
    pred_x = out["x"]
    l2 = np.array([per_particle_l2(pred_x[k], xref[s0 + 1 + k]) for k in range(steps)])
    return {
        "l2": l2, "fallback_fraction": out["fallback_fraction"],
        "ms_per_step": 1000.0 * wall / steps, "steps": steps,
        "finite": bool(np.isfinite(pred_x).all()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--n-steps", type=int, default=300)
    ap.add_argument("--drape-thr", type=float, default=0.02)
    ap.add_argument("--collision-thr", type=float, default=0.258)
    ap.add_argument("--out", default=str(ROOT / "results" / "hybrid_eval.csv"))
    args = ap.parse_args()
    df = pd.read_csv(args.manifest)
    df = df[df["status"] == "OK"].reset_index(drop=True)
    rows = []
    for _, r in df.iterrows():
        clip = np.load(ROOT / r["path"], allow_pickle=True)
        meta = clip["meta"].item()
        thr = args.collision_thr if r["scenario"] == "collision" else args.drape_thr
        for mode, t in [("neural", float("inf")), ("fallback", float("-inf")),
                        ("hybrid", thr)]:
            res = _run(Path(args.ckpt), clip, meta, mode, t, args.n_steps)
            rows.append({"scenario": r["scenario"], "seed": int(r["seed"]),
                         "mode": mode, "l2_final": round(float(res["l2"][-1]), 4),
                         "l2_at_200": round(float(res["l2"][min(199, res["steps"]-1)]), 4),
                         "fallback_frac": round(res["fallback_fraction"], 3),
                         "ms_per_step": round(res["ms_per_step"], 2),
                         "finite": res["finite"]})
            print(f"  {r['scenario']:>9} {int(r['seed']):>6} {mode:>8}: "
                  f"l2_final={rows[-1]['l2_final']}  fb={rows[-1]['fallback_frac']}  "
                  f"{rows[-1]['ms_per_step']}ms/step", flush=True)
    res = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    print("\n=== mean by scenario x mode ===")
    print(res.groupby(["scenario", "mode"]).agg(
        l2_final=("l2_final", "mean"), fb=("fallback_frac", "mean"),
        ms=("ms_per_step", "mean")).round(3).to_string())


if __name__ == "__main__":
    main()
