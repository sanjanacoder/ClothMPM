"""M4/RQ2 (H2) detector separability: do per-frame complexity features predict
the frames where the pure-neural rollout drifts past epsilon?

For each clip we run the trained model as a pure-neural rollout, record per-frame
positional L2 vs the MPM reference, label y_t = 1[L2(t) > eps], and score each of
the four detector features (cos_sim_window, strain_rate, contact_frac,
edge_len_change_max) plus D1 (1 - cos_sim). We report AUROC per feature/detector,
overall and per scenario, so H2 (AUROC >= 0.8) can be judged directly.

Note: features are computed on the *reference* clip (F, contact available) -- an
offline separability study. A deployment detector during pure-neural rollout only
sees neural-state features (cos_sim of predicted accel, edge-length change); F and
contact_flag are simulator outputs. That gap is called out in the report.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_kring_index
from src.detector import (compute_detector_features, strain_rate_from_positions,
                          warmup_mask)
from src.eval import per_particle_l2
from src.hybrid import HybridRollout
from scripts.eval_rollout import _NoFallbackDetector, _load_ckpt

# strain_rate (from F) is offline-only; strain_proxy (from positions) is the
# deployment-available stand-in we want to validate against it.
FEATS = ["cos_sim", "strain_rate", "strain_proxy", "contact_frac", "edge_len_change"]


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney U AUROC with tie-averaged ranks. nan if one class absent."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = scores.argsort(kind="mergesort")
    s_sorted = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0  # average rank (1-based)
        i = j + 1
    r_pos = ranks[labels == 1].sum()
    return float((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


@torch.no_grad()
def collect_per_frame(ckpt_path: Path, manifest: Path, eps: float,
                      n_steps: int, cosine_steps: int = 10) -> pd.DataFrame:
    model, stats, model_kind, include_F, C = _load_ckpt(ckpt_path)
    df = pd.read_csv(manifest)
    df = df[df["status"] == "OK"].reset_index(drop=True)
    rows = []
    for _, r in df.iterrows():
        clip = np.load(ROOT / r["path"], allow_pickle=True)
        meta = clip["meta"].item()
        gx, gy = meta["grid"]
        dt = float(meta["dt_s"]) * int(meta["log_every_substeps"])
        xref, vref = clip["x"], clip["v"]
        s0 = C - 1
        steps = min(n_steps, xref.shape[0] - 1 - s0)

        ro = HybridRollout(model, model_kind, stats, dt=dt,
                           detector=_NoFallbackDetector(), threshold=float("inf"),
                           fallback_sim=None, history_C=C, include_F=include_F,
                           device="cpu")
        ro.reset(xref[s0], vref[s0], grid_x=int(gx), grid_y=int(gy),
                 v_history=vref[s0 - C + 1: s0 + 1])
        out = ro.rollout(steps, log_every=1)
        pred_x = out["x"]
        l2 = np.array([per_particle_l2(pred_x[k], xref[s0 + 1 + k]) for k in range(steps)])

        ei = build_kring_index(int(gx), int(gy), 1).numpy()
        clipd = {"x": clip["x"], "a": clip["a"], "F": clip["F"],
                 "contact_flag": clip["contact_flag"]}
        feats = compute_detector_features(clipd, edge_index=ei,
                                          cosine_window_steps=cosine_steps)
        # Deployment-available strain proxy from positions (no F). We score it on
        # the REFERENCE positions here for a like-for-like separability compare;
        # at deployment it would run on the predicted positions.
        strain_proxy = strain_rate_from_positions(clip["x"], ei)
        wm = warmup_mask(feats, cosine_steps)
        for k in range(steps):
            t = s0 + 1 + k
            if not wm[t]:
                continue
            rows.append({
                "scenario": r["scenario"], "seed": int(r["seed"]),
                "label": int(l2[k] > eps),
                "cos_sim": float(feats[t, 0]), "strain_rate": float(feats[t, 1]),
                "strain_proxy": float(strain_proxy[t]),
                "contact_frac": float(feats[t, 2]), "edge_len_change": float(feats[t, 3]),
                "l2": float(l2[k]),
            })
        print(f"  {r['scenario']:>9} seed={int(r['seed']):>6}  "
              f"l2_final={l2[-1]:.3f}  pos_frac={np.mean(l2[C:] > eps):.2f}", flush=True)
    return pd.DataFrame(rows)


def report(pf: pd.DataFrame, eps: float) -> None:
    def block(sub: pd.DataFrame, tag: str) -> None:
        n = len(sub)
        pos = int(sub["label"].sum())
        print(f"\n[{tag}] frames={n}  positives(L2>{eps})={pos} ({100*pos/max(1,n):.1f}%)")
        if pos == 0 or pos == n:
            print("  single-class -> AUROC undefined (model rarely/always exceeds eps here)")
            return
        print(f"  D1 (1-cos_sim)       AUROC={auroc(1 - sub['cos_sim'].values, sub['label'].values):.3f}")
        for f in FEATS:
            print(f"  feat {f:18s} AUROC={auroc(sub[f].values, sub['label'].values):.3f}")

    block(pf, "ALL")
    for scen, g in pf.groupby("scenario"):
        block(g, scen)


# Deployment-available feature set: strain_proxy (positions, not F), cosine of
# predicted accel, geometric contact fraction, and edge-length change. Excludes
# the F-based strain_rate, which a no-F rollout cannot produce.
DEPLOY_FEATS = ["strain_proxy", "cos_sim", "contact_frac", "edge_len_change"]


def train_eval_detectors(train_pf: pd.DataFrame, eval_pf: pd.DataFrame,
                         feature_cols: list[str] = DEPLOY_FEATS) -> None:
    """Train D2 (logreg) + D3 (tiny MLP) on the train split's deployment features
    and report AUROC on the held-out eval split (pooled + per scenario), against
    the D1 cosine baseline and the single-feature strain_proxy baseline."""
    from src.detector import LogRegDetector, TinyMLPDetector

    Xtr = train_pf[feature_cols].values.astype(np.float64)
    ytr = train_pf["label"].values.astype(int)
    print(f"\ntrain frames={len(ytr)} positives={int(ytr.sum())} "
          f"({100*ytr.mean():.1f}%)  features={feature_cols}")

    d2 = LogRegDetector().fit(Xtr, ytr)
    d3 = TinyMLPDetector(n_features=len(feature_cols), device="cpu").fit(
        Xtr.astype(np.float32), ytr)

    scorers = {
        "D1 (1-cos_sim)  ": lambda pf: 1.0 - pf["cos_sim"].values,
        "strain_proxy    ": lambda pf: pf["strain_proxy"].values,
        "D2 logreg       ": lambda pf: d2.score(pf[feature_cols].values.astype(np.float64)),
        "D3 tiny-MLP      ": lambda pf: d3.score(pf[feature_cols].values.astype(np.float32)),
    }

    def block(sub: pd.DataFrame, tag: str) -> None:
        y = sub["label"].values
        print(f"\n[EVAL {tag}] frames={len(sub)} positives={int(y.sum())} "
              f"({100*y.mean():.1f}%)")
        if y.sum() == 0 or y.sum() == len(y):
            print("  single-class -> AUROC undefined")
            return
        for name, fn in scorers.items():
            print(f"  {name} AUROC={auroc(fn(sub), y):.3f}")

    block(eval_pf, "ALL (pooled)")
    for scen, g in eval_pf.groupby("scenario"):
        block(g, scen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--eps", type=float, default=0.10)
    ap.add_argument("--n-steps", type=int, default=400)
    ap.add_argument("--out", default=str(ROOT / "results" / "detector_separability.csv"))
    args = ap.parse_args()
    pf = collect_per_frame(Path(args.ckpt), Path(args.manifest), args.eps, args.n_steps)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pf.to_csv(args.out, index=False)
    report(pf, args.eps)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
