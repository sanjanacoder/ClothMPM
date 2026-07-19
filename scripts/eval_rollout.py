"""Pure-neural rollout evaluation for the pilot: does the surrogate work as a
*simulator*, not just a one-step predictor?

For each reference clip we take the initial state (x0, v0), run the trained
model autoregressively (HybridRollout in pure-neural mode -- threshold=inf so
the fallback never fires), and compare the predicted trajectory to the MPM
reference. One-step loss can look great while rollout drifts or blows up; this
measures the thing that actually matters.

Metrics per clip:
  - rollout_horizon: time (s) until per-particle L2 first exceeds tau (higher is
    better; capped at the rolled-out duration)
  - l2 @ frame k: positional drift at a few checkpoints
  - energy_drift: kinetic-energy ratio vs reference at the end
  - finite: whether the rollout stayed numerically stable

Usage:
  python scripts/eval_rollout.py --manifest data/fullres100/index.csv \
      --ckpt mlp=results/checkpoints/mlp_noF_pilot/best.pt \
      --ckpt gnn=results/checkpoints/gnn_noF_pilot/best.pt \
      --n-clips 3 --n-steps 400
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from src.data import NormalizationStats
from src.eval import (energy_drift, energy_drift_peak, kinetic_energy,
                       per_particle_l2, rollout_horizon)
from src.hybrid import HybridRollout
from src.neural_solver import build_solver


class _NoFallbackDetector:
    """Detector stub: always returns score 0 so, with threshold=inf, the
    rollout stays pure-neural and never touches the implicit fallback."""
    def score(self, feats: np.ndarray) -> np.ndarray:
        return np.zeros(len(feats), dtype=np.float32)


def _load_ckpt(path: Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    stats = NormalizationStats.from_dict(ckpt["stats"])
    model = build_solver(ckpt["model_cfg"]).eval()
    model.load_state_dict(ckpt["model"])
    model_kind = ckpt["model_cfg"]["type"]
    include_F = bool(ckpt.get("include_F", True))
    # C from feature width: 3(x) + 3C(v) + [9 F] + 3(f_ext)
    fdim = int(stats.feature_dim)
    C = (fdim - 6 - (9 if include_F else 0)) // 3
    return model, stats, model_kind, include_F, C


@torch.no_grad()
def eval_ckpt(name: str, ckpt_path: Path, manifest: Path,
              scenarios: list[str] | None, n_clips: int, n_steps: int,
              tau: float, mass_per: float) -> pd.DataFrame:
    model, stats, model_kind, include_F, C = _load_ckpt(ckpt_path)
    print(f"\n[{name}] {model_kind}  include_F={include_F}  C={C}  "
          f"feat_dim={stats.feature_dim}")

    df = pd.read_csv(manifest)
    df = df[df["status"] == "OK"].reset_index(drop=True)
    if scenarios:
        df = df[df["scenario"].isin(scenarios)]
    # A few clips per scenario present, deterministic pick
    picks = df.groupby("scenario").head(n_clips)

    rows = []
    for _, r in picks.iterrows():
        clip = np.load(ROOT / r["path"], allow_pickle=True)
        gx, gy = clip["meta"].item()["grid"]
        dt = float(clip["meta"].item()["dt_s"]) * int(clip["meta"].item()["log_every_substeps"])
        xref, vref = clip["x"], clip["v"]
        # Start at frame C-1 and seed the model's velocity history with the C
        # true prior velocities. The model infers acceleration from the velocity
        # *slope*, so a flat (v0-repeated) history yields wrong predictions and
        # a spurious rollout collapse -- this is the correct initialization.
        s0 = C - 1
        steps = min(n_steps, xref.shape[0] - 1 - s0)

        ro = HybridRollout(model, model_kind, stats, dt=dt,
                           detector=_NoFallbackDetector(), threshold=float("inf"),
                           fallback_sim=None, history_C=C, include_F=include_F,
                           device="cpu")
        ro.reset(xref[s0], vref[s0], grid_x=int(gx), grid_y=int(gy),
                 v_history=vref[s0 - C + 1: s0 + 1])
        out = ro.rollout(steps, log_every=1)
        pred_x, pred_v = out["x"], out["v"]                    # (steps, N, 3)

        # per-frame positional L2 (pred step k corresponds to ref frame s0+1+k)
        l2 = np.array([per_particle_l2(pred_x[k], xref[s0 + 1 + k]) for k in range(steps)])
        finite = bool(np.isfinite(pred_x).all())
        horizon = rollout_horizon(l2, tau, dt)
        # Kinetic-energy series over the rollout: final-frame drift plus a
        # peak-normalized drift (well-conditioned when the reference settles or
        # swings through a low-KE turning point, where the instantaneous ratio
        # blows up -- e.g. pinned drape).
        ke_p_series = np.array([kinetic_energy(pred_v[k], mass_per) for k in range(steps)])
        ke_r_series = np.array([kinetic_energy(vref[s0 + 1 + k], mass_per) for k in range(steps)])
        ke_p, ke_r = float(ke_p_series[-1]), float(ke_r_series[-1])
        ke_r_peak = float(ke_r_series.max())
        rows.append({
            "model": name, "scenario": r["scenario"], "seed": int(r["seed"]),
            "finite": finite,
            "horizon_s": round(horizon, 4),
            "l2_at_50": round(float(l2[min(49, steps - 1)]), 4),
            "l2_at_200": round(float(l2[min(199, steps - 1)]), 4),
            "l2_final": round(float(l2[-1]), 4),
            "energy_drift": round(energy_drift(ke_p, ke_r), 3),
            "energy_drift_peak": round(energy_drift_peak(ke_p, ke_r, ke_r_peak), 4),
            "ke_pred_final": round(ke_p, 6),
            "ke_ref_final": round(ke_r, 6),
            "ke_ref_peak": round(ke_r_peak, 6),
        })
        print(f"  {r['scenario']:>9} seed={int(r['seed']):>6}  "
              f"finite={finite}  horizon={horizon*1000:.0f}ms  "
              f"l2@200={rows[-1]['l2_at_200']}  l2_final={rows[-1]['l2_final']}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ckpt", action="append", required=True,
                    help="name=path/to/best.pt (repeatable for A-vs-B)")
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--n-clips", type=int, default=3,
                    help="clips per scenario")
    ap.add_argument("--n-steps", type=int, default=400)
    ap.add_argument("--tau", type=float, default=0.05,
                    help="drift threshold (m) for rollout horizon")
    ap.add_argument("--mass-per", type=float, default=0.2 / 4096,
                    help="per-particle mass for the energy metric")
    ap.add_argument("--out", default=str(ROOT / "results" / "pilot_rollout.csv"))
    args = ap.parse_args()

    all_df = []
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        all_df.append(eval_ckpt(name, Path(path), Path(args.manifest),
                                args.scenarios, args.n_clips, args.n_steps,
                                args.tau, args.mass_per))
    res = pd.concat(all_df, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)

    print("\n=== rollout summary (mean over clips) ===")
    summ = (res.groupby("model")
               .agg(finite_frac=("finite", "mean"),
                    horizon_s=("horizon_s", "mean"),
                    l2_at_200=("l2_at_200", "mean"),
                    l2_final=("l2_final", "mean"),
                    energy_drift=("energy_drift", "mean"),
                    energy_drift_peak=("energy_drift_peak", "mean"))
               .round(4))
    print(summ.to_string())
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
