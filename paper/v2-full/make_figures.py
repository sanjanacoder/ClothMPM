#!/usr/bin/env python3
"""
Generate the paper figures from the project's REAL reported results.

Every number here is transcribed from:
  - docs/handoff-ablation.md   (matched-budget ablation, error-over-time,
                                per-transition disagreement decay)
  - docs/results-summary.md    (detector AUROC per feature/scenario)

No synthetic or interpolated data: the curves are the exact checkpoints /
bins reported in those docs (markers sit on the measured values).
Outputs vector PDFs into ./figures/.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,   # embed TrueType (editable, avoids Type-3 warnings)
})

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

C_NAIVE = "#c1442e"
C_MID   = "#e08214"
C_STICKY= "#2b6a99"

# ---------------------------------------------------------------------------
# Figure 1 -- rollout error over time, by routing policy (matched ~48% budget)
# Source: handoff-ablation.md, "Error over time" table.
# ---------------------------------------------------------------------------
t   = [100, 200, 300, 400]
err_naive = [0.0006, 0.0090, 0.083, 0.185]
err_mb20  = [0.0006, 0.0090, 0.073, 0.161]
err_hyst  = [0.0006, 0.0090, 0.052, 0.130]

fig, ax = plt.subplots(figsize=(3.3, 2.5))
ax.plot(t, err_naive, "o-", color=C_NAIVE,  lw=1.6, ms=4, label="naive (22 switches)")
ax.plot(t, err_mb20,  "s-", color=C_MID,    lw=1.6, ms=4, label="min-burst-20 (14)")
ax.plot(t, err_hyst,  "^-", color=C_STICKY, lw=1.6, ms=4, label="hysteresis (5)")
ax.axvspan(0, 200, color="0.9", zorder=0)
ax.text(150, 0.028, "pure neural\n(no fallback fires yet)", ha="center", va="bottom",
        fontsize=6.3, color="0.4")
ax.set_xlabel("rollout step")
ax.set_ylabel(r"positional L2 error (m)")
ax.set_xlim(80, 410)
ax.set_ylim(0, 0.205)
ax.legend(frameon=False, fontsize=6.8, loc="upper left", bbox_to_anchor=(0.02, 1.02))
fig.savefig(os.path.join(OUT, "fig_error_over_time.pdf"))
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2 -- final error vs number of solver switches, at a MATCHED budget
# Source: handoff-ablation.md, main table (~48% fallback fraction).
# ---------------------------------------------------------------------------
policies = ["naive", "burst-5", "burst-20", "burst-50", "hysteresis"]
switches = [22.4, 20.8, 14.2, 6.9, 5.0]
l2final  = [0.185, 0.194, 0.161, 0.131, 0.124]
NEURAL_ONLY = 0.190  # pure-neural drape held-out baseline (results-summary.md)

fig, ax = plt.subplots(figsize=(3.3, 2.5))
ax.plot(switches, l2final, "-", color="0.6", lw=1.0, zorder=1)
ax.scatter(switches, l2final, c=C_STICKY, s=28, zorder=3)
# hand-tuned label offsets (pts) to avoid the naive/burst-5 cluster overlapping
label_off = {"naive": (6, -11), "burst-5": (6, 6), "burst-20": (5, 6),
             "burst-50": (5, 7), "hysteresis": (-4, -12)}
for x, y, p in zip(switches, l2final, policies):
    ax.annotate(p, (x, y), textcoords="offset points", xytext=label_off[p],
                fontsize=6.3, color="0.25",
                ha="right" if p == "hysteresis" else "left")
ax.axhline(NEURAL_ONLY, ls="--", color=C_NAIVE, lw=1.0)
ax.text(21.5, NEURAL_ONLY + 0.001, "pure neural", fontsize=6.5, color=C_NAIVE, va="bottom", ha="right")
ax.set_xlabel("solver switches per rollout")
ax.set_ylabel(r"final L2 error (m)")
ax.set_xlim(2, 25)
ax.set_ylim(0.11, 0.20)
ax.invert_xaxis()  # fewer switches -> right; shows "less switching, less error"
fig.savefig(os.path.join(OUT, "fig_switches_vs_error.pdf"))
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 3 -- the mechanism: solver disagreement decays after each handoff
# Source: handoff-ablation.md, "frames since switch" table.
# ---------------------------------------------------------------------------
bins   = ["0", "1–2", "3–5", "6–10", "11–30"]
xb     = range(len(bins))
dis_naive = [153, 173, 176, 175, 78]
dis_mb20  = [176, 186, 191, 189, 79]

fig, ax = plt.subplots(figsize=(3.3, 2.5))
ax.plot(xb, dis_naive, "o-", color=C_NAIVE,  lw=1.6, ms=4, label="naive")
ax.plot(xb, dis_mb20,  "s-", color=C_MID,    lw=1.6, ms=4, label="min-burst-20")
ax.axhline(78, ls=":", color="0.4", lw=1.0)
ax.text(4.0, 88, "steady-state", fontsize=6.5, color="0.4", ha="right")
ax.annotate("~10-frame\ntransient", (1.5, 185), fontsize=6.5, color="0.25", ha="center")
ax.set_xticks(list(xb)); ax.set_xticklabels(bins)
ax.set_xlabel("frames since last solver switch")
ax.set_ylabel("neural–fallback accel.\ndisagreement (m/s$^2$)")
ax.set_ylim(0, 210)
ax.legend(frameon=False, fontsize=6.8, loc="lower left")
fig.savefig(os.path.join(OUT, "fig_disagreement_decay.pdf"))
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 4 -- detector separability (AUROC) per feature, per scenario
# Source: results-summary.md, detector table.
# ---------------------------------------------------------------------------
feats = ["strain\nproxy", "edge-len\nchange", "strain-rate\n(needs F)",
         "cosine accel\n(prior default)", "contact\nfrac"]
drape = [0.980, 0.995, 0.99, 0.65, 0.50]
coll  = [0.985, 0.985, 0.98, 0.81, 0.50]
import numpy as np
x = np.arange(len(feats)); w = 0.38

fig, ax = plt.subplots(figsize=(3.5, 2.6))
b1 = ax.bar(x - w/2, drape, w, label="drape", color=C_STICKY)
b2 = ax.bar(x + w/2, coll,  w, label="collision", color=C_MID)
ax.axhline(0.8, ls="--", color="0.4", lw=1.0)
ax.text(4.4, 0.815, "H2 target (0.8)", fontsize=6.3, color="0.4", ha="right")
# mark the one we deploy
ax.text(0, 1.02, "deployed", ha="center", fontsize=6.3, color=C_STICKY, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(feats, fontsize=6.2)
ax.set_ylabel("AUROC (drift detection)")
ax.set_ylim(0.4, 1.06)
ax.legend(frameon=False, fontsize=6.8, loc="lower left")
fig.savefig(os.path.join(OUT, "fig_detector_auroc.pdf"))
plt.close(fig)

print("wrote:")
for f in sorted(os.listdir(OUT)):
    print("  figures/" + f)
