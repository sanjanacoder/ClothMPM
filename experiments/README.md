# Experiments (Modal / A10G)

Driver scripts for the hybrid neural–MPM cloth evaluation. Each is a
[Modal](https://modal.com) app that runs on an A10G GPU against the
`cloth-mpm-trajectories` volume (trained checkpoints + `.npz` clips live there).
Run with e.g. `modal run experiments/<script>.py`. They import the reusable code
from `src/` and `scripts/`; only the orchestration lives here.

`manifests/` holds the clip splits used (paths point at the volume's `.npz`):

| manifest | clips | use |
|---|---|---|
| `index_heldout_npz.csv` | 5 drape + 5 collision (seeds …1495–1499) | the original small held-out set (speed benchmarks) |
| `index_heldout_large.csv` | 35 drape + 35 collision (seeds …1525–1559) | the larger held-out set (accuracy / ablations) |
| `index_dagger_npz.csv` | 80 drape (seeds 1200–1279) | DAgger state-collection (training-domain, disjoint from held-out) |
| `index_ft_orig_npz.csv` | 40 drape + 40 collision | original data for fine-tuning (anti-forgetting) |

## Speed (RQ2 / H3)
- `gpu_benchmark.py` — neural vs Taichi-MPM vs implicit fallback, fair per-frame (MPM = 10 substeps), warmup + device sync.
- `gpu_neural_opt.py` — neural inference optimization: fp32 → torch.compile → fp16 → fp16+compile; plus fp16-vs-fp32 accuracy.
- `gpu_scale_probe.py` — neural-vs-MPM as cloth resolution + MPM grid scale together.
- `gpu_fallback_bench.py` — GPU implicit fallback timing (eager / fixed-iter / +torch.compile) and hybrid effective per-frame.

## Hybrid accuracy + wall-clock (RQ2 / H3)
- `hybrid_inloop.py` — end-to-end GPU hybrid rollout: 70-clip accuracy (fp32) + real wall-clock (fp16), scenario-aware routing.
- `hybrid_sweep.py` — threshold × hysteresis sweep with a **held-out knob-selection split** (tune on one half, report on the disjoint half).

## Handoff effect (the mechanism result)
- `handoff_ablation.py` — matched fallback budget, vary #solver transitions (naive / hysteresis / min-burst); reports switches, budget, error-over-time, final error, and per-transition neural-vs-fallback acceleration disagreement + decay.
- `blended_handoff.py` — option (b): ramp a fallback weight over K frames and blend next-states; hard vs blended at matched budget.

## Fine-tuning (DAgger — obstructed, kept for the record)
`dagger_collect.py` → `dagger_targets.py` → `dagger_finetune.py`. Collect off-manifold
(handoff/fallback) states, label them, fine-tune. Outcome: MPM targets on these
states are off-manifold/unstable and fallback-accel targets are distributionally
incompatible — i.e. the neural (MPM) and fallback (mass-spring) occupy different
manifolds, so neither fine-tuning (a) nor blending (b) closes the handoff gap;
minimizing transitions (hysteresis) is the practical mitigation.
