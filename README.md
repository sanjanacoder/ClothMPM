# Hybrid Neural–MPM for Cloth

A learned surrogate for real-time cloth simulation, with a physics safety net. A
small graph neural network replaces the expensive grid-update step of a Material
Point Method (MLS-MPM) cloth solver; a lightweight **complexity detector** watches
the rollout and routes hard frames to an implicit mass-spring **fallback**. The
reference simulator (Taichi MLS-MPM), the neural surrogate, the detector, the GPU
fallback, and the full evaluation/experiment harness all live in this repo.

> Reference solver: **Taichi MLS-MPM** thin-shell cloth · Surrogate: **MeshGraphNets-style GNN** (~847k params) · Fallback: **implicit mass-spring** (CPU + a matrix-free GPU port) · Scenarios: **drape** and **collision** (wind deferred).

---

## Headline results (held-out clips)

| Research question | Result |
|---|---|
| **RQ1 — surrogate accuracy** | Pushforward-trained GNN; rollout energy drift **0.017** on contact (collision), stable rollouts; one-step val error 0.405 m/s². |
| **RQ2 / H2 — detector** | A position-only **strain proxy** separates hard (drift) frames at **AUROC 0.98** — deployment-viable (no deformation-gradient F needed). |
| **RQ2 / H3 — hybrid speed** | With fp16 + `torch.compile` on the net and a matrix-free GPU fallback, the hybrid is **1.8–2.0× faster than full-MPM**, measured end-to-end. |
| **RQ2 / H3 — hybrid accuracy** | **+30.8%** lower drape drift than pure-neural on held-out clips (hysteresis tuned on a disjoint split), matching the physics fallback at half the physics. |
| **Handoff finding** | Naive per-frame neural↔fallback switching is *worse than either solver alone*; the surrogate and fallback occupy **different manifolds**. Neither fine-tuning nor blending closes the gap — **minimizing transitions (hysteresis) is the practical mitigation.** |

See `docs/` (roadmap, scope, eval-plan) and `experiments/README.md` for details.

---

## Approach

Full-MPM is accurate but slow (it advances in many small substeps for stability).
The surrogate predicts per-particle acceleration from `(position, velocity history)`
and takes one large stable step — but a pure-neural rollout eventually drifts. The
hybrid runs the net by default and, when the detector's complexity signal crosses a
threshold, hands those frames to a stable implicit solver, then hands back. The
central research finding is that this handoff is not free: the two solvers produce
different dynamics, so the useful design lever is *how often you switch*, not how
you blend.

```
            detector (strain proxy)
                   │  score > threshold?
   state ──────────┼─────────────────────────► next state
        │          │                        ▲
        ├── no ──►  neural GNN (fast) ───────┤
        └── yes ─►  implicit fallback  ──────┘
                   (hysteresis: stay engaged to minimize transitions)
```

---

## Repository layout

```
src/
  mpm_cloth.py           Taichi MLS-MPM reference cloth solver (thin-shell membrane)
  cloth_implicit.py      Implicit mass-spring fallback (CPU, SciPy modified-PCG)
  cloth_implicit_gpu.py  Matrix-free GPU port of the fallback (torch, CG)
  neural_solver.py       MLP + GNN surrogate architectures
  train.py               Training (one-step + pushforward/short-unroll)
  data.py                Trajectory dataset, feature assembly, normalization
  detector.py            Complexity detectors (cosine D1, logreg D2, MLP D3, strain proxy)
  hybrid.py              HybridRollout: detector-gated neural/fallback rollout
  eval.py                Rollout metrics (L2, horizon, energy drift, AUROC helpers)
  contact.py             Shared collider geometry (sphere / box / ground)
scripts/
  generate_dataset.py        Local dataset generator (.npz clips)
  generate_dataset_modal.py  Parallel GPU dataset generation on Modal
  train_modal.py             Training on a Modal GPU (from an mmap store)
  npz_to_mmap.py             Convert .npz clips to a memory-mappable store
  eval_rollout.py            Pure-neural rollout evaluation (corrected seeding)
  eval_detector.py           H2 detector-separability (AUROC per feature)
  eval_hybrid.py             Neural vs fallback vs hybrid rollout
  benchmark_gpu.py           Neural vs Taichi-MPM vs fallback per-step timing
experiments/           Modal driver scripts for the paper experiments (+ manifests)
configs/               mpm.yaml, mlp.yaml, gnn.yaml, hybrid.yaml, eval.yaml
tests/                 84 pytest tests (run on CPU, no GPU needed)
docs/                  roadmap, scope, eval-plan, stack notes, findings
```

---

## Setup

Python **3.10 or 3.11** (Taichi 1.7 does not support 3.12+).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Verify (CPU only, no GPU required):

```bash
pytest tests/ -v        # 84 tests
```

### GPU work (training, full datagen, benchmarks) uses [Modal](https://modal.com)

Training, large-scale data generation, and all GPU benchmarks/experiments run on
Modal (serverless A10G GPUs), writing to a persistent volume
`cloth-mpm-trajectories` (trained checkpoints + `.npz` clips). One-time:

```bash
pip install modal
modal token new           # authenticate
```

The reference simulator and the implicit fallback also run on CPU locally (used by
the tests and small smoke runs); only the neural training / GPU timing need Modal.

---

## Quickstart

```bash
# 1. environment + tests
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v

# 2. generate a few clips locally (CPU smoke)
python scripts/generate_dataset.py --config configs/mpm.yaml \
    --out data/cloth_trajectories --smoke

# 3. (GPU) generate the full dataset on Modal
modal run scripts/generate_dataset_modal.py

# 4. (GPU) train the GNN surrogate with the pushforward curriculum
modal run scripts/train_modal.py --config configs/gnn.yaml \
    --mmap-subdir balanced3k_mmap --window 5 --epochs 8 --batch-size 32 --gpu A10G

# 5. evaluate a trained checkpoint (pure-neural rollout, corrected seeding)
python scripts/eval_rollout.py --manifest data/cloth_trajectories/index.csv \
    --ckpt gnn=results/checkpoints/gnn_noF_b3k_pf5/best.pt --n-clips 5 --n-steps 400
```

---

## Running the pipeline

**1 · Data generation.** Each clip is one MLS-MPM cloth trajectory saved as `.npz`
(`x, v, a, F, contact_flag, meta`). Locally with `scripts/generate_dataset.py`
(`--smoke` for 6 clips); at scale with `scripts/generate_dataset_modal.py`
(fire-and-forget, resumable). Convert to a memory-mappable store for fast training
with `scripts/npz_to_mmap.py` (or the `prepare_mmap` step in `train_modal.py`).

**2 · Training.** `scripts/train_modal.py` runs `src.train` on a Modal GPU. Key
flags: `--config configs/gnn.yaml` (or `mlp.yaml`), `--window K` (pushforward
unroll horizon; the curriculum ramps to K), `--max-train-frames` (budget cap),
`--noise-sigma` (GNS-style train noise), `--manifest-override` (train a specific
on-volume manifest, e.g. a seed-holdout split). Checkpoints stream to the volume as
validation improves.

**3 · Evaluation.**
- `scripts/eval_rollout.py` — pure-neural rollout: horizon, positional L2, energy drift (peak-normalized).
- `scripts/eval_detector.py` — H2 detector separability: per-frame drift labels, AUROC per feature.
- `scripts/eval_hybrid.py` — neural / fallback / hybrid rollout comparison.
- `scripts/benchmark_gpu.py` — fair per-frame timing (MPM = 10 substeps vs neural = 1 step).

**4 · Experiments (paper).** `experiments/` holds the Modal drivers for the speed
benchmarks, end-to-end hybrid wall-clock + accuracy, the threshold×hysteresis sweep
(with a held-out knob-selection split), the **handoff ablation** (matched budget,
transitions vs error + per-transition solver disagreement), the blended-handoff
test, and the DAgger fine-tuning pipeline. Each is `modal run experiments/<name>.py`.
See `experiments/README.md`; `experiments/manifests/` pins the exact clip splits.

```bash
modal run experiments/gpu_benchmark.py        # neural vs MPM vs fallback timing
modal run experiments/hybrid_inloop.py        # end-to-end hybrid wall-clock + accuracy
modal run experiments/handoff_ablation.py     # the handoff-effect ablation
```

---

## Key design notes

- **No deformation gradient F in the surrogate.** F is codimensional for thin-shell
  cloth and its determinant drifts, so the net uses only `(x, v)`. (This is also why
  the detector needed a *position-based* strain proxy — see H2.)
- **Pushforward training.** The net is unrolled K steps on its own predictions and
  backpropagated through the integration (curriculum ramps K→5), which fixes
  rollout error compounding. Backprop is done **per clip** so peak memory is `1×K`,
  not `batch×K`.
- **Speed.** fp16 is essentially lossless for inference (0.07% acceleration error);
  `torch.compile` (CUDA graphs) plus a matrix-free GPU fallback are what make the
  hybrid faster than the (already well-optimized) Taichi MPM at these scales.
- **Fair timing.** The surrogate advances one `dt = 1e-3` frame per forward; MPM
  needs 10 substeps of `dt = 1e-4` for the same physical time — all comparisons use
  this per-frame basis, with warmup and device synchronization.

---

## Limitations & honest caveats

- Speedup is **~2×**, not the ambitious 3–5×; results are on **drape + collision** at
  64×64 (wind deferred; larger resolutions timed but not accuracy-validated).
- The hybrid's accuracy gain requires **hysteresis**; naive per-frame switching hurts.
- The fallback is an *approximate* mass-spring model, so "matches physics" means
  matching that fallback, not the MPM reference exactly.
- The neural surrogate and the physics fallback occupy **different dynamics
  manifolds** — a fundamental limitation that neither fine-tuning nor blending
  removes (see the handoff ablation in `experiments/handoff_ablation.py` /
  `experiments/blended_handoff.py`).

## Testing

```bash
pytest tests/ -v      # 84 tests, CPU-only: simulator, fallback, data, detector,
                      #                     neural solver, hybrid, eval, contact
```
