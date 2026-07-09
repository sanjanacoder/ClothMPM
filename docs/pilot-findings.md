# M3 pilot — MLP vs GNN on the 100-clip full-res batch

First real M3 experiment on production-resolution data (64×64 / 4096 particles),
after the F-out-of-scope decision. Trained both neural grid-update models with
`x, v → a` (no deformation gradient) and evaluated **rollout**, not just one-step
loss. See `scripts/eval_rollout.py`, results in `results/pilot_rollout.csv`.

## Training (100 clips: 50 drape + 50 collision)

| model | params | epochs | one-step val_accel_l2 | val_mse |
|---|---|---|---|---|
| MLP | 58 k | 10 (h→8) | 0.584 | 0.0028 |
| GNN | 847 k | 6 (h→2, hit 4h timeout) | **0.288** | **0.0006** |

The GNN is ~2× better one-step **despite** stopping at epoch 6 (its rollout
curriculum only reached horizon h=2, vs the MLP's full ramp to h=8).

## Rollout (pure-neural, 400 steps, 3 clips/scenario)

| metric | MLP | GNN |
|---|---|---|
| stable (finite) | ✅ | ✅ |
| rollout horizon @ 5 cm | 98 ms | 100 ms |
| positional L2 @ frame 200 | 0.24 m | 0.19 m |
| positional L2 @ frame 400 | 1.47 m | **0.70 m** |
| kinetic-energy drift (end) | **242×** | **0.99×** |

## Verdict

- **F-drop survives rollout for both** — neither model blows up over 400
  autoregressive steps. The mentor's decision holds beyond one-step.
- **The GNN is decisively better on multi-step fidelity.** It drifts ~2× less
  and, critically, **conserves energy (0.99×)** where the MLP's velocities
  inflate ~240×. The MLP stays *finite* but becomes *unphysical* late in the
  rollout; the GNN stays physical. Message passing clearly buys rollout quality.
- This holds even though the GNN was **handicapped** (fewer epochs, truncated
  curriculum), so the gap is if anything understated.

## Decisions this unblocks

1. **Carry the GNN forward** as the primary architecture for the 10k phase.
2. **100 clips limits fidelity** (100 ms faithful horizon) → the **10k dataset
   is justified**; more data should extend the horizon.
3. Re-run both to full convergence on 10k with the complete rollout curriculum
   (the GNN never finished its high-horizon stages here) — use the 8h-timeout
   `train_modal.py` so it doesn't get cut short again.

## Caveats

- GNN vs MLP epoch counts differ (timeout) — the one-step gap is clean, the
  rollout gap is directionally clear but not curriculum-matched.
- Rollout starts from frame 0 with velocity history seeded from v0 (cold start);
  same handicap for both, so fair for A-vs-B.
