"""GPU speed benchmark (H3): neural inference vs Taichi MPM vs implicit fallback.

Fair per-frame comparison. The neural surrogate advances one dt=1e-3 frame in a
single forward. The Taichi MLS-MPM reference needs `log_every` substeps of
dt=1e-4 for the same physical time (CFL). The implicit mass-spring fallback is
unconditionally stable, so it takes the big dt=1e-3 in one step. Each solver is
timed with warmup (excluding Taichi JIT / CUDA graph capture) and proper device
synchronisation (ti.sync / torch.cuda.synchronize), so we compare steady-state
CUDA kernel execution -- not one-off Python/compile overhead. We then form the
hybrid effective per-frame time from the measured per-scenario fallback fraction
and report the speedup vs full-MPM.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np


def time_mpm_substep(cfg, pinned_mask, n_warm=30, n_time=200):
    """Wall time of one Taichi MLS-MPM substep (dt=1e-4), steady-state."""
    import taichi as ti
    from src.mpm_cloth import MPMClothSim
    sim = MPMClothSim(cfg)
    sim.reset(pinned_mask=pinned_mask)
    for _ in range(n_warm):        # JIT-compile every kernel, reach steady state
        sim.step(f_ext=None)
    ti.sync()
    t0 = time.perf_counter()
    for _ in range(n_time):
        sim.step(f_ext=None)
    ti.sync()
    return (time.perf_counter() - t0) / n_time


def time_neural_forward(ckpt_path, clip, meta, n_warm=30, n_time=200, device="cuda"):
    """Wall time of one neural inference (feature assembly + GNN forward +
    integration) on the given device, steady-state."""
    import torch
    from src.hybrid import HybridRollout
    from scripts.eval_rollout import _load_ckpt
    model, stats, model_kind, include_F, C = _load_ckpt(ckpt_path)
    model = model.to(device)
    gx, gy = meta["grid"]
    dt = float(meta["dt_s"]) * int(meta["log_every_substeps"])
    xref, vref = clip["x"], clip["v"]
    s0 = C - 1
    ro = HybridRollout(model, model_kind, stats, dt=dt, detector=None,
                       threshold=float("inf"), fallback_sim=None, history_C=C,
                       include_F=include_F, device=device)
    ro.reset(xref[s0], vref[s0], grid_x=int(gx), grid_y=int(gy),
             v_history=vref[s0 - C + 1: s0 + 1])
    sync = (lambda: torch.cuda.synchronize()) if device == "cuda" else (lambda: None)
    with torch.no_grad():
        for _ in range(n_warm):
            ro._neural_accel()
        sync()
        t0 = time.perf_counter()
        for _ in range(n_time):
            ro._neural_accel()
        sync()
    return (time.perf_counter() - t0) / n_time


def time_fallback_step(meta, clip, n_warm=5, n_time=40):
    """Wall time of one implicit mass-spring step (dt=1e-3, CPU/scipy)."""
    import scripts.eval_hybrid as eh
    sim = eh._fallback_for_clip(meta)
    xref, vref = clip["x"], clip["v"]
    sim.x = xref[0].astype(np.float64)
    sim.v = vref[0].astype(np.float64)
    dt = float(meta["dt_s"]) * int(meta["log_every_substeps"])
    for _ in range(n_warm):
        sim.step(f_ext=None, dt=dt, cg_max_iters=50, cg_tol=1e-4)
    t0 = time.perf_counter()
    for _ in range(n_time):
        sim.step(f_ext=None, dt=dt, cg_max_iters=50, cg_tol=1e-4)
    return (time.perf_counter() - t0) / n_time


def report(t_mpm_sub, log_every, t_neural, t_fallback, fallback_fracs):
    ms = lambda s: f"{s*1e3:.2f} ms"
    t_mpm_frame = t_mpm_sub * log_every
    print("\n=== per-frame wall time (dt=1e-3) ===")
    print(f"  full-MPM (Taichi, {log_every} substeps): {ms(t_mpm_frame)}  "
          f"({ms(t_mpm_sub)}/substep)")
    print(f"  neural  (GNN forward, 1 step):           {ms(t_neural)}")
    print(f"  implicit fallback (1 step, CPU):         {ms(t_fallback)}")
    print(f"\n  pure-neural speedup vs full-MPM: {t_mpm_frame / t_neural:.1f}x")
    print("\n=== hybrid effective per-frame (by fallback fraction) ===")
    for scen, f in fallback_fracs.items():
        t_hy = (1 - f) * t_neural + f * t_fallback
        print(f"  {scen:>9} (f={f:.2f}): {ms(t_hy)}  ->  "
              f"{t_mpm_frame / t_hy:.1f}x faster than full-MPM")
