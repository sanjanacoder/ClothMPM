"""Edge-case validation sweep for the MPM cloth dataset generator (CPU).

Purpose: before spending money on the full GPU dataset run, confirm the
*production* config (64x64 cloth, 128^3 grid, dt=1e-4, full 1.0 s duration)
stays stable and physically sane at the extremes of every scenario sampler.
The --smoke path of generate_dataset.py never exercises this config, so a
green smoke suite does not by itself de-risk the GPU run.

For each curated clip this runs the real solver on CPU and checks:
  * finite            -- no NaN/Inf in x, v, a, F over the whole trajectory
  * stable            -- solver's own `is_unstable` flag never fires
  * pinned_drift      -- pinned corners stay put (max displacement ~ 0)
  * drape_falls       -- drape clips descend (mean y_end < y_start)
  * energy            -- peak kinetic energy stays bounded (no blow-up)
It also runs one spec twice to confirm bit-for-bit determinism, which the
dataset's reproducibility guarantees depend on.

Results stream to report/edge_case_report.{csv,json} as each clip finishes,
so a long CPU sweep is inspectable while running and survives interruption.

Usage:
  # fast self-test of this harness (16x16, short) -- ~1 min, no GPU needed
  python scripts/validate_edge_cases.py --quick

  # full-resolution, full-duration sweep (~12 min/clip on CPU)
  python scripts/validate_edge_cases.py

  # full resolution but shorter clips to catch blow-ups cheaply
  python scripts/validate_edge_cases.py --duration 0.4

  # run a single named case
  python scripts/validate_edge_cases.py --only wind_max
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.generate_dataset import ClipSpec, cfg_for_clip, cfg_hash

ROOT = Path(__file__).resolve().parents[1]

# Peak kinetic energy above this (Joules) is treated as a blow-up. The cloth is
# 0.2 kg; a coherent 10 m/s motion is ~10 J, so 50 J is a generous ceiling that
# still flags a runaway solver long before it reaches NaN.
KE_BLOWUP_J = 50.0
# Pinned corners are enforced by the solver; anything past a few grid cells of
# drift is a real leak, not float noise. dx at full res is ~0.0156 m.
PINNED_DRIFT_TOL_M = 0.05


# -----------------------------------------------------------------------------
# Curated edge cases -- the extremes of each scenario's sampler.
# -----------------------------------------------------------------------------
# Each entry is raw params; pinned-corner indices depend on the runtime grid
# resolution (16 in --quick, 64 in full) so they are resolved in the loop.

EDGE_CASES: list[dict] = [
    # Wind: maximum force magnitude is the most likely explicit-MPM blow-up.
    dict(name="wind_max", scenario="wind", pin=True, height=1.0,
         sphere=(0.0, -1.0, 0.0), radius=0.05, wind=(4.0, 0.0, 0.0)),
    dict(name="wind_diag_max", scenario="wind", pin=True, height=1.0,
         sphere=(0.0, -1.0, 0.0), radius=0.05,
         wind=(4.0 * np.cos(np.pi / 4), 0.0, 4.0 * np.sin(np.pi / 4))),
    dict(name="wind_min", scenario="wind", pin=True, height=1.0,
         sphere=(0.0, -1.0, 0.0), radius=0.05, wind=(0.5, 0.0, 0.0)),
    # Collision: largest sphere, centered, high enough to punch deep into the
    # falling sheet -> most contact nodes, hardest constraint projection.
    dict(name="collision_deep", scenario="collision", pin=True, height=1.0,
         sphere=(1.0, 0.4, 1.0), radius=0.18, wind=(0.0, 0.0, 0.0)),
    # Smallest sphere -> few nodes carry the whole projection (stiffest local).
    dict(name="collision_small", scenario="collision", pin=True, height=1.0,
         sphere=(1.0, 0.2, 1.0), radius=0.10, wind=(0.0, 0.0, 0.0)),
    # Off-center collision, corner of the sampled box.
    dict(name="collision_offset", scenario="collision", pin=True, height=1.0,
         sphere=(1.15, 0.35, 0.85), radius=0.15, wind=(0.0, 0.0, 0.0)),
    # Drape: highest drop + largest sphere -> highest impact speed onto a big
    # obstacle. Lowest drop is the gentle-baseline sanity check.
    dict(name="drape_high", scenario="drape", pin=False, height=0.9,
         sphere=(1.0, 0.5, 1.0), radius=0.25, wind=(0.0, 0.0, 0.0)),
    dict(name="drape_low", scenario="drape", pin=False, height=0.6,
         sphere=(1.0, 0.35, 1.0), radius=0.15, wind=(0.0, 0.0, 0.0)),
    dict(name="drape_offset", scenario="drape", pin=False, height=0.8,
         sphere=(1.15, 0.5, 0.85), radius=0.20, wind=(0.0, 0.0, 0.0)),
]


def build_spec(case: dict, gy: int, duration_s: float, log_every: int) -> ClipSpec:
    """Materialize a ClipSpec, resolving pinned corners against the grid size."""
    pinned = [0 * gy + 0, 0 * gy + (gy - 1)] if case["pin"] else []
    # Seed is cosmetic here (params are fixed, not sampled) but keeps clip names
    # and the config hash distinct per case.
    seed = abs(hash(case["name"])) % 1000
    return ClipSpec(
        scenario=case["scenario"],
        seed=seed,
        duration_s=duration_s,
        initial_height_m=case["height"],
        sphere_center_m=tuple(case["sphere"]),
        sphere_radius_m=case["radius"],
        pinned_corner_indices=pinned,
        wind_force_n=tuple(case["wind"]),
        log_every_substeps=log_every,
    )


def run_case(spec: ClipSpec, base_cfg: dict, smoke: bool) -> dict:
    """Run one clip on CPU with periodic stability diagnostics.

    Returns a dict of trajectory arrays plus a per-frame stability record.
    Mirrors generate_dataset.run_one_clip but toggles the solver's diagnostics
    path on logged frames so we capture `is_unstable` and grid max velocity.
    """
    import taichi as ti
    ti.reset()
    from src.mpm_cloth import MPMClothSim

    cfg = cfg_for_clip(base_cfg, spec, smoke)
    cfg["backend"]["arch"] = "cpu"  # this harness is CPU-only by design
    gx, gy = cfg["cloth"]["grid"]
    n = gx * gy

    pinned_mask = np.zeros(n, dtype=bool)
    for idx in spec.pinned_corner_indices:
        pinned_mask[idx] = True

    total_wind = np.array(spec.wind_force_n, dtype=np.float32)
    f_ext = np.tile(total_wind / n, (n, 1))

    sim = MPMClothSim(cfg)
    sim.reset(pinned_mask=pinned_mask)

    dt = float(cfg["mpm"]["dt_s"])
    n_substeps = int(round(spec.duration_s / dt))
    log_every = spec.log_every_substeps

    xs, vs = [], []
    any_unstable = False
    max_grid_v = 0.0
    for s in range(n_substeps):
        want_diag = (s % log_every == 0)
        diag = sim.step(f_ext=f_ext, diagnostics=want_diag)
        if want_diag:
            st = sim.state()
            xs.append(st["x"].astype(np.float32))
            vs.append(st["v"].astype(np.float32))
            if diag is not None:
                any_unstable = any_unstable or bool(diag["is_unstable"])
                max_grid_v = max(max_grid_v, float(diag["max_velocity"]))

    x = np.stack(xs, axis=0)
    v = np.stack(vs, axis=0)
    p_mass = float(base_cfg["cloth"]["mass_kg"]) / n
    ke = 0.5 * p_mass * (v ** 2).sum(axis=(1, 2))  # (T,)
    return dict(x=x, v=v, ke=ke, any_unstable=any_unstable, max_grid_v=max_grid_v,
                pinned_mask=pinned_mask, config_hash=cfg_hash(cfg),
                n_particles=n, n_frames=x.shape[0])


def validate(case: dict, spec: ClipSpec, res: dict) -> dict:
    """Turn raw trajectory arrays into pass/fail checks + headline numbers."""
    x, v, ke = res["x"], res["v"], res["ke"]
    pinned = res["pinned_mask"]

    finite = bool(np.isfinite(x).all() and np.isfinite(v).all())
    stable = not res["any_unstable"]

    if pinned.any():
        pin_ref = x[0:1, pinned, :]
        pinned_drift = float(np.abs(x[:, pinned, :] - pin_ref).max())
        pinned_ok = pinned_drift <= PINNED_DRIFT_TOL_M
    else:
        pinned_drift = 0.0
        pinned_ok = True

    if case["scenario"] == "drape":
        y0, y1 = float(x[0, :, 1].mean()), float(x[-1, :, 1].mean())
        drape_ok = y1 < y0
        drape_dy = y1 - y0
    else:
        drape_ok = True
        drape_dy = float("nan")

    peak_ke = float(ke.max())
    energy_ok = np.isfinite(peak_ke) and peak_ke < KE_BLOWUP_J

    checks = dict(finite=finite, stable=stable, pinned_ok=pinned_ok,
                  drape_ok=drape_ok, energy_ok=energy_ok)
    return dict(
        name=case["name"], scenario=case["scenario"],
        n_frames=res["n_frames"], n_particles=res["n_particles"],
        passed=all(checks.values()),
        finite=finite, stable=stable, pinned_ok=pinned_ok,
        drape_ok=drape_ok, energy_ok=energy_ok,
        pinned_drift_m=round(pinned_drift, 6),
        peak_ke_j=round(peak_ke, 4),
        max_grid_v=round(res["max_grid_v"], 4),
        drape_dy_m=None if np.isnan(drape_dy) else round(drape_dy, 4),
        config_hash=res["config_hash"],
        failed_checks=[k for k, ok in checks.items() if not ok],
    )


def compute_row(case: dict, base_cfg: dict, gy: int, duration: float,
                quick: bool) -> dict:
    """Run + validate one case in-process, returning a JSON-safe result row."""
    spec = build_spec(case, gy, duration, int(base_cfg["mpm"]["substeps_per_log"]))
    t0 = time.perf_counter()
    res = run_case(spec, base_cfg, smoke=quick)
    row = validate(case, spec, res)
    # Hash of positions: lets the driver check determinism across processes
    # without shipping arrays back.
    row["x_sha256"] = hashlib.sha256(res["x"].tobytes()).hexdigest()[:16]
    row["wall_s"] = round(time.perf_counter() - t0, 1)
    return row


def run_worker(args, base_cfg: dict, gy: int) -> None:
    """Subprocess entrypoint: run one named case, write its row, exit.

    Isolation matters here: a runaway clip can eject a particle out of the grid
    and trigger a C-level SIGBUS that no Python `try/except` can catch. Running
    each case in its own process turns that crash into a non-zero exit code the
    driver records as a failure, instead of killing the whole sweep.
    """
    case = next(c for c in EDGE_CASES if c["name"] == args.worker)
    row = compute_row(case, base_cfg, gy, args.duration, args.quick)
    Path(args.result_file).write_text(json.dumps(row))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "mpm.yaml"))
    ap.add_argument("--duration", type=float, default=1.0,
                    help="Per-clip duration in seconds (default 1.0 = full).")
    ap.add_argument("--quick", action="store_true",
                    help="16x16/CPU self-test of this harness (fast, not a real check).")
    ap.add_argument("--only", default="", help="Run a single named case.")
    ap.add_argument("--out", default=str(ROOT / "report"))
    ap.add_argument("--worker", default="", help=argparse.SUPPRESS)
    ap.add_argument("--result-file", default="", help=argparse.SUPPRESS)
    args = ap.parse_args()

    base_cfg = yaml.safe_load(Path(args.config).read_text())
    gy = 16 if args.quick else base_cfg["cloth"]["grid"][1]

    # Subprocess worker path: run exactly one case and return.
    if args.worker:
        run_worker(args, base_cfg, gy)
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "edge_case_report.csv"
    json_path = out_dir / "edge_case_report.json"
    result_file = out_dir / "_worker_result.json"

    cases = [c for c in EDGE_CASES if not args.only or c["name"] == args.only]
    if not cases:
        raise SystemExit(f"no case named {args.only!r}; "
                         f"known: {[c['name'] for c in EDGE_CASES]}")

    mode = "quick 16x16" if args.quick else "full 64x64/128^3"
    print(f"Edge-case sweep [{mode}] duration={args.duration}s  "
          f"{len(cases)} case(s) on CPU (subprocess-isolated)", flush=True)

    def run_isolated(case_name: str) -> dict:
        """Run one case in a child process; a crash becomes a FAIL row."""
        result_file.unlink(missing_ok=True)
        cmd = [sys.executable, __file__, "--worker", case_name,
               "--result-file", str(result_file), "--duration", str(args.duration),
               "--config", args.config]
        if args.quick:
            cmd.append("--quick")
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=str(ROOT),
                              env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
                              capture_output=True, text=True)
        wall = round(time.perf_counter() - t0, 1)
        if proc.returncode == 0 and result_file.exists():
            return json.loads(result_file.read_text())
        # Non-zero exit: negative == killed by signal (e.g. -10 = SIGBUS).
        rc = proc.returncode
        why = (f"killed by signal {-rc}" if rc < 0
               else f"exited {rc}")
        case = next(c for c in EDGE_CASES if c["name"] == case_name)
        return dict(name=case_name, scenario=case["scenario"], passed=False,
                    failed_checks=["crashed"], error=f"process {why}", wall_s=wall)

    rows: list[dict] = []
    started = time.perf_counter()
    for i, case in enumerate(cases, 1):
        row = run_isolated(case["name"])
        rows.append(row)
        pd.DataFrame(rows).to_csv(csv_path, index=False)  # stream after each clip
        json_path.write_text(json.dumps(rows, indent=2))
        tag = "PASS" if row["passed"] else "FAIL"
        detail = "" if row["passed"] else f"  <-- {row.get('failed_checks')} {row.get('error','')}"
        print(f"[{i}/{len(cases)}] {tag}  {case['name']:16s} "
              f"{row['wall_s']:6.1f}s{detail}", flush=True)

    # Determinism: rerun a case that passed and require an identical position
    # hash. Skip if nothing passed (nothing trustworthy to re-run).
    det_ok = None
    passed_rows = [r for r in rows if r["passed"] and "x_sha256" in r]
    if not args.only and passed_rows:
        ref = passed_rows[0]
        again = run_isolated(ref["name"])
        det_ok = bool(again.get("x_sha256") == ref["x_sha256"])
        print(f"[determinism] {ref['name']} rerun hash match: {det_ok}", flush=True)

    result_file.unlink(missing_ok=True)
    n_pass = sum(r["passed"] for r in rows)
    elapsed = (time.perf_counter() - started) / 60
    summary = dict(rows=rows, determinism_ok=det_ok, n_pass=n_pass,
                   n_total=len(rows), duration_s=args.duration, mode=mode,
                   elapsed_min=round(elapsed, 1))
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\n{n_pass}/{len(rows)} cases passed"
          f"{'' if det_ok is None else f', determinism={det_ok}'} "
          f"in {elapsed:.1f} min -> {csv_path}", flush=True)
    if n_pass != len(rows) or det_ok is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
