# Running Dataset Generation on Modal

Generates the full 10k-clip MPM cloth trajectory dataset in ~1-2 h by fanning
out `run_one_clip()` across up to 100 T4 GPU containers simultaneously.
Compare: ~27 h sequential on a single T4.

Script: [`scripts/generate_dataset_modal.py`](../scripts/generate_dataset_modal.py)

---

## Prerequisites

### 1. Create a Modal account

Sign up at [modal.com](https://modal.com). The free tier includes $30/month of
compute credit — enough for a full smoke test and several partial runs.

### 2. Install the Modal CLI

```bash
# Inside the ClothMPM venv
pip install modal
```

### 3. Authenticate

```bash
modal setup
# Opens a browser tab. Log in and copy the token. One-time per machine.
```

---

## Smoke test — 6 clips, ~3 min, < $0.10

Run this first to confirm the image builds, the mount resolves, and the physics
is correct before spending money on the full dataset.

```bash
cd /path/to/ClothMPM

modal run scripts/generate_dataset_modal.py --smoke
```

What happens:

- Modal builds the container image (CUDA 12.1 + Taichi) — takes ~2 min on
  first run, cached on all subsequent runs.
- Mounts `src/`, `scripts/`, and `configs/` into each container.
- Fans out 6 tasks (2 drape / 2 wind / 2 collision) to T4 GPU containers.
- Results stream back as containers finish.
- `data/cloth_trajectories/index.csv` is written locally.
- `index.csv` is also written into the Modal Volume alongside the `.npz` files.

Expected terminal output:

```
Smoke mode: 6 clips at 16×16 grid (schema + physics check).
Dispatching 6 clips to Modal  (2 drape / 2 wind / 2 collision)
[OK] drape seed=0 idx=0 14.3s
[OK] wind  seed=100 idx=0 12.1s
...
     6/6  ✓ 6  ✗ 0
Local manifest → .../data/cloth_trajectories/index.csv
```

### Download smoke clips

```bash
modal volume get cloth-mpm-trajectories /data/cloth_trajectories ./data/cloth_trajectories
```

### Verify smoke clips

```bash
.venv/bin/python -m pytest tests/test_dataset.py -v
```

All tests should pass, including `test_wind_pins_held` and `test_drape_y_decreases`.

---

## Full-resolution validation — 6 clips, ~5 min, < $0.20

**Run this before the full dataset.** The smoke above runs at 16×16; it validates
the schema and pipeline but *not* the production 64×64 / 128³ config. On macOS/arm64
CPU, full-resolution clips crash inside Taichi regardless of physics (see
`docs/datagen-validation-findings.md`, issue ③), so full-res can only be validated
on the GPU target. This step runs a handful of full-resolution clips on CUDA to
confirm (a) drape/collision are stable at production resolution, and (b) issue ③ is
arm64-only and does not reproduce on the GPU.

```bash
# smoke=False (full 64×64/128³), just a few clips per scenario
modal run scripts/generate_dataset_modal.py --n-drape 3 --n-collision 3
```

Expect 6/6 `[OK]` with no container crashes. Then download and check finiteness:

```bash
modal volume get cloth-mpm-trajectories /data/cloth_trajectories ./data/cloth_trajectories
.venv/bin/python -c "
import numpy as np, glob
for p in sorted(glob.glob('data/cloth_trajectories/**/*.npz', recursive=True)):
    d = np.load(p, allow_pickle=True)
    ok = all(np.isfinite(d[k]).all() for k in ('x','v','a','F'))
    print(('OK ' if ok else 'BAD'), p.split('/')[-1], 'detF range',
          f\"{np.linalg.det(d['F']).min():.2e}..{np.linalg.det(d['F']).max():.2e}\")
"
```

`detF` should stay near 1 (no exponential drift). If all 6 are `OK`, proceed to the
full run. If they crash on CUDA too, the instability is not arm64-specific and the
solver work in `docs/wind-deferral.md` §7 applies more broadly than just wind.

---

## Full dataset — 10k clips, ~1-2 h, ~$30-60

```bash
modal run scripts/generate_dataset_modal.py
```

Default counts: 5000 drape / 5000 collision. **Wind is deferred** (default 0) —
a corner-pinned sheet under sustained wind inverts an element the explicit solver
cannot integrate, which crashes the container on CUDA too (this is physics-level,
not the arm64-only issue). See `docs/wind-deferral.md`; pass `--n-wind N` to opt in
once the solver is fixed. All 10k tasks are dispatched immediately; Modal runs up
to 100 containers concurrently.

### Custom counts

```bash
# Trial run — 1k clips
modal run scripts/generate_dataset_modal.py --n-drape 400 --n-wind 300 --n-collision 300
```

### Switch to A10G (3-4× faster, ~2.5× the cost)

Edit line 105 of `scripts/generate_dataset_modal.py`:

```python
# change
gpu="T4",
# to
gpu="A10G",
```

---

## Monitor progress

**Terminal** — progress logs every 5% of clips as results stream in.

**Modal dashboard** — [modal.com/apps](https://modal.com/apps) shows live
container count, per-container logs, and errors. Click any container to tail
its stdout (`[OK]` / `[ERROR]` lines from the remote function).

**CLI log tail:**

```bash
modal app logs cloth-mpm-datagen
```

---

## Download the full dataset

```bash
# ~10 GB for the full 10k clips — takes 5-15 min depending on network
modal volume get cloth-mpm-trajectories /data/cloth_trajectories ./data/cloth_trajectories
```

The local directory layout matches the paths stored in `index.csv` exactly.

---

## Verify downloaded data

```bash
.venv/bin/python -c "
import numpy as np, glob

clips = sorted(glob.glob('data/cloth_trajectories/**/*.npz', recursive=True))
print(f'{len(clips)} clips downloaded')

# Pinned corner check — should be 0.00e+00 for all wind/collision clips
for p in [c for c in clips if '/wind/' in c or '/collision/' in c][:4]:
    d = np.load(p, allow_pickle=True)
    pinned = d['meta'].item()['pinned_corner_indices']
    drift = np.abs(d['x'][:, pinned, :] - d['x'][0:1, pinned, :]).max()
    print(f'{p.split(\"/\")[-1]}  pinned_drift={drift:.2e}')
"
```

---

## Cost reference

| Run | Clips | GPU | Approx time | Approx cost |
|---|---|---|---|---|
| Smoke | 6 | T4 | 3 min | < $0.10 |
| Trial | 1 000 | T4 | 15 min | ~$3 |
| Full | 10 000 | T4 | 90 min | ~$35 |
| Full | 10 000 | A10G | 30 min | ~$45 |

Modal bills per-second of actual GPU time. Idle time between dispatch and
execution is not charged.

---

## Handling errors

If some clips fail (network blip, OOM, Taichi kernel error), the row in
`index.csv` will show `status=ERROR`. Each container already retries once
automatically (`retries=1` in the decorator). For persistent failures:

1. Filter `index.csv` for `status != OK`.
2. Note the `scenario` and `seed` values.
3. Adjust `seed_base` and `counts` in a one-off invocation to re-run only
   those seeds.

The Modal Volume is append-safe — re-running a subset of seeds will add the
new clips without touching existing ones.

---

## Architecture notes

- **One clip per container.** Each invocation calls `ti.reset()` → `ti.init()`
  → simulate → save. Taichi's global state is per-process, so isolation is
  guaranteed.
- **Volume path layout.** Clips are saved to `/data/cloth_trajectories/` inside
  the container (the volume mount point). `modal volume get` downloads them to a
  matching local path so `index.csv` paths resolve without edits.
- **Source mount vs baked image.** `src/`, `scripts/`, and `configs/` are
  mounted at runtime via `modal.Mount`. This means local edits to the simulator
  or config take effect immediately without rebuilding the image.
- **Manifest.** `index.csv` is written both locally (by the `@app.local_entrypoint`)
  and into the volume (by `write_manifest_to_volume`) so a full `modal volume get`
  is sufficient to reproduce the dataset on any machine.
