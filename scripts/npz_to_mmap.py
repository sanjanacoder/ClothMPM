"""Re-export .npz trajectory clips to a per-array .npy layout for true mmap.

Why: NpzFile members cannot be memory-mapped (mmap_mode is a no-op for .npz;
each access fully decompresses the array). At 64x64 that is ~300MB/clip, so
random-access training either OOMs (unbounded cache) or thrashes (bounded cache
re-decompresses on every miss). Standalone uncompressed .npy files CAN be
mmap'd, so np.load(mmap_mode='r')[frame] reads only that frame's bytes via the
OS page cache -- flat memory AND fast random access, scaling to the 10k set.

Layout produced (mirrors the source scenario/clip structure):
  <out>/<scenario>/<clip_stem>/
      x.npy  v.npy  a.npy  [F.npy]  contact_flag.npy
      meta.json                      # holds grid + the original clip meta
  <out>/index.csv                    # same schema as the source manifest, but
                                     # 'path' points (absolute) at each clip dir

Usage (from the parent repo root):
  .venv/bin/python scripts/npz_to_mmap.py \
      --manifest data/fullres100/index.csv --out data/fullres100_mmap
  # skip the (unused, ~half the bytes) F array when training with --no-F:
  .venv/bin/python scripts/npz_to_mmap.py ... --no-F
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

ARRAYS_ALWAYS = ("x", "v", "a", "contact_flag")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def convert(manifest: Path, out: Path, include_F: bool,
            scenarios: list[str] | None) -> Path:
    df = pd.read_csv(manifest)
    df = df[df["status"] == "OK"].reset_index(drop=True)
    if scenarios:
        df = df[df["scenario"].isin(scenarios)].reset_index(drop=True)
    if len(df) == 0:
        raise SystemExit(f"no OK clips in {manifest} after filters")

    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    keys = list(ARRAYS_ALWAYS) + (["F"] if include_F else [])

    new_rows = []
    n = len(df)
    for i, row in df.iterrows():
        src = ROOT / row["path"]
        scenario = Path(row["path"]).parent.name
        stem = Path(row["path"]).stem
        clip_dir = out / scenario / stem
        clip_dir.mkdir(parents=True, exist_ok=True)

        npz = np.load(src, allow_pickle=True)
        for k in keys:
            # np.save writes an uncompressed .npy -> memory-mappable.
            np.save(clip_dir / f"{k}.npy", np.ascontiguousarray(npz[k]))
        meta = npz["meta"].item() if "meta" in npz.files else {}
        (clip_dir / "meta.json").write_text(json.dumps(meta, default=_json_default))

        new = dict(row)
        new["path"] = str(clip_dir)          # absolute -> ROOT/path resolves anywhere
        new_rows.append(new)
        if (i + 1) % max(1, n // 20) == 0 or i + 1 == n:
            print(f"  {i + 1:>4}/{n}  {scenario}/{stem}", flush=True)

    out_manifest = out / "index.csv"
    pd.DataFrame(new_rows).to_csv(out_manifest, index=False)
    print(f"\nwrote {n} clips -> {out}")
    print(f"manifest -> {out_manifest}  (F {'included' if include_F else 'skipped'})")
    return out_manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--no-F", dest="include_F", action="store_false",
                    help="skip the deformation gradient (unused when training "
                         "--no-F; saves ~half the disk)")
    args = ap.parse_args()
    convert(Path(args.manifest), Path(args.out), args.include_F, args.scenarios)


if __name__ == "__main__":
    main()
