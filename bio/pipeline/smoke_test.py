#!/usr/bin/env python3
"""Smoke-test for Step1 (prelabel) and Step2 (synth_build).

Usage:
  python -m bio smoke-test \
    --skeletons_sign /path/to/skeletons_dir_or_combined.json \
    --skeletons_no_event /path/to/no_event_skeletons_dir \
    --csv /path/to/train.csv \
    --workdir /tmp/bio_smoke \
    --split train

What it checks:
  - Step1 runs and produces canonical index.json/index.csv rows with split + dataset metadata.
  - For a sign sample: exactly one B and some I.
  - If there is any no_event with frames: B==0 and I==0.
  - Step2 runs (using native tails/no_event negatives) and produces shards with B present in many samples.

Notes:
  - Current Step2 implementation still needs some O-source frames, but they may come from sign tails as well.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _run(cmd: List[str], cwd: Path | None = None) -> None:
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def sniff_csv(csv_path: Path) -> Tuple[csv.Dialect, List[Dict[str, str]]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        except csv.Error:
            class _D(csv.Dialect):
                delimiter = "\t" if ("\t" in sample and "," not in sample) else ","
                quotechar = '"'
                doublequote = True
                skipinitialspace = False
                lineterminator = "\n"
                quoting = csv.QUOTE_MINIMAL
            dialect = _D()
        rdr = csv.DictReader(f, dialect=dialect)
        rows = list(rdr)
    if not rows:
        raise RuntimeError(f"CSV seems empty: {csv_path}")
    return dialect, rows


def make_mini_csv(
    src_csv: Path,
    dst_csv: Path,
    split: str,
    max_sign: int = 200,
    max_noev: int = 30,
) -> Tuple[Dict[str, int], List[Dict[str, str]]]:
    dialect, rows = sniff_csv(src_csv)

    # heuristic split filter compatible with prelabel: uses 'split' or 'train/is_train'
    def use_row(r: Dict[str, str]) -> bool:
        raw_split = (r.get("split") or "").strip().lower()
        if raw_split:
            return raw_split == split
        raw_train = r.get("train") or r.get("is_train")
        if raw_train is None:
            return True
        v = str(raw_train).strip().lower()
        if v in ("true", "1", "yes", "y"):
            return split == "train"
        if v in ("false", "0", "no", "n"):
            return split != "train"
        # fallback: allow comparing string to split
        return v == split

    sign_rows: List[Dict[str, str]] = []
    noev_rows: List[Dict[str, str]] = []

    for r in rows:
        if not use_row(r):
            continue
        label = (r.get("text") or "").strip().lower()
        if not label:
            continue
        if label == "no_event":
            noev_rows.append(r)
        else:
            sign_rows.append(r)

    sign_rows = sign_rows[: max_sign]
    noev_rows = noev_rows[: max_noev]

    keep = sign_rows + noev_rows
    if not keep:
        raise RuntimeError("No rows selected for mini CSV (check split/filter columns)")

    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    with dst_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), dialect=dialect)
        w.writeheader()
        w.writerows(keep)

    stats = {"sign": len(sign_rows), "no_event": len(noev_rows), "total": len(keep)}
    print("Mini CSV:", stats, "->", dst_csv)
    return stats, keep


def build_merged_skeletons(
    rows: List[Dict[str, str]],
    sign_root: Path,
    noev_root: Path | None,
    workdir: Path,
) -> Path:
    if noev_root is None:
        return sign_root
    if sign_root.resolve() == noev_root.resolve():
        return sign_root
    if not sign_root.is_dir() or not noev_root.is_dir():
        print("[WARN] Expected both skeleton roots to be directories. Using sign skeletons only.")
        return sign_root

    merged = workdir / "skeletons_merged"
    merged.mkdir(parents=True, exist_ok=True)
    missing: List[str] = []
    copied = 0

    def _resolve_path(root: Path, vid: str) -> Path:
        pp = root / f"{vid}_pp.json"
        base = root / f"{vid}.json"
        if pp.exists():
            return pp
        if base.exists():
            return base
        return base

    for r in rows:
        vid = Path((r.get("attachment_id") or "").strip()).stem
        if not vid:
            continue
        label = (r.get("text") or "").strip().lower()
        src_root = noev_root if label == "no_event" else sign_root
        src = _resolve_path(src_root, vid)
        if not src.exists():
            alt_root = sign_root if src_root == noev_root else noev_root
            alt = _resolve_path(alt_root, vid)
            if alt.exists():
                src = alt
            else:
                missing.append(str(src))
                continue
        dst = merged / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1

    if missing:
        print(f"[WARN] Missing {len(missing)} skeleton JSONs; example: {missing[0]}")
    print(f"Merged skeletons dir: {merged} (copied {copied} files)")
    return merged


def load_index(prelabel_dir: Path) -> List[Dict[str, object]]:
    idx = prelabel_dir / "index.json"
    if not idx.exists():
        raise FileNotFoundError(f"Missing index.json: {idx}")
    return json.loads(idx.read_text(encoding="utf-8"))


def pick_sample(index_rows: List[Dict[str, object]], want_no_event: bool) -> Dict[str, object] | None:
    for r in index_rows:
        if bool(r.get("is_no_event", False)) != want_no_event:
            continue
        if int(r.get("T_total", 0) or 0) <= 0:
            continue
        path = str(r.get("path_to_npz", ""))
        if not path:
            continue
        return r
    return None


def check_npz(prelabel_dir: Path, row: Dict[str, object]) -> Dict[str, int]:
    p = prelabel_dir / str(row["path_to_npz"])
    z = np.load(p, allow_pickle=False)
    bio = z["bio"].astype(np.uint8)
    stats = {
        "T": int(bio.shape[0]),
        "B": int((bio == 1).sum()),
        "I": int((bio == 2).sum()),
        "O": int((bio == 0).sum()),
        "start": int(z["start_idx"]),
        "end": int(z["end_idx"]),
    }
    print("NPZ stats:", p.name, stats)
    return stats


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skeletons_sign", default=None, help="Path to sign skeletons (dir or combined JSON).")
    ap.add_argument("--skeletons_no_event", default=None, help="Path to no_event skeletons (dir or combined JSON).")
    ap.add_argument("--skeletons", default=None, help="Deprecated alias for --skeletons_sign.")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--split", default="train")

    ap.add_argument("--max_sign", type=int, default=200)
    ap.add_argument("--max_noev", type=int, default=30)

    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--num_synth_samples", type=int, default=128)
    ap.add_argument("--shard_size", type=int, default=64)

    ap.add_argument("--require_step2", action="store_true")
    args = ap.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    if args.csv is None:
        args.csv = str(repo_root / "datasets" / "data" / "annotations.csv")
    if args.skeletons_sign is None:
        if args.skeletons:
            args.skeletons_sign = args.skeletons
        else:
            args.skeletons_sign = str(repo_root / "datasets" / "skeletons")
    if args.skeletons_no_event is None:
        args.skeletons_no_event = str(repo_root / "datasets" / "skeletons" / "no_event_old")

    src_csv = Path(args.csv).resolve()
    mini_csv = workdir / "mini.csv"
    _, mini_rows = make_mini_csv(
        src_csv,
        mini_csv,
        split=str(args.split).strip().lower(),
        max_sign=args.max_sign,
        max_noev=args.max_noev,
    )

    sign_root = Path(args.skeletons_sign).resolve()
    noev_arg = (args.skeletons_no_event or "").strip()
    noev_root = Path(noev_arg).resolve() if noev_arg else None
    skeletons_path = build_merged_skeletons(mini_rows, sign_root, noev_root, workdir)

    prelabel_dir = workdir / "prelabels"
    prelabel_dir.mkdir(parents=True, exist_ok=True)

    # Step1
    _run(
        [
            sys.executable,
            "-m",
            "bio",
            "prelabel",
            "--skeletons",
            str(skeletons_path),
            "--csv",
            str(mini_csv),
            "--split",
            str(args.split),
            "--out",
            str(prelabel_dir),
            "--num_workers",
            "0",
        ],
        cwd=repo_root,
    )

    index_rows = load_index(prelabel_dir)
    print(f"Step1 produced {len(index_rows)} index rows")
    required_fields = {"vid", "label_str", "path_to_npz", "T_total", "start_idx", "end_idx", "is_no_event", "split", "dataset", "source_group"}
    missing_fields = required_fields - set(index_rows[0].keys())
    if missing_fields:
        raise RuntimeError(f"Step1 index rows are missing canonical fields: {sorted(missing_fields)}")

    sign_row = pick_sample(index_rows, want_no_event=False)
    if sign_row is None:
        raise RuntimeError("No non-empty sign sample produced in Step1 (T_total>0).")
    sign_stats = check_npz(prelabel_dir, sign_row)
    if sign_stats["B"] != 1 or sign_stats["I"] <= 0:
        raise RuntimeError(f"Sign sample BIO looks wrong: {sign_stats}")

    noev_row = pick_sample(index_rows, want_no_event=True)
    if noev_row is None:
        print("\n[WARN] No non-empty no_event samples found after Step1 (T_total>0).")
        print("       Step2 (current implementation) will FAIL because it requires no_event negatives for O.")
        if args.require_step2:
            raise RuntimeError("require_step2 set but no_event samples with frames are available.")
        print("       Skipping Step2.\n")
        return

    noev_stats = check_npz(prelabel_dir, noev_row)
    if noev_stats["B"] != 0 or noev_stats["I"] != 0:
        raise RuntimeError(f"no_event BIO looks wrong: {noev_stats}")

    # Step2
    synth_dir = workdir / "synth"
    synth_dir.mkdir(parents=True, exist_ok=True)

    _run(
        [
            sys.executable,
            "-m",
            "bio",
            "synth-build",
            "--prelabel_dir",
            str(prelabel_dir),
            "--out_dir",
            str(synth_dir),
            "--num_samples",
            str(args.num_synth_samples),
            "--seq_len",
            str(args.seq_len),
            "--shard_size",
            str(args.shard_size),
            "--min_signs",
            "2",
            "--max_signs",
            "6",
            "--gap_min",
            "0",
            "--gap_max",
            "20",
            "--blend_prob",
            "0.25",
            "--blend_k_min",
            "2",
            "--blend_k_max",
            "6",
        ],
        cwd=repo_root,
    )

    stats_path = synth_dir / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    gen = stats.get("generated", {})
    print("Step2 stats:", json.dumps(gen, ensure_ascii=True, indent=2))
    for key in ("no_event_source_counts", "gap_stats", "pad_stats", "tail_len_stats"):
        if key not in gen:
            raise RuntimeError(f"Step2 stats.json is missing generated.{key}")

    frac = float(gen.get("samples_with_B_frac", 0.0))
    if frac < 0.5:
        raise RuntimeError(f"Too few samples contain B after Step2: samples_with_B_frac={frac}")

    # check first shard
    shards = sorted((synth_dir / "shards").glob("*.npz"))
    if not shards:
        raise RuntimeError("No shards produced by Step2")
    z = np.load(shards[0], allow_pickle=False)
    pts, bio = z["pts"], z["bio"]
    print("Shard shapes:", shards[0].name, "pts", pts.shape, "bio", bio.shape)
    if pts.ndim != 4 or bio.ndim != 2:
        raise RuntimeError("Bad shard tensor shapes")

    # optional: test shard dataset loader and sampler
    try:
        from bio.core.datasets.shard_dataset import ShardedBiosDataset, make_boundary_aware_sampler

        ds = ShardedBiosDataset(synth_dir)
        sampler = make_boundary_aware_sampler(ds, p_with_b=0.85)
        print(f"ShardDataset len={len(ds)}; sampler ok; has_b_frac={float(ds.has_b.mean()):.3f}")
        # draw a few samples
        it = iter(sampler)
        for _ in range(3):
            idx = int(next(it))
            _ = ds[idx]
        print("ShardDataset sampling OK")
    except Exception as e:
        print("[WARN] ShardDataset/sampler test skipped/failed:", e)

    print("\nSmoke test PASSED (Step1+Step2).")


if __name__ == "__main__":
    main()
