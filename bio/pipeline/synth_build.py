from __future__ import annotations

"""
Step 2 (offline): build synthetic "continuous" BIO sequences from Step-1 prelabels.

This script creates a new dataset directory with:
- shards/*.npz (or samples/*.npz)
- index.json + index.csv describing the generated sequences
- stats.json with pool + generation statistics

The generator consumes Step-1 artifacts from `bio/pipeline/prelabel.py`:
- prelabel_dir/index.json or index.csv
- prelabel_dir/<sample_id>.npz with keys: pts, mask, bio, label_str, is_no_event, start_idx, end_idx

Usage (example):
  python -m bio synth-build \
      --prelabel_dir /path/to/prelabels_train \
      --out_dir /path/to/synth_train \
      --num_samples 200000 \
      --seq_len 256 \
      --shard_size 256 \
      --seed 1337

If you want "infinite" synthetic data, you can skip this script and use
`SyntheticContinuousDataset` from bio/core/datasets/synth_dataset.py directly during training.
"""

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bio.core.config_utils import load_config_section, write_run_config
from bio.core.datasets.synth_dataset import (
    SynthConfig,
    load_prelabel_index,
    summarize_pools,
    SyntheticContinuousDataset,
)


@dataclass
class BuildStats:
    num_samples: int
    seq_len: int
    shard_size: int
    seed: int
    pools: Dict[str, Any]
    generated: Dict[str, Any]


def _write_index(out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    (out_dir / "index.json").write_text(json.dumps(rows, ensure_ascii=True, indent=2), encoding="utf-8")

    fields = ["id", "path_to_npz", "seq_len", "V"]
    with (out_dir / "index.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def _save_shard(path: Path, pts: np.ndarray, mask: np.ndarray, bio: np.ndarray, metas: List[str]) -> None:
    # pts: (N,L,V,3), mask: (N,L,V,1), bio: (N,L)
    np.savez(
        path,
        pts=pts.astype(np.float32, copy=False),
        mask=mask.astype(np.float32, copy=False),
        bio=bio.astype(np.uint8, copy=False),
        meta=np.asarray(metas, dtype=np.unicode_),
    )


def build_offline(
    prelabel_dir: Path,
    out_dir: Path,
    cfg: SynthConfig,
    num_samples: int,
    shard_size: int,
    seed: int,
    min_sign_len: int,
    epoch_size: Optional[int] = None,
    extra_noev_prelabel_dirs: Optional[List[Path]] = None,
) -> BuildStats:
    out_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    rows = load_prelabel_index(prelabel_dir)
    extra_dirs = [Path(d) for d in (extra_noev_prelabel_dirs or [])]

    # Online dataset as a generator primitive
    ds = SyntheticContinuousDataset(
        prelabel_dir=prelabel_dir,
        rows=rows,
        cfg=cfg,
        epoch_size=int(epoch_size or num_samples),
        seed=seed,
        min_sign_len=min_sign_len,
        extra_noev_prelabel_dirs=extra_dirs,
    )

    pools_summary = summarize_pools(ds.sign_pool, ds.noev_pool)

    index_rows: List[Dict[str, Any]] = []
    N = int(num_samples)
    shard_size = max(1, int(shard_size))
    L = int(cfg.seq_len)

    pts_buf: List[np.ndarray] = []
    mask_buf: List[np.ndarray] = []
    bio_buf: List[np.ndarray] = []
    meta_buf: List[str] = []

    shard_id = 0
    generated_b_count = 0
    generated_i_count = 0
    generated_o_count = 0
    samples_with_b = 0

    for i in range(N):
        sample = ds[i]
        pts = sample["pts"]
        mask = sample["mask"]
        bio = sample["bio"]
        meta = sample["meta"]

        if pts.shape[0] != L:
            raise RuntimeError(f"BUG: seq_len mismatch (got {pts.shape[0]}, expected {L})")

        pts_buf.append(pts[None, ...])
        mask_buf.append(mask[None, ...])
        bio_buf.append(bio[None, ...])
        meta_buf.append(json.dumps(meta, ensure_ascii=True))

        generated_b_count += int((bio == 1).sum())
        generated_i_count += int((bio == 2).sum())
        generated_o_count += int((bio == 0).sum())
        if (bio == 1).any():
            samples_with_b += 1

        if len(pts_buf) >= shard_size or (i == N - 1):
            pts_sh = np.concatenate(pts_buf, axis=0)
            mask_sh = np.concatenate(mask_buf, axis=0)
            bio_sh = np.concatenate(bio_buf, axis=0)

            shard_name = f"shard_{shard_id:06d}.npz"
            shard_path = shards_dir / shard_name
            _save_shard(shard_path, pts_sh, mask_sh, bio_sh, meta_buf)

            index_rows.append(
                {
                    "id": f"shard_{shard_id:06d}",
                    "path_to_npz": str(Path("shards") / shard_name),
                    "seq_len": int(L),
                    "V": int(pts_sh.shape[2]),
                    "num_samples": int(pts_sh.shape[0]),
                }
            )

            shard_id += 1
            pts_buf.clear()
            mask_buf.clear()
            bio_buf.clear()
            meta_buf.clear()

    _write_index(out_dir, index_rows)

    # Save stats
    generated = {
        "total_samples": int(N),
        "total_frames": int(N * L),
        "bio_counts": {"B": int(generated_b_count), "I": int(generated_i_count), "O": int(generated_o_count)},
        "bio_fracs": {
            "B": float(generated_b_count) / float(N * L),
            "I": float(generated_i_count) / float(N * L),
            "O": float(generated_o_count) / float(N * L),
        },
        "samples_with_B_frac": float(samples_with_b) / float(N) if N > 0 else 0.0,
        "shards": int(shard_id),
    }
    stats = BuildStats(
        num_samples=int(N),
        seq_len=int(L),
        shard_size=int(shard_size),
        seed=int(seed),
        pools=pools_summary,
        generated=generated,
    )
    (out_dir / "stats.json").write_text(json.dumps(asdict(stats), ensure_ascii=True, indent=2), encoding="utf-8")
    return stats


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default="")
    pre_args, _ = pre.parse_known_args(argv)
    defaults = load_config_section(pre_args.config, "synth_build")

    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=pre_args.config, help="Path to config JSON (section: synth_build).")
    p.add_argument(
        "--prelabel_dir",
        type=str,
        default=defaults.get("prelabel_dir"),
        required=not bool(defaults.get("prelabel_dir")),
        help="Directory produced by prelabel step.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=defaults.get("out_dir"),
        required=not bool(defaults.get("out_dir")),
        help="Where to write synthetic continuous dataset",
    )

    # generation size
    p.add_argument("--num_samples", type=int, default=int(defaults.get("num_samples", 200_000)), help="How many synthetic sequences to generate")
    p.add_argument("--seq_len", type=int, default=int(defaults.get("seq_len", 256)), help="Frames per synthetic sequence")
    p.add_argument("--shard_size", type=int, default=int(defaults.get("shard_size", 256)), help="How many sequences per shard .npz")
    p.add_argument("--seed", type=int, default=int(defaults.get("seed", 1337)))

    # pools
    p.add_argument("--min_sign_len", type=int, default=int(defaults.get("min_sign_len", 8)), help="Discard sign segments shorter than this")
    extra_default = defaults.get("extra_noev_prelabel_dir", [])
    if isinstance(extra_default, str):
        extra_default = [extra_default] if extra_default.strip() else []
    if not isinstance(extra_default, list):
        extra_default = []
    p.add_argument(
        "--extra_noev_prelabel_dir",
        action="append",
        default=list(extra_default),
        help="Additional Step1 prelabel dirs containing no_event samples only (e.g., IPNHand background).",
    )

    # composition
    p.add_argument("--min_signs", type=int, default=int(defaults.get("min_signs", 2)))
    p.add_argument("--max_signs", type=int, default=int(defaults.get("max_signs", 6)))
    p.add_argument("--gap_min", type=int, default=int(defaults.get("gap_min", 0)))
    p.add_argument("--gap_max", type=int, default=int(defaults.get("gap_max", 20)))
    p.add_argument("--sign_ctx_pre", type=int, default=int(defaults.get("sign_ctx_pre", 0)))
    p.add_argument("--sign_ctx_post", type=int, default=int(defaults.get("sign_ctx_post", 0)))

    # blending
    p.add_argument("--blend_prob", type=float, default=float(defaults.get("blend_prob", 0.25)))
    p.add_argument("--blend_k_min", type=int, default=int(defaults.get("blend_k_min", 2)))
    p.add_argument("--blend_k_max", type=int, default=int(defaults.get("blend_k_max", 6)))
    p.add_argument("--blend_only_when_gap_leq", type=int, default=int(defaults.get("blend_only_when_gap_leq", 2)))

    # crop/pad
    p.add_argument("--crop_mode", type=str, default=defaults.get("crop_mode", "random"), choices=["random", "start", "center"])
    p.add_argument("--pad_mode", type=str, default=defaults.get("pad_mode", "end_no_event"), choices=["end_no_event", "both_no_event"])

    # caching
    p.add_argument("--npz_cache_items", type=int, default=int(defaults.get("npz_cache_items", 32)))

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    prelabel_dir = Path(args.prelabel_dir)
    out_dir = Path(args.out_dir)
    extra_dirs = [Path(x) for x in args.extra_noev_prelabel_dir]
    write_run_config(out_dir, args, config_path=args.config, section="synth_build")

    cfg = SynthConfig(
        seq_len=int(args.seq_len),
        min_signs=int(args.min_signs),
        max_signs=int(args.max_signs),
        gap_min=int(args.gap_min),
        gap_max=int(args.gap_max),
        sign_ctx_pre=int(args.sign_ctx_pre),
        sign_ctx_post=int(args.sign_ctx_post),
        blend_prob=float(args.blend_prob),
        blend_k_min=int(args.blend_k_min),
        blend_k_max=int(args.blend_k_max),
        blend_only_when_gap_leq=int(args.blend_only_when_gap_leq),
        crop_mode=args.crop_mode,
        pad_mode=args.pad_mode,
        npz_cache_items=int(args.npz_cache_items),
    )

    stats = build_offline(
        prelabel_dir=prelabel_dir,
        out_dir=out_dir,
        cfg=cfg,
        num_samples=int(args.num_samples),
        shard_size=int(args.shard_size),
        seed=int(args.seed),
        min_sign_len=int(args.min_sign_len),
        extra_noev_prelabel_dirs=extra_dirs,
    )
    print(json.dumps(asdict(stats), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
