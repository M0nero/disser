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
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bio.core.config_utils import load_config_section, write_dataset_manifest, write_run_config
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
    dataset_signature: Dict[str, Any]


def _path_str(path: Path) -> str:
    return path.as_posix()


def _length_stats(values: Sequence[int]) -> Dict[str, Any]:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.int32)
    return {
        "count": int(arr.size),
        "min": int(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
    }


def _count_histogram(counts: Sequence[int]) -> Dict[str, int]:
    if not counts:
        return {}
    bins = [
        ("1", 1, 1),
        ("2", 2, 2),
        ("3", 3, 3),
        ("4", 4, 4),
        ("5-9", 5, 9),
        ("10-19", 10, 19),
        ("20+", 20, None),
    ]
    out: Dict[str, int] = {}
    for name, lo, hi in bins:
        total = 0
        for value in counts:
            if hi is None:
                ok = int(value) >= lo
            else:
                ok = lo <= int(value) <= hi
            if ok:
                total += 1
        out[name] = int(total)
    return out


def _git_sha(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True)
    except Exception:
        return ""
    return out.strip()


def _config_sha(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _load_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def _summarize_prelabel_source(path: Path) -> Dict[str, Any]:
    rows = load_prelabel_index(path)
    datasets: Dict[str, int] = {}
    splits: Dict[str, int] = {}
    for row in rows:
        dataset = str(row.dataset or "")
        split = str(row.split or "")
        datasets[dataset or ""] = datasets.get(dataset or "", 0) + 1
        splits[split or ""] = splits.get(split or "", 0) + 1
    return {
        "dir": _path_str(path),
        "summary": _load_json_if_exists(path / "summary.json"),
        "dataset_manifest": _load_json_if_exists(path / "dataset_manifest.json"),
        "rows": int(len(rows)),
        "datasets": datasets,
        "splits": splits,
    }


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
    preferred_noev_prelabel_dirs: Optional[List[Path]] = None,
    extra_noev_prelabel_dirs: Optional[List[Path]] = None,
    overlap_report_summary: Optional[Dict[str, Any]] = None,
) -> BuildStats:
    out_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

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
        preferred_noev_prelabel_dirs=[Path(d) for d in (preferred_noev_prelabel_dirs or [])],
        extra_noev_prelabel_dirs=extra_dirs,
    )

    pools_summary = summarize_pools(
        ds.sign_pool,
        ds.noev_pool,
        noev_pool_primary=ds.noev_pool_primary,
        noev_pool_extra=ds.noev_pool_extra,
    )

    index_rows: List[Dict[str, Any]] = []
    N = int(num_samples)
    shard_size = max(1, int(shard_size))
    L = int(cfg.seq_len)
    total_shards = max(1, (N + shard_size - 1) // shard_size)

    print(
        f"[synth-build] start out_dir={out_dir} samples={N} seq_len={L} shard_size={shard_size} total_shards={total_shards}",
        flush=True,
    )

    pts_buf: List[np.ndarray] = []
    mask_buf: List[np.ndarray] = []
    bio_buf: List[np.ndarray] = []
    meta_buf: List[str] = []

    shard_id = 0
    generated_b_count = 0
    generated_i_count = 0
    generated_o_count = 0
    samples_with_b = 0
    no_event_source_counts: Dict[str, int] = {}
    no_event_role_counts: Dict[str, int] = {}
    source_group_counts: Dict[str, int] = {}
    sign_label_counts: Dict[str, int] = {}
    sign_label_frame_counts: Dict[str, int] = {}
    sign_source_group_counts: Dict[str, int] = {}
    sign_source_group_frames: Dict[str, int] = {}
    gap_lens: List[int] = []
    pad_lens: List[int] = []
    tail_lens: List[int] = []

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

        for part in meta.get("parts", []):
            length = int(part.get("length", 0) or 0)
            if length <= 0:
                continue
            role = str(part.get("role", ""))
            part_type = str(part.get("type", ""))
            source_type = str(part.get("source_type", ""))
            source_group = str(part.get("source_group", ""))
            if part_type == "sign":
                label = str(part.get("label_str", "") or "")
                sign_label_counts[label] = sign_label_counts.get(label, 0) + 1
                sign_label_frame_counts[label] = sign_label_frame_counts.get(label, 0) + length
                if source_group:
                    sign_source_group_counts[source_group] = sign_source_group_counts.get(source_group, 0) + 1
                    sign_source_group_frames[source_group] = sign_source_group_frames.get(source_group, 0) + length
                continue
            if part_type not in ("no_event", "blend"):
                continue
            if role == "gap":
                gap_lens.append(length)
            if role in ("pad_pre", "pad_post"):
                pad_lens.append(length)
            if source_type in ("tail_pre", "tail_post"):
                tail_lens.append(length)
            if part_type == "blend":
                key = "blend"
            elif role in ("pad_pre", "pad_post"):
                key = "pad"
            elif source_type:
                key = source_type
            else:
                key = part_type or "unknown"
            no_event_source_counts[key] = no_event_source_counts.get(key, 0) + length
            role_key = role or part_type or "unknown"
            no_event_role_counts[role_key] = no_event_role_counts.get(role_key, 0) + length
            if source_group:
                source_group_counts[source_group] = source_group_counts.get(source_group, 0) + length

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
                    "path_to_npz": (Path("shards") / shard_name).as_posix(),
                    "seq_len": int(L),
                    "V": int(pts_sh.shape[2]),
                    "num_samples": int(pts_sh.shape[0]),
                }
            )

            shard_done = shard_id + 1
            samples_done = min(N, shard_done * shard_size)
            elapsed = time.time() - start_time
            print(
                f"[synth-build] wrote shard {shard_done}/{total_shards} samples={samples_done}/{N} "
                f"elapsed_sec={elapsed:.1f}",
                flush=True,
            )
            shard_id += 1
            pts_buf.clear()
            mask_buf.clear()
            bio_buf.clear()
            meta_buf.clear()

    _write_index(out_dir, index_rows)

    prelabel_sources = [_summarize_prelabel_source(prelabel_dir)]
    for path in preferred_noev_prelabel_dirs or []:
        prelabel_sources.append(_summarize_prelabel_source(path))
    for path in extra_noev_prelabel_dirs or []:
        prelabel_sources.append(_summarize_prelabel_source(path))
    signature_payload = {
        "prelabel_dir": _path_str(prelabel_dir),
        "preferred_noev_prelabel_dirs": [_path_str(x) for x in (preferred_noev_prelabel_dirs or [])],
        "extra_noev_prelabel_dirs": [_path_str(x) for x in (extra_noev_prelabel_dirs or [])],
        "cfg": asdict(cfg),
        "num_samples": int(N),
        "shard_size": int(shard_size),
        "seed": int(seed),
    }
    dataset_signature = {
        "git_sha": _git_sha(Path(__file__).resolve().parents[2]),
        "config_sha": _config_sha(signature_payload),
        "inputs": {
            "prelabel_dir": _path_str(prelabel_dir),
            "preferred_noev_prelabel_dirs": [_path_str(x) for x in (preferred_noev_prelabel_dirs or [])],
            "extra_noev_prelabel_dirs": [_path_str(x) for x in (extra_noev_prelabel_dirs or [])],
        },
        "prelabel_sources": prelabel_sources,
        "overlap_report_summary": overlap_report_summary or {},
    }

    # Save stats
    total_o_frames = max(1, int(generated_o_count))
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
        "no_event_source_counts": dict(sorted(no_event_source_counts.items())),
        "no_event_source_fracs": {
            k: float(v) / float(total_o_frames)
            for k, v in sorted(no_event_source_counts.items())
        },
        "no_event_role_counts": dict(sorted(no_event_role_counts.items())),
        "gap_stats": _length_stats(gap_lens),
        "pad_stats": _length_stats(pad_lens),
        "tail_len_stats": _length_stats(tail_lens),
        "sign_label_counts": dict(sorted(sign_label_counts.items())),
        "sign_label_frame_counts": dict(sorted(sign_label_frame_counts.items())),
        "sign_source_group_coverage": {
            "unique_groups": int(len(sign_source_group_counts)),
            "reuse_histogram": _count_histogram(list(sign_source_group_counts.values())),
            "top_groups_by_segments": [
                {"source_group": key, "segments": int(value)}
                for key, value in sorted(sign_source_group_counts.items(), key=lambda item: (-item[1], item[0]))[:10]
            ],
            "top_groups_by_frames": [
                {"source_group": key, "frames": int(value)}
                for key, value in sorted(sign_source_group_frames.items(), key=lambda item: (-item[1], item[0]))[:10]
            ],
        },
        "source_group_coverage": {
            "unique_groups": int(len(source_group_counts)),
            "top_groups_by_frames": [
                {"source_group": key, "frames": int(value)}
                for key, value in sorted(source_group_counts.items(), key=lambda item: (-item[1], item[0]))[:10]
            ],
        },
    }
    stats = BuildStats(
        num_samples=int(N),
        seq_len=int(L),
        shard_size=int(shard_size),
        seed=int(seed),
        pools=pools_summary,
        generated=generated,
        dataset_signature=dataset_signature,
    )
    (out_dir / "stats.json").write_text(json.dumps(asdict(stats), ensure_ascii=True, indent=2), encoding="utf-8")
    print(
        f"[synth-build] done out_dir={out_dir} shards={shard_id} total_samples={N} elapsed_sec={time.time() - start_time:.1f}",
        flush=True,
    )
    write_dataset_manifest(
        out_dir,
        stage="synth_build",
        args={
            "prelabel_dir": _path_str(prelabel_dir),
            "out_dir": _path_str(out_dir),
            "cfg": asdict(cfg),
            "num_samples": int(N),
            "shard_size": int(shard_size),
            "seed": int(seed),
            "min_sign_len": int(min_sign_len),
            "preferred_noev_prelabel_dirs": [_path_str(x) for x in (preferred_noev_prelabel_dirs or [])],
            "extra_noev_prelabel_dirs": [_path_str(x) for x in (extra_noev_prelabel_dirs or [])],
        },
        section="synth_build",
        inputs=dataset_signature.get("inputs", {}),
        counts={
            "num_samples": int(N),
            "seq_len": int(L),
            "shards": int(shard_id),
        },
        extra={
            "dataset_signature": dataset_signature,
            "generated": generated,
            "pools": pools_summary,
        },
    )
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
    preferred_default = defaults.get("preferred_noev_prelabel_dir", [])
    if isinstance(preferred_default, str):
        preferred_default = [preferred_default] if preferred_default.strip() else []
    if not isinstance(preferred_default, list):
        preferred_default = []
    p.add_argument(
        "--preferred_noev_prelabel_dir",
        action="append",
        default=None,
        help="Preferred canonical Step1 no_event dirs from the same domain (sampled before external negatives).",
    )
    extra_default = defaults.get("extra_noev_prelabel_dir", [])
    if isinstance(extra_default, str):
        extra_default = [extra_default] if extra_default.strip() else []
    if not isinstance(extra_default, list):
        extra_default = []
    p.add_argument(
        "--extra_noev_prelabel_dir",
        action="append",
        default=None,
        help="Additional canonical Step1 no_event dirs containing no_event samples only (e.g., IPNHand background).",
    )

    # composition
    p.add_argument("--min_signs", type=int, default=int(defaults.get("min_signs", 2)))
    p.add_argument("--max_signs", type=int, default=int(defaults.get("max_signs", 6)))
    p.add_argument("--gap_min", type=int, default=int(defaults.get("gap_min", 0)))
    p.add_argument("--gap_max", type=int, default=int(defaults.get("gap_max", 20)))
    p.add_argument("--sign_ctx_pre", type=int, default=int(defaults.get("sign_ctx_pre", 0)))
    p.add_argument("--sign_ctx_post", type=int, default=int(defaults.get("sign_ctx_post", 0)))
    p.add_argument("--include_sign_tails_as_noev", dest="include_sign_tails_as_noev", action="store_true", default=bool(defaults.get("include_sign_tails_as_noev", True)))
    p.add_argument("--no_include_sign_tails_as_noev", dest="include_sign_tails_as_noev", action="store_false")
    p.add_argument("--min_tail_len", type=int, default=int(defaults.get("min_tail_len", 4)))
    p.add_argument("--primary_noev_prob", type=float, default=float(defaults.get("primary_noev_prob", 0.90)))
    p.add_argument(
        "--source_sampling",
        type=str,
        default=str(defaults.get("source_sampling", "uniform_source")),
        choices=["uniform_segment", "uniform_source"],
        help="How to sample no_event clips inside each pool (uniform_source = source-group aware).",
    )
    p.add_argument(
        "--sign_sampling",
        type=str,
        default=str(defaults.get("sign_sampling", defaults.get("source_sampling", "uniform_source"))),
        choices=["uniform_segment", "uniform_source", "uniform_label_source"],
        help="How to sample sign clips (uniform_label_source = label-balanced, then source-group balanced).",
    )
    p.add_argument(
        "--stitch_noev_chunks",
        dest="stitch_noev_chunks",
        action="store_true",
        default=bool(defaults.get("stitch_noev_chunks", True)),
        help="Fill long no_event spans by stitching multiple clips instead of repeat-padding a single clip.",
    )
    p.add_argument("--no_stitch_noev_chunks", dest="stitch_noev_chunks", action="store_false")
    p.add_argument(
        "--overlap_report",
        type=str,
        default=str(defaults.get("overlap_report", "")),
        help="Optional overlap report JSON. If provided, a short summary is embedded into stats.json.",
    )

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

    args = p.parse_args(argv)
    if args.preferred_noev_prelabel_dir is None:
        args.preferred_noev_prelabel_dir = list(preferred_default)
    if args.extra_noev_prelabel_dir is None:
        args.extra_noev_prelabel_dir = list(extra_default)
    return args


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    prelabel_dir = Path(args.prelabel_dir)
    out_dir = Path(args.out_dir)
    preferred_dirs = list(dict.fromkeys(Path(x) for x in args.preferred_noev_prelabel_dir))
    extra_dirs = list(dict.fromkeys(Path(x) for x in args.extra_noev_prelabel_dir))
    write_run_config(out_dir, args, config_path=args.config, section="synth_build")

    cfg = SynthConfig(
        seq_len=int(args.seq_len),
        min_signs=int(args.min_signs),
        max_signs=int(args.max_signs),
        gap_min=int(args.gap_min),
        gap_max=int(args.gap_max),
        sign_ctx_pre=int(args.sign_ctx_pre),
        sign_ctx_post=int(args.sign_ctx_post),
        include_sign_tails_as_noev=bool(args.include_sign_tails_as_noev),
        min_tail_len=int(args.min_tail_len),
        primary_noev_prob=float(args.primary_noev_prob),
        source_sampling=str(args.source_sampling),
        sign_sampling=str(args.sign_sampling),
        stitch_noev_chunks=bool(args.stitch_noev_chunks),
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
        preferred_noev_prelabel_dirs=preferred_dirs,
        extra_noev_prelabel_dirs=extra_dirs,
        overlap_report_summary=_load_json_if_exists(Path(args.overlap_report)) if args.overlap_report else None,
    )
    print(json.dumps(asdict(stats), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
