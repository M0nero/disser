from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bio.core.config_utils import write_dataset_manifest, write_run_config
from bio.ipn import prelabel as ipn_prelabel
from bio.pipeline import prelabel as slovo_prelabel
from bio.pipeline import synth_build

STRICT_SPLIT_POLICY = {
    "mode": "strict_source_group_separation",
    "description": "The same raw video/source_group must not appear in multiple splits. Any overlap blocks the build.",
    "enforced_on": ["slovo_sign", "slovo_no_event", "slovo_union", "ipn_hand"],
}


def _read_manifest(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
        return rows
    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return [x for x in raw if isinstance(x, dict)]
        if isinstance(raw, dict):
            return [raw]
        return []
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported manifest format: {path}")


def _norm_split(raw: object) -> str:
    val = str(raw or "").strip().lower()
    if val == "test":
        return "val"
    return val


def _sample(values: Sequence[str], limit: int = 10) -> List[str]:
    items = sorted(str(v) for v in values if str(v))
    return items[:limit]


def _audit_overlap(
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    *,
    source_name: str,
) -> Dict[str, Any]:
    train_set = {str(x) for x in train_ids if str(x)}
    val_set = {str(x) for x in val_ids if str(x)}
    overlap = train_set & val_set
    return {
        "source": source_name,
        "train_count": int(len(train_set)),
        "val_count": int(len(val_set)),
        "overlap_count": int(len(overlap)),
        "overlap_examples": _sample(overlap),
    }


def _load_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def _promote_many_transactional(pairs: Sequence[Tuple[Path, Path]]) -> None:
    token = uuid.uuid4().hex
    backups: Dict[Path, Path] = {}
    promoted: List[Path] = []
    rollback_paths: List[Path] = []
    try:
        for src, dst in pairs:
            src = Path(src)
            dst = Path(dst)
            if not src.exists():
                raise FileNotFoundError(src)
            dst.parent.mkdir(parents=True, exist_ok=True)
            backup = dst.parent / f".{dst.name}.bak_tmp_{token}"
            if backup.exists():
                shutil.rmtree(backup, ignore_errors=True)
            if dst.exists():
                os.replace(str(dst), str(backup))
                backups[dst] = backup
            os.replace(str(src), str(dst))
            promoted.append(dst)
    except Exception:
        for dst in reversed(promoted):
            rollback_tmp = dst.parent / f".{dst.name}.rollback_tmp_{token}"
            if rollback_tmp.exists():
                shutil.rmtree(rollback_tmp, ignore_errors=True)
            if dst.exists():
                os.replace(str(dst), str(rollback_tmp))
                rollback_paths.append(rollback_tmp)
        for dst, backup in backups.items():
            if backup.exists() and not dst.exists():
                os.replace(str(backup), str(dst))
        for path in rollback_paths:
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
        raise
    for backup in backups.values():
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)


def build_overlap_report(
    slovo_csv: Path,
    slovo_no_event_csv: Path,
    ipn_manifest: Path,
) -> Dict[str, Any]:
    sign_train = slovo_prelabel.parse_csv(slovo_csv, "train")
    sign_val = slovo_prelabel.parse_csv(slovo_csv, "val")
    noev_train = slovo_prelabel.parse_csv(slovo_no_event_csv, "train")
    noev_val = slovo_prelabel.parse_csv(slovo_no_event_csv, "val")

    slovo_sign_report = _audit_overlap(
        [row.source_group for row in sign_train.rows],
        [row.source_group for row in sign_val.rows],
        source_name="slovo_sign",
    )
    slovo_noev_report = _audit_overlap(
        [row.source_group for row in noev_train.rows],
        [row.source_group for row in noev_val.rows],
        source_name="slovo_no_event",
    )
    slovo_union_report = _audit_overlap(
        [row.source_group for row in sign_train.rows] + [row.source_group for row in noev_train.rows],
        [row.source_group for row in sign_val.rows] + [row.source_group for row in noev_val.rows],
        source_name="slovo_union",
    )

    manifest_rows = _read_manifest(ipn_manifest)
    ipn_train_ids = [str(r.get("video_id") or "") for r in manifest_rows if _norm_split(r.get("split")) == "train"]
    ipn_val_ids = [str(r.get("video_id") or "") for r in manifest_rows if _norm_split(r.get("split")) == "val"]
    ipn_report = _audit_overlap(ipn_train_ids, ipn_val_ids, source_name="ipn_hand")

    report = {
        "split_policy": dict(STRICT_SPLIT_POLICY),
        "slovo_sign": slovo_sign_report,
        "slovo_no_event": slovo_noev_report,
        "slovo_union": slovo_union_report,
        "ipn_hand": ipn_report,
    }
    report["ok"] = all(
        int(item.get("overlap_count", 0)) == 0
        for key, item in report.items()
        if isinstance(item, dict) and key != "split_policy"
    )
    return report


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Canonical BIO v2 dataset rebuild.")
    ap.add_argument("--out_root", type=str, default="outputs/bio_out_v2", help="Root directory for rebuilt BIO artifacts.")
    ap.add_argument("--ipn_out_root", type=str, default="outputs/ipnhand", help="Root directory for IPN Step1 pools.")
    ap.add_argument("--slovo_skeletons", type=str, default="datasets/skeletons/Slovo")
    ap.add_argument("--slovo_csv", type=str, default="datasets/data/annotations.csv")
    ap.add_argument("--slovo_no_event_csv", type=str, default="datasets/data/annotations_no_event.csv")
    ap.add_argument("--ipn_manifest", type=str, default="outputs/ipnhand/manifests/ipn_d0x_manifest.jsonl")
    ap.add_argument("--ipn_segments_dir", type=str, default="datasets/skeletons/ipnhand")

    ap.add_argument("--slovo_prelabel_config", type=str, default="bio/configs/bio_default.json")
    ap.add_argument("--synth_train_config", type=str, default="bio/configs/bio_default.json")
    ap.add_argument("--synth_val_config", type=str, default="bio/configs/bio_val.json")
    ap.add_argument("--ipn_train_config", type=str, default="bio/configs/ipn_default.json")
    ap.add_argument("--ipn_val_config", type=str, default="bio/configs/ipn_val.json")

    ap.add_argument("--slovo_num_workers", type=int, default=8)
    ap.add_argument("--train_num_samples", type=int, default=100000)
    ap.add_argument("--val_num_samples", type=int, default=10000)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--shard_size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--min_tail_len", type=int, default=4)
    ap.add_argument("--primary_noev_prob", type=float, default=0.90)
    ap.add_argument("--source_sampling", type=str, default="uniform_source", choices=["uniform_segment", "uniform_source"])
    ap.add_argument("--sign_sampling", type=str, default="uniform_label_source", choices=["uniform_segment", "uniform_source", "uniform_label_source"])
    ap.add_argument("--stitch_noev_chunks", dest="stitch_noev_chunks", action="store_true", default=True)
    ap.add_argument("--no_stitch_noev_chunks", dest="stitch_noev_chunks", action="store_false")
    ap.add_argument("--include_sign_tails_as_noev", dest="include_sign_tails_as_noev", action="store_true", default=True)
    ap.add_argument("--no_include_sign_tails_as_noev", dest="include_sign_tails_as_noev", action="store_false")
    ap.add_argument("--prefer_pp", dest="prefer_pp", action="store_true", default=True)
    ap.add_argument("--no_prefer_pp", dest="prefer_pp", action="store_false")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    out_root = Path(args.out_root).resolve()
    ipn_out_root = Path(args.ipn_out_root).resolve()
    out_root.parent.mkdir(parents=True, exist_ok=True)
    ipn_out_root.parent.mkdir(parents=True, exist_ok=True)

    stage_root = Path(
        tempfile.mkdtemp(
            prefix=f".{out_root.name}.build_tmp_",
            dir=str(out_root.parent),
        )
    )
    try:
        stage_out_root = stage_root / out_root.name
        stage_out_root.mkdir(parents=True, exist_ok=True)
        stage_ipn_root = stage_root / "__ipnhand_stage__"
        write_run_config(stage_out_root, args, section="build_dataset")

        overlap_report = build_overlap_report(
            Path(args.slovo_csv),
            Path(args.slovo_no_event_csv),
            Path(args.ipn_manifest),
        )
        overlap_path = stage_out_root / "overlap_report.json"
        overlap_path.write_text(json.dumps(overlap_report, ensure_ascii=True, indent=2), encoding="utf-8")
        if not bool(overlap_report.get("ok", False)):
            raise RuntimeError(f"Split overlap detected. See {overlap_path}")

        slovo_train_dir = stage_out_root / "prelabels_slovo_train"
        slovo_val_dir = stage_out_root / "prelabels_slovo_val"
        slovo_noev_train_dir = stage_out_root / "prelabels_slovo_noev_train"
        slovo_noev_val_dir = stage_out_root / "prelabels_slovo_noev_val"
        ipn_train_dir = stage_ipn_root / "prelabels_train"
        ipn_val_dir = stage_ipn_root / "prelabels_val"
        synth_train_dir = stage_out_root / "synth_train"
        synth_val_dir = stage_out_root / "synth_val"

        print(f"[build-dataset] step1 slovo train -> {slovo_train_dir}", flush=True)
        slovo_prelabel.main(
            [
                "--config", args.slovo_prelabel_config,
                "--skeletons", args.slovo_skeletons,
                "--csv", args.slovo_csv,
                "--split", "train",
                "--out", str(slovo_train_dir),
                "--num_workers", str(args.slovo_num_workers),
            ] + (["--prefer_pp"] if args.prefer_pp else ["--no_prefer_pp"])
        )
        print(f"[build-dataset] step1 slovo val -> {slovo_val_dir}", flush=True)
        slovo_prelabel.main(
            [
                "--config", args.slovo_prelabel_config,
                "--skeletons", args.slovo_skeletons,
                "--csv", args.slovo_csv,
                "--split", "val",
                "--out", str(slovo_val_dir),
                "--num_workers", str(args.slovo_num_workers),
            ] + (["--prefer_pp"] if args.prefer_pp else ["--no_prefer_pp"])
        )
        print(f"[build-dataset] step1 slovo no_event train -> {slovo_noev_train_dir}", flush=True)
        slovo_prelabel.main(
            [
                "--config", args.slovo_prelabel_config,
                "--skeletons", args.slovo_skeletons,
                "--csv", args.slovo_no_event_csv,
                "--split", "train",
                "--out", str(slovo_noev_train_dir),
                "--num_workers", str(args.slovo_num_workers),
            ] + (["--prefer_pp"] if args.prefer_pp else ["--no_prefer_pp"])
        )
        print(f"[build-dataset] step1 slovo no_event val -> {slovo_noev_val_dir}", flush=True)
        slovo_prelabel.main(
            [
                "--config", args.slovo_prelabel_config,
                "--skeletons", args.slovo_skeletons,
                "--csv", args.slovo_no_event_csv,
                "--split", "val",
                "--out", str(slovo_noev_val_dir),
                "--num_workers", str(args.slovo_num_workers),
            ] + (["--prefer_pp"] if args.prefer_pp else ["--no_prefer_pp"])
        )

        print(f"[build-dataset] step1 ipn train -> {ipn_train_dir}", flush=True)
        ipn_prelabel.main(
            [
                "--config", args.ipn_train_config,
                "--manifest", args.ipn_manifest,
                "--segments_dir", args.ipn_segments_dir,
                "--out_dir", str(ipn_train_dir),
                "--split", "train",
            ] + (["--prefer_pp"] if args.prefer_pp else ["--no_prefer_pp"])
        )
        print(f"[build-dataset] step1 ipn val -> {ipn_val_dir}", flush=True)
        ipn_prelabel.main(
            [
                "--config", args.ipn_val_config,
                "--manifest", args.ipn_manifest,
                "--segments_dir", args.ipn_segments_dir,
                "--out_dir", str(ipn_val_dir),
                "--split", "val",
            ] + (["--prefer_pp"] if args.prefer_pp else ["--no_prefer_pp"])
        )

        synth_common = [
            "--seq_len", str(args.seq_len),
            "--shard_size", str(args.shard_size),
            "--seed", str(args.seed),
            "--include_sign_tails_as_noev" if args.include_sign_tails_as_noev else "--no_include_sign_tails_as_noev",
            "--min_tail_len", str(args.min_tail_len),
            "--primary_noev_prob", str(args.primary_noev_prob),
            "--source_sampling", args.source_sampling,
            "--sign_sampling", args.sign_sampling,
            "--stitch_noev_chunks" if args.stitch_noev_chunks else "--no_stitch_noev_chunks",
            "--overlap_report", str(overlap_path),
        ]

        print(f"[build-dataset] step2 synth train -> {synth_train_dir}", flush=True)
        synth_build.main(
            [
                "--config", args.synth_train_config,
                "--prelabel_dir", str(slovo_train_dir),
                "--preferred_noev_prelabel_dir", str(slovo_noev_train_dir),
                "--extra_noev_prelabel_dir", str(ipn_train_dir),
                "--out_dir", str(synth_train_dir),
                "--num_samples", str(args.train_num_samples),
            ]
            + synth_common
        )
        print(f"[build-dataset] step2 synth val -> {synth_val_dir}", flush=True)
        synth_build.main(
            [
                "--config", args.synth_val_config,
                "--prelabel_dir", str(slovo_val_dir),
                "--preferred_noev_prelabel_dir", str(slovo_noev_val_dir),
                "--extra_noev_prelabel_dir", str(ipn_val_dir),
                "--out_dir", str(synth_val_dir),
                "--num_samples", str(args.val_num_samples),
            ]
            + synth_common
        )

        summary = {
            "out_root": str(out_root),
            "ipn_out_root": str(ipn_out_root),
            "staged_transactionally": True,
            "split_policy": dict(STRICT_SPLIT_POLICY),
            "overlap_report": overlap_report,
            "artifacts": {
                "prelabels_slovo_train": _load_summary(slovo_train_dir / "summary.json"),
                "prelabels_slovo_val": _load_summary(slovo_val_dir / "summary.json"),
                "prelabels_slovo_noev_train": _load_summary(slovo_noev_train_dir / "summary.json"),
                "prelabels_slovo_noev_val": _load_summary(slovo_noev_val_dir / "summary.json"),
                "ipn_prelabels_train": _load_summary(ipn_train_dir / "summary.json"),
                "ipn_prelabels_val": _load_summary(ipn_val_dir / "summary.json"),
                "synth_train": _load_summary(synth_train_dir / "stats.json"),
                "synth_val": _load_summary(synth_val_dir / "stats.json"),
            },
        }
        summary_path = stage_out_root / "build_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
        write_dataset_manifest(
            stage_out_root,
            stage="build_dataset",
            args=args,
            section="build_dataset",
            inputs={
                "slovo_skeletons": str(Path(args.slovo_skeletons).resolve()),
                "slovo_csv": str(Path(args.slovo_csv).resolve()),
                "slovo_no_event_csv": str(Path(args.slovo_no_event_csv).resolve()),
                "ipn_manifest": str(Path(args.ipn_manifest).resolve()),
                "ipn_segments_dir": str(Path(args.ipn_segments_dir).resolve()),
            },
            counts={
                "strict_overlap_ok": bool(overlap_report.get("ok", False)),
            },
            extra={"summary": summary},
        )

        required_paths = [
            stage_out_root / "overlap_report.json",
            stage_out_root / "build_summary.json",
            stage_out_root / "dataset_manifest.json",
            slovo_train_dir / "index.json",
            slovo_val_dir / "index.json",
            slovo_noev_train_dir / "index.json",
            slovo_noev_val_dir / "index.json",
            synth_train_dir / "stats.json",
            synth_val_dir / "stats.json",
            ipn_train_dir / "index.json",
            ipn_val_dir / "index.json",
        ]
        missing_paths = [str(p) for p in required_paths if not p.exists()]
        if missing_paths:
            raise RuntimeError(f"Atomic build validation failed. Missing artifacts: {missing_paths}")

        print(f"[build-dataset] promote -> {out_root}", flush=True)
        _promote_many_transactional(
            [
                (ipn_train_dir, ipn_out_root / "prelabels_train"),
                (ipn_val_dir, ipn_out_root / "prelabels_val"),
                (stage_out_root, out_root),
            ]
        )
        print(json.dumps(summary, ensure_ascii=True, indent=2))
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


if __name__ == "__main__":
    main()
