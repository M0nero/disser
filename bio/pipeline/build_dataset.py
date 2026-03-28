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
from bio.pipeline import continuous_stats
from bio.pipeline import prelabel as slovo_prelabel
from bio.pipeline import synth_build

STRICT_SPLIT_POLICY = {
    "mode": "strict_source_group_separation",
    "description": "The same raw Slovo video/source_group must not appear in multiple splits. Any overlap blocks the build.",
    "enforced_on": ["slovo_sign", "slovo_no_event", "slovo_union"],
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
    ipn_manifest: Path | None = None,
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

    report = {
        "split_policy": dict(STRICT_SPLIT_POLICY),
        "slovo_sign": slovo_sign_report,
        "slovo_no_event": slovo_noev_report,
        "slovo_union": slovo_union_report,
    }
    if ipn_manifest is not None and Path(ipn_manifest).exists():
        manifest_rows = _read_manifest(Path(ipn_manifest))
        ipn_train_ids = [str(r.get("video_id") or "") for r in manifest_rows if _norm_split(r.get("split")) == "train"]
        ipn_val_ids = [str(r.get("video_id") or "") for r in manifest_rows if _norm_split(r.get("split")) == "val"]
        report["ipn_hand"] = _audit_overlap(ipn_train_ids, ipn_val_ids, source_name="ipn_hand")
    report["ok"] = all(
        int(item.get("overlap_count", 0)) == 0
        for key, item in report.items()
        if isinstance(item, dict) and key != "split_policy"
    )
    return report


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Canonical BIO Slovo-only trimmed rebuild.")
    ap.add_argument("--out_root", type=str, default="outputs/bio_out_v2", help="Root directory for rebuilt BIO artifacts.")
    ap.add_argument("--slovo_skeletons", type=str, default="datasets/skeletons/Slovo")
    ap.add_argument("--slovo_csv", type=str, default="datasets/data/annotations.csv")
    ap.add_argument("--slovo_no_event_csv", type=str, default="datasets/data/annotations_no_event.csv")

    ap.add_argument("--slovo_prelabel_config", type=str, default="bio/configs/bio_default.json")
    ap.add_argument("--synth_train_config", type=str, default="bio/configs/bio_default.json")
    ap.add_argument("--synth_val_config", type=str, default="bio/configs/bio_val.json")

    ap.add_argument("--slovo_num_workers", type=int, default=8)
    ap.add_argument("--synth_workers", type=int, default=0, help="Optional override for synth-build worker processes. 0 = use synth auto-workers.")
    ap.add_argument("--train_num_samples", type=int, default=100000)
    ap.add_argument("--val_num_samples", type=int, default=10000)
    ap.add_argument("--warmup_train_num_samples", type=int, default=50000)
    ap.add_argument("--warmup_val_num_samples", type=int, default=5000)
    ap.add_argument("--stress_train_num_samples", type=int, default=20000)
    ap.add_argument("--stress_val_num_samples", type=int, default=2000)
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
    ap.add_argument("--emit_warmup_dataset", dest="emit_warmup_dataset", action="store_true", default=False)
    ap.add_argument("--no_emit_warmup_dataset", dest="emit_warmup_dataset", action="store_false")
    ap.add_argument("--dense_signer_min_clips", type=int, default=8)
    ap.add_argument("--real_session_dir", action="append", default=[], help="Review session directory used to extract runtime empirical continuous_stats.")
    ap.add_argument("--emit_stress_dataset", dest="emit_stress_dataset", action="store_true", default=False)
    ap.add_argument("--no_emit_stress_dataset", dest="emit_stress_dataset", action="store_false")
    ap.add_argument("--allow_prelabel_empirical_fallback", action="store_true", default=False)
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    real_session_dirs = list(dict.fromkeys(Path(x).resolve() for x in (args.real_session_dir or []) if str(x).strip()))
    out_root = Path(args.out_root).resolve()
    out_root.parent.mkdir(parents=True, exist_ok=True)

    stage_root = Path(
        tempfile.mkdtemp(
            prefix=f".{out_root.name}.build_tmp_",
            dir=str(out_root.parent),
        )
    )
    build_succeeded = False
    try:
        stage_out_root = stage_root / out_root.name
        stage_out_root.mkdir(parents=True, exist_ok=True)
        write_run_config(stage_out_root, args, section="build_dataset")

        overlap_report = build_overlap_report(
            Path(args.slovo_csv),
            Path(args.slovo_no_event_csv),
        )
        overlap_path = stage_out_root / "overlap_report.json"
        overlap_path.write_text(json.dumps(overlap_report, ensure_ascii=True, indent=2), encoding="utf-8")
        if not bool(overlap_report.get("ok", False)):
            raise RuntimeError(f"Split overlap detected. See {overlap_path}")

        slovo_train_dir = stage_out_root / "prelabels_slovo_train"
        slovo_val_dir = stage_out_root / "prelabels_slovo_val"
        slovo_noev_train_dir = stage_out_root / "prelabels_slovo_noev_train"
        slovo_noev_val_dir = stage_out_root / "prelabels_slovo_noev_val"
        synth_train_warmup_dir = stage_out_root / "synth_train_warmup"
        synth_val_warmup_dir = stage_out_root / "synth_val_warmup"
        synth_train_dir = stage_out_root / "synth_train"
        synth_val_dir = stage_out_root / "synth_val"
        synth_train_stress_dir = stage_out_root / "synth_train_stress"
        synth_val_stress_dir = stage_out_root / "synth_val_stress"
        continuous_stats_dir = stage_out_root / "continuous_stats"

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

        continuous_stats_summary: Dict[str, Any] = {}
        synth_sampling_profile = "runtime_empirical"
        synth_continuous_stats_dir = continuous_stats_dir
        if real_session_dirs:
            print(f"[build-dataset] empirical continuous stats -> {continuous_stats_dir}", flush=True)
            continuous_stats_dir.mkdir(parents=True, exist_ok=True)
            payload = continuous_stats.build_continuous_stats(
                session_dirs=real_session_dirs,
                prelabel_dirs=[],
                motion_epsilon=0.01,
            )
            continuous_stats_path = continuous_stats_dir / "continuous_stats.json"
            continuous_stats_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            continuous_stats_summary = payload
        else:
            synth_sampling_profile = "prelabel_empirical"
            synth_continuous_stats_dir = None

        synth_common = [
            "--seq_len", str(args.seq_len),
            "--shard_size", str(args.shard_size),
            "--seed", str(args.seed),
            "--include_sign_tails_as_noev" if args.include_sign_tails_as_noev else "--no_include_sign_tails_as_noev",
            "--min_tail_len", str(args.min_tail_len),
            "--primary_noev_prob", str(args.primary_noev_prob),
            "--source_sampling", args.source_sampling,
            "--sign_sampling", args.sign_sampling,
            "--dataset_profile", "main_continuous",
            "--dense_signer_min_clips", str(args.dense_signer_min_clips),
            "--sampling_profile", synth_sampling_profile,
            "--stitch_noev_chunks" if args.stitch_noev_chunks else "--no_stitch_noev_chunks",
            "--overlap_report", str(overlap_path),
        ]
        if synth_continuous_stats_dir is not None:
            synth_common.extend(["--continuous_stats_dir", str(synth_continuous_stats_dir)])
        if int(args.synth_workers) > 0:
            synth_common.extend(["--workers", str(args.synth_workers), "--no_auto_workers"])
        else:
            synth_common.append("--auto_workers")

        if bool(args.emit_warmup_dataset):
            warmup_common = [
                "--seq_len", str(args.seq_len),
                "--shard_size", str(args.shard_size),
                "--seed", str(args.seed),
                "--include_sign_tails_as_noev" if args.include_sign_tails_as_noev else "--no_include_sign_tails_as_noev",
                "--min_tail_len", str(args.min_tail_len),
                "--primary_noev_prob", str(args.primary_noev_prob),
                "--source_sampling", args.source_sampling,
                "--sign_sampling", args.sign_sampling,
                "--dataset_profile", "warmup_single_sign",
                "--dense_signer_min_clips", str(args.dense_signer_min_clips),
                "--sampling_profile", "prelabel_empirical",
                "--stitch_noev_chunks" if args.stitch_noev_chunks else "--no_stitch_noev_chunks",
                "--overlap_report", str(overlap_path),
            ]
            if int(args.synth_workers) > 0:
                warmup_common.extend(["--workers", str(args.synth_workers), "--no_auto_workers"])
            else:
                warmup_common.append("--auto_workers")

            print(f"[build-dataset] step2 synth train warmup -> {synth_train_warmup_dir}", flush=True)
            synth_build.main(
                [
                    "--config", args.synth_train_config,
                    "--prelabel_dir", str(slovo_train_dir),
                    "--preferred_noev_prelabel_dir", str(slovo_noev_train_dir),
                    "--out_dir", str(synth_train_warmup_dir),
                    "--num_samples", str(args.warmup_train_num_samples),
                ] + warmup_common
            )
            print(f"[build-dataset] step2 synth val warmup -> {synth_val_warmup_dir}", flush=True)
            synth_build.main(
                [
                    "--config", args.synth_val_config,
                    "--prelabel_dir", str(slovo_val_dir),
                    "--preferred_noev_prelabel_dir", str(slovo_noev_val_dir),
                    "--out_dir", str(synth_val_warmup_dir),
                    "--num_samples", str(args.warmup_val_num_samples),
                ] + warmup_common
            )

        print(f"[build-dataset] step2 synth train -> {synth_train_dir}", flush=True)
        synth_build.main(
            [
                "--config", args.synth_train_config,
                "--prelabel_dir", str(slovo_train_dir),
                "--preferred_noev_prelabel_dir", str(slovo_noev_train_dir),
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
                "--out_dir", str(synth_val_dir),
                "--num_samples", str(args.val_num_samples),
            ]
            + synth_common
        )

        if bool(args.emit_stress_dataset):
            stress_common = [
                "--seq_len", str(args.seq_len),
                "--shard_size", str(args.shard_size),
                "--seed", str(args.seed),
                "--include_sign_tails_as_noev" if args.include_sign_tails_as_noev else "--no_include_sign_tails_as_noev",
                "--min_tail_len", str(args.min_tail_len),
                "--primary_noev_prob", str(args.primary_noev_prob),
                "--source_sampling", args.source_sampling,
                "--sign_sampling", args.sign_sampling,
                "--dataset_profile", "stress",
                "--sampling_profile", synth_sampling_profile,
                "--stitch_noev_chunks" if args.stitch_noev_chunks else "--no_stitch_noev_chunks",
                "--overlap_report", str(overlap_path),
            ]
            if synth_continuous_stats_dir is not None:
                stress_common.extend(["--continuous_stats_dir", str(synth_continuous_stats_dir)])
            if int(args.synth_workers) > 0:
                stress_common.extend(["--workers", str(args.synth_workers), "--no_auto_workers"])
            else:
                stress_common.append("--auto_workers")

            print(f"[build-dataset] step2 synth train stress -> {synth_train_stress_dir}", flush=True)
            synth_build.main(
                [
                    "--config", args.synth_train_config,
                    "--prelabel_dir", str(slovo_train_dir),
                    "--preferred_noev_prelabel_dir", str(slovo_noev_train_dir),
                    "--out_dir", str(synth_train_stress_dir),
                    "--num_samples", str(args.stress_train_num_samples),
                ]
                + stress_common
            )
            print(f"[build-dataset] step2 synth val stress -> {synth_val_stress_dir}", flush=True)
            synth_build.main(
                [
                    "--config", args.synth_val_config,
                    "--prelabel_dir", str(slovo_val_dir),
                    "--preferred_noev_prelabel_dir", str(slovo_noev_val_dir),
                    "--out_dir", str(synth_val_stress_dir),
                    "--num_samples", str(args.stress_val_num_samples),
                ]
                + stress_common
            )

        summary = {
            "out_root": str(out_root),
            "dataset_scope": "slovo_only",
            "staged_transactionally": True,
            "split_policy": dict(STRICT_SPLIT_POLICY),
            "overlap_report": overlap_report,
            "main_dataset_profile": "main_continuous",
            "warmup_dataset_emitted": bool(args.emit_warmup_dataset),
            "stress_dataset_emitted": bool(args.emit_stress_dataset),
            "sampling_profile": synth_sampling_profile,
            "real_session_dirs": [str(x) for x in real_session_dirs],
            "allow_prelabel_empirical_fallback": bool(args.allow_prelabel_empirical_fallback),
            "artifacts": {
                "prelabels_slovo_train": _load_summary(slovo_train_dir / "summary.json"),
                "prelabels_slovo_val": _load_summary(slovo_val_dir / "summary.json"),
                "prelabels_slovo_noev_train": _load_summary(slovo_noev_train_dir / "summary.json"),
                "prelabels_slovo_noev_val": _load_summary(slovo_noev_val_dir / "summary.json"),
                "continuous_stats": continuous_stats_summary,
                "synth_train": _load_summary(synth_train_dir / "stats.json"),
                "synth_val": _load_summary(synth_val_dir / "stats.json"),
            },
        }
        if bool(args.emit_warmup_dataset):
            summary["artifacts"]["synth_train_warmup"] = _load_summary(synth_train_warmup_dir / "stats.json")
            summary["artifacts"]["synth_val_warmup"] = _load_summary(synth_val_warmup_dir / "stats.json")
        if bool(args.emit_stress_dataset):
            summary["artifacts"]["synth_train_stress"] = _load_summary(synth_train_stress_dir / "stats.json")
            summary["artifacts"]["synth_val_stress"] = _load_summary(synth_val_stress_dir / "stats.json")
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
        ]
        if bool(args.emit_warmup_dataset):
            required_paths.extend(
                [
                    synth_train_warmup_dir / "stats.json",
                    synth_val_warmup_dir / "stats.json",
                ]
            )
        if real_session_dirs:
            required_paths.append(continuous_stats_dir / "continuous_stats.json")
        if bool(args.emit_stress_dataset):
            required_paths.extend(
                [
                    synth_train_stress_dir / "stats.json",
                    synth_val_stress_dir / "stats.json",
                ]
            )
        missing_paths = [str(p) for p in required_paths if not p.exists()]
        if missing_paths:
            raise RuntimeError(f"Atomic build validation failed. Missing artifacts: {missing_paths}")

        print(f"[build-dataset] promote -> {out_root}", flush=True)
        _promote_many_transactional(
            [
                (stage_out_root, out_root),
            ]
        )
        build_succeeded = True
        print(json.dumps(summary, ensure_ascii=True, indent=2))
    except Exception:
        print(f"[build-dataset] failed; preserving stage_root={stage_root}", flush=True)
        raise
    finally:
        if build_succeeded:
            shutil.rmtree(stage_root, ignore_errors=True)


if __name__ == "__main__":
    main()
