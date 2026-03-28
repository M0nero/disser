from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from bio.pipeline import train


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Two-stage BIO curriculum training: warmup_single_sign -> main_continuous")
    ap.add_argument("--train_warmup_dir", type=str, required=True)
    ap.add_argument("--val_warmup_dir", type=str, required=True)
    ap.add_argument("--train_dir", type=str, required=True)
    ap.add_argument("--val_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True, help="Top-level curriculum run directory.")
    ap.add_argument("--config", type=str, default="bio/configs/bio_default.json")
    ap.add_argument("--warmup_epochs", type=int, default=12)
    ap.add_argument("--main_epochs", type=int, default=20)
    ap.add_argument("--warmup_lr", type=float, default=0.002)
    ap.add_argument("--main_lr", type=float, default=0.0008)
    ap.add_argument("--allow_bad_synth_stats", action="store_true", default=False)
    ap.add_argument("--tensorboard", action="store_true", default=False)
    ap.add_argument("--save_analysis_artifacts", action="store_true", default=False)
    ap.add_argument("--logdir", type=str, default="runs")
    return ap.parse_args(argv)


def _run_train_stage(
    *,
    dataset_train: Path,
    dataset_val: Path,
    out_dir: Path,
    config: str,
    epochs: int,
    lr: float,
    tensorboard: bool,
    save_analysis_artifacts: bool,
    logdir: str,
    allow_bad_synth_stats: bool,
    resume: Path | None = None,
    resume_model_only: bool = False,
) -> None:
    args = [
        "--train_dir", str(dataset_train),
        "--val_dir", str(dataset_val),
        "--out_dir", str(out_dir),
        "--config", str(config),
        "--epochs", str(int(epochs)),
        "--lr", str(float(lr)),
        "--logdir", str(logdir),
    ]
    if tensorboard:
        args.append("--tensorboard")
    if save_analysis_artifacts:
        args.append("--save_analysis_artifacts")
    if allow_bad_synth_stats:
        args.append("--allow_bad_synth_stats")
    if resume is not None:
        args.extend(["--resume", str(resume)])
    if resume_model_only:
        args.append("--resume_model_only")
    train.main(args)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    warmup_out = out_root / "warmup"
    main_out = out_root / "main"

    _run_train_stage(
        dataset_train=Path(args.train_warmup_dir),
        dataset_val=Path(args.val_warmup_dir),
        out_dir=warmup_out,
        config=str(args.config),
        epochs=int(args.warmup_epochs),
        lr=float(args.warmup_lr),
        tensorboard=bool(args.tensorboard),
        save_analysis_artifacts=bool(args.save_analysis_artifacts),
        logdir=str(args.logdir),
        allow_bad_synth_stats=bool(args.allow_bad_synth_stats),
    )

    resume_path = warmup_out / "best_recall_safe.pt"
    if not resume_path.exists():
        raise FileNotFoundError(f"Warmup stage completed without best_recall_safe.pt: {resume_path}")

    _run_train_stage(
        dataset_train=Path(args.train_dir),
        dataset_val=Path(args.val_dir),
        out_dir=main_out,
        config=str(args.config),
        epochs=int(args.main_epochs),
        lr=float(args.main_lr),
        tensorboard=bool(args.tensorboard),
        save_analysis_artifacts=bool(args.save_analysis_artifacts),
        logdir=str(args.logdir),
        allow_bad_synth_stats=bool(args.allow_bad_synth_stats),
        resume=resume_path,
        resume_model_only=True,
    )

    summary = {
        "warmup": {
            "train_dir": str(Path(args.train_warmup_dir).resolve()),
            "val_dir": str(Path(args.val_warmup_dir).resolve()),
            "out_dir": str(warmup_out),
            "epochs": int(args.warmup_epochs),
            "lr": float(args.warmup_lr),
            "checkpoint": str(resume_path),
        },
        "main": {
            "train_dir": str(Path(args.train_dir).resolve()),
            "val_dir": str(Path(args.val_dir).resolve()),
            "out_dir": str(main_out),
            "epochs": int(args.main_epochs),
            "lr": float(args.main_lr),
            "resume_model_only": True,
        },
    }
    (out_root / "curriculum_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
