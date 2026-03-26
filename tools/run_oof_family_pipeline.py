from __future__ import annotations

import argparse
import multiprocessing
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the full train-only OOF family pipeline in one command.")
    p.add_argument("--json", required=True, type=str)
    p.add_argument("--csv", required=True, type=str)
    p.add_argument("--ckpt", required=True, type=str)
    p.add_argument("--out_root", required=True, type=str)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--fold_epochs", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=-1)
    p.add_argument("--batch", type=int, default=0)
    p.add_argument("--array_shard_size", type=int, default=4096)
    p.add_argument("--save_full_logits", action="store_true")
    p.add_argument("--limit_train_batches", type=int, default=0)
    p.add_argument("--limit_holdout_batches", type=int, default=0)
    p.add_argument("--num_families", type=int, default=128)
    p.add_argument("--auto_num_families", action="store_true")
    p.add_argument("--auto_num_families_candidates", type=str, default="")
    p.add_argument("--proto_weight", type=float, default=0.50)
    p.add_argument("--conf_weight", type=float, default=0.30)
    p.add_argument("--kin_weight", type=float, default=0.20)
    p.add_argument("--top_neighbors", type=int, default=10)
    p.add_argument("--skip_finetune", action="store_true")
    p.add_argument("--finetune_out", type=str, default="")
    p.add_argument("--finetune_epochs", type=int, default=0, help="Override epochs for the family fine-tune stage (0 = keep checkpoint args)")
    p.add_argument("--finetune_limit_train_batches", type=int, default=0)
    p.add_argument("--finetune_limit_val_batches", type=int, default=0)
    p.add_argument("--family_loss_weight", type=float, default=0.25)
    p.add_argument("--family_warmup_epochs", type=int, default=0)
    p.add_argument("--family_eval", action="store_true")
    p.add_argument("--head_only_rebalance_epochs", type=int, default=0)
    p.add_argument("--head_only_rebalance_lr", type=float, default=1e-4)
    p.add_argument("--auto_head_only_rebalance_stop", action="store_true")
    p.add_argument("--head_only_rebalance_use_logit_adjustment", action="store_true")
    p.add_argument("--head_only_rebalance_weighted_sampler", action="store_true")
    p.add_argument("--auto_workers", action="store_true")
    return p.parse_args()


def _run_subprocess(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def main() -> None:
    multiprocessing.freeze_support()
    args = parse_args()
    from msagcn.training import parse_args as parse_train_args
    from msagcn.training import run_training
    from msagcn.training.oof_utils import load_checkpoint_training_state

    out_root = Path(args.out_root).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)
    oof_out = out_root / "oof_cache"
    family_map_path = out_root / "family_map.json"
    finetune_out = Path(args.finetune_out).expanduser() if args.finetune_out else (out_root / "family_finetune")

    oof_cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "build_oof_cache.py"),
        "--json",
        str(args.json),
        "--csv",
        str(args.csv),
        "--ckpt",
        str(args.ckpt),
        "--out",
        str(oof_out),
        "--folds",
        str(int(args.folds)),
        "--fold_epochs",
        str(int(args.fold_epochs)),
        "--seed",
        str(int(args.seed)),
        "--array_shard_size",
        str(int(args.array_shard_size)),
    ]
    if int(args.workers) >= 0:
        oof_cmd += ["--workers", str(int(args.workers))]
    if int(args.batch) > 0:
        oof_cmd += ["--batch", str(int(args.batch))]
    if bool(args.save_full_logits):
        oof_cmd.append("--save_full_logits")
    if int(args.limit_train_batches) > 0:
        oof_cmd += ["--limit_train_batches", str(int(args.limit_train_batches))]
    if int(args.limit_holdout_batches) > 0:
        oof_cmd += ["--limit_holdout_batches", str(int(args.limit_holdout_batches))]
    _run_subprocess(oof_cmd)

    family_cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "build_family_map.py"),
        "--oof_dir",
        str(oof_out),
        "--out",
        str(family_map_path),
        "--proto_weight",
        str(float(args.proto_weight)),
        "--conf_weight",
        str(float(args.conf_weight)),
        "--kin_weight",
        str(float(args.kin_weight)),
        "--top_neighbors",
        str(int(args.top_neighbors)),
    ]
    if bool(args.auto_num_families):
        family_cmd.append("--auto_num_families")
        if str(args.auto_num_families_candidates).strip():
            family_cmd += ["--auto_num_families_candidates", str(args.auto_num_families_candidates)]
    else:
        family_cmd += ["--num_families", str(int(args.num_families))]
    _run_subprocess(family_cmd)

    if args.skip_finetune:
        print(f"Skipping family fine-tune stage. family_map at {family_map_path}")
        return

    ckpt_state = load_checkpoint_training_state(args.ckpt)
    train_args = parse_train_args(["--json", str(args.json), "--csv", str(args.csv), "--out", str(finetune_out)])
    for key, value in ckpt_state["args"].items():
        setattr(train_args, key, value)

    train_args.json = str(args.json)
    train_args.csv = str(args.csv)
    train_args.out = str(finetune_out)
    train_args.resume = str(args.ckpt)
    train_args.resume_model_only = True
    train_args.use_family_head = True
    train_args.family_map = str(family_map_path)
    train_args.family_loss_weight = float(args.family_loss_weight)
    train_args.family_warmup_epochs = int(args.family_warmup_epochs)
    train_args.family_eval = bool(args.family_eval)
    train_args.head_only_rebalance_epochs = int(args.head_only_rebalance_epochs)
    train_args.head_only_rebalance_lr = float(args.head_only_rebalance_lr)
    train_args.auto_head_only_rebalance_stop = bool(args.auto_head_only_rebalance_stop)
    train_args.head_only_rebalance_use_logit_adjustment = bool(args.head_only_rebalance_use_logit_adjustment)
    train_args.head_only_rebalance_weighted_sampler = bool(args.head_only_rebalance_weighted_sampler)
    train_args.auto_workers = bool(args.auto_workers)
    train_args.limit_train_batches = int(args.finetune_limit_train_batches)
    train_args.limit_val_batches = int(args.finetune_limit_val_batches)
    if int(args.finetune_epochs) > 0:
        train_args.epochs = int(args.finetune_epochs)
    if int(args.workers) >= 0:
        train_args.workers = int(args.workers)
    if int(args.batch) > 0:
        train_args.batch = int(args.batch)

    print(f"$ family fine-tune -> {finetune_out}")
    run_training(train_args)


if __name__ == "__main__":
    main()
