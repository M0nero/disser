#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _safe_close():
    plt.tight_layout()
    plt.close()


def _plot_msagcn_learning_curves(history: List[Dict[str, float]], out_dir: Path) -> Dict[str, float]:
    history = sorted(history, key=lambda x: int(x["epoch"]))
    epochs = [int(h["epoch"]) for h in history]
    train_loss = [float(h["train_loss"]) for h in history]
    val_loss = [float(h["val_loss"]) for h in history]
    val_acc = [float(h["val_acc"]) for h in history]
    val_f1 = [float(h["val_f1"]) for h in history]

    best_idx = int(np.argmax(np.array(val_f1)))
    best_epoch = epochs[best_idx]
    best_acc = val_acc[best_idx]
    best_f1 = val_f1[best_idx]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, label="Train loss")
    plt.plot(epochs, val_loss, label="Val loss")
    plt.axvline(best_epoch, linestyle="--", alpha=0.4, color="tab:red", label=f"Best epoch={best_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MSA-GCN: training and validation loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(out_dir / "msagcn_learning_curves_loss.png", dpi=220)
    _safe_close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, val_acc, label="Val accuracy (Top-1)")
    plt.plot(epochs, val_f1, label="Val macro-F1")
    plt.scatter([best_epoch], [best_acc], color="tab:blue", s=35, zorder=5)
    plt.scatter([best_epoch], [best_f1], color="tab:orange", s=35, zorder=5)
    plt.axvline(best_epoch, linestyle="--", alpha=0.4, color="tab:red", label=f"Best epoch={best_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("MSA-GCN: validation metrics")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(out_dir / "msagcn_learning_curves_val_metrics.png", dpi=220)
    _safe_close()

    return {
        "best_epoch": best_epoch,
        "best_val_acc": best_acc,
        "best_val_f1": best_f1,
    }


def _parse_report_epoch(path: Path) -> int:
    m = re.search(r"report_ep(\d+)\.json$", path.name)
    if not m:
        raise ValueError(f"Unexpected report file name: {path.name}")
    return int(m.group(1))


def _select_nearest_report(report_dir: Path, target_epoch: int) -> Tuple[int, Path]:
    reports = sorted(report_dir.glob("report_ep*.json"))
    if not reports:
        raise FileNotFoundError(f"No report_ep*.json found in {report_dir}")
    pairs = [(_parse_report_epoch(p), p) for p in reports]
    pairs.sort(key=lambda ep_path: (abs(ep_path[0] - target_epoch), ep_path[0]))
    return pairs[0]


def _extract_f1_by_class(report: Dict[str, object]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for k, v in report.items():
        if k in ("accuracy", "macro avg", "weighted avg"):
            continue
        if not str(k).isdigit():
            continue
        if not isinstance(v, dict):
            continue
        out[int(k)] = float(v.get("f1-score", 0.0))
    return out


def _plot_msagcn_f1_distribution(
    f1_by_idx: Dict[int, float],
    idx2label: Dict[int, str],
    report_epoch: int,
    out_dir: Path,
) -> None:
    if not f1_by_idx:
        return

    vals = np.array(list(f1_by_idx.values()), dtype=float)
    bins = np.linspace(0.0, 1.0, 21)

    plt.figure(figsize=(8.5, 5))
    plt.hist(vals, bins=bins, edgecolor="black")
    plt.xlabel("Per-class F1")
    plt.ylabel("Classes")
    plt.title(f"MSA-GCN: per-class F1 histogram (report_ep{report_epoch:03d})")
    plt.grid(True, axis="y", alpha=0.3)
    plt.savefig(out_dir / f"msagcn_f1_histogram_ep{report_epoch:03d}.png", dpi=220)
    _safe_close()

    sorted_vals = np.sort(vals)
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(sorted_vals)), sorted_vals)
    plt.xlabel("Class rank (sorted by F1)")
    plt.ylabel("F1")
    plt.title(f"MSA-GCN: per-class F1 sorted curve (report_ep{report_epoch:03d})")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / f"msagcn_f1_sorted_curve_ep{report_epoch:03d}.png", dpi=220)
    _safe_close()

    items = sorted(f1_by_idx.items(), key=lambda x: x[1])
    bottom = items[:20]
    top = items[-20:]

    def _barh(data: List[Tuple[int, float]], title: str, fname: str):
        data = sorted(data, key=lambda x: x[1])
        labels = [idx2label.get(idx, str(idx)) for idx, _ in data]
        scores = [score for _, score in data]
        y = np.arange(len(labels))
        fig_h = max(5.0, 0.35 * len(labels) + 1.0)
        plt.figure(figsize=(11, fig_h))
        plt.barh(y, scores)
        plt.yticks(y, labels, fontsize=8)
        plt.xlim(0.0, 1.0)
        plt.xlabel("F1")
        plt.title(title)
        for i, s in enumerate(scores):
            plt.text(min(0.985, s + 0.01), i, f"{s:.2f}", va="center", fontsize=7)
        plt.savefig(out_dir / fname, dpi=220)
        _safe_close()

    _barh(
        bottom,
        f"MSA-GCN: bottom-20 classes by F1 (report_ep{report_epoch:03d})",
        f"msagcn_bottom20_f1_ep{report_epoch:03d}.png",
    )
    _barh(
        top,
        f"MSA-GCN: top-20 classes by F1 (report_ep{report_epoch:03d})",
        f"msagcn_top20_f1_ep{report_epoch:03d}.png",
    )


def _plot_bio_curves(records: List[Dict[str, object]], out_dir: Path) -> Dict[str, float]:
    train = [r for r in records if r.get("event") == "train_step"]
    val = [r for r in records if r.get("event") == "val_epoch"]

    train = sorted(train, key=lambda x: int(x["step"]))
    val = sorted(val, key=lambda x: int(x["epoch"]))

    tr_step = [int(x["step"]) for x in train]
    tr_loss = [float(x["loss"]) for x in train]
    tr_acc = [float(x["acc"]) for x in train]
    tr_f1m = [float(x.get("f1_macro", 0.0)) for x in train]
    tr_bf1 = [float(x.get("b_f1_tol", 0.0)) for x in train]

    ve = [int(x["epoch"]) for x in val]
    vl = [float(x["loss"]) for x in val]
    va = [float(x["acc"]) for x in val]
    vf1_o = [float(x.get("f1_O", 0.0)) for x in val]
    vf1_b = [float(x.get("f1_B", 0.0)) for x in val]
    vf1_i = [float(x.get("f1_I", 0.0)) for x in val]
    vbf1 = [float(x.get("b_f1_tol", 0.0)) for x in val]
    vf1_macro = [float((o + b + i) / 3.0) for o, b, i in zip(vf1_o, vf1_b, vf1_i)]

    best_idx = int(np.argmax(np.array(vf1_b)))
    best_epoch = ve[best_idx]

    plt.figure(figsize=(11, 7))
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(tr_step, tr_loss, label="Train loss")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("BIO tagger: train loss by step")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(tr_step, tr_acc, label="Train acc")
    ax2.plot(tr_step, tr_f1m, label="Train macro-F1")
    ax2.plot(tr_step, tr_bf1, label="Train boundary-F1 (tol)")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Score")
    ax2.set_title("BIO tagger: train metrics by step")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.savefig(out_dir / "bio_train_step_curves.png", dpi=220)
    _safe_close()

    plt.figure(figsize=(11, 6))
    plt.plot(ve, vf1_o, label="Val F1_O")
    plt.plot(ve, vf1_b, label="Val F1_B")
    plt.plot(ve, vf1_i, label="Val F1_I")
    plt.plot(ve, vf1_macro, label="Val macro-F1", linestyle="--")
    plt.axvline(best_epoch, linestyle="--", alpha=0.35, color="tab:red", label=f"Best F1_B epoch={best_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title("BIO tagger: validation class F1 by epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(out_dir / "bio_val_class_f1_by_epoch.png", dpi=220)
    _safe_close()

    plt.figure(figsize=(11, 6))
    plt.plot(ve, va, label="Val accuracy")
    plt.plot(ve, vbf1, label="Val boundary-F1 (tol)")
    plt.plot(ve, vl, label="Val loss")
    plt.axvline(best_epoch, linestyle="--", alpha=0.35, color="tab:red", label=f"Best F1_B epoch={best_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("Score / Loss")
    plt.title("BIO tagger: validation overview")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(out_dir / "bio_val_overview_by_epoch.png", dpi=220)
    _safe_close()

    return {
        "best_epoch_by_f1_B": int(ve[best_idx]),
        "best_f1_B": float(vf1_b[best_idx]),
        "f1_I_at_best_f1_B": float(vf1_i[best_idx]),
        "f1_O_at_best_f1_B": float(vf1_o[best_idx]),
        "macro_f1_at_best_f1_B": float(vf1_macro[best_idx]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate report-ready figures for BIO and MSA-GCN runs.")
    ap.add_argument(
        "--bio-log",
        type=Path,
        default=Path("outputs/runs/bio_gru_v1/train_log.jsonl"),
        help="BIO jsonl training log",
    )
    ap.add_argument(
        "--msagcn-history",
        type=Path,
        default=Path("datasets/skeletons_new_crop/history.json"),
        help="MSA-GCN history.json",
    )
    ap.add_argument(
        "--msagcn-report-dir",
        type=Path,
        default=Path("datasets/skeletons_new_crop"),
        help="Directory with report_ep*.json",
    )
    ap.add_argument(
        "--msagcn-label2idx",
        type=Path,
        default=Path("datasets/skeletons_new_crop/label2idx.json"),
        help="label2idx.json for class names",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/report_figures/training_refresh_2026-02-24"),
        help="Output directory",
    )
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # MSA-GCN plots
    msagcn_history = _load_json(args.msagcn_history)
    msagcn_summary = _plot_msagcn_learning_curves(msagcn_history, out_dir)
    best_epoch = int(msagcn_summary["best_epoch"])
    nearest_report_epoch, nearest_report_path = _select_nearest_report(args.msagcn_report_dir, best_epoch)
    report = _load_json(nearest_report_path)
    f1_by_idx = _extract_f1_by_class(report)

    idx2label: Dict[int, str] = {}
    if args.msagcn_label2idx.exists():
        label2idx = _load_json(args.msagcn_label2idx)
        idx2label = {int(v): str(k) for k, v in label2idx.items()}
    _plot_msagcn_f1_distribution(f1_by_idx, idx2label, nearest_report_epoch, out_dir)
    msagcn_summary["nearest_report_epoch"] = nearest_report_epoch
    msagcn_summary["nearest_report_file"] = str(nearest_report_path)

    # BIO plots
    bio_records = _load_jsonl(args.bio_log)
    bio_summary = _plot_bio_curves(bio_records, out_dir)

    summary = {
        "bio": bio_summary,
        "msagcn": msagcn_summary,
        "inputs": {
            "bio_log": str(args.bio_log),
            "msagcn_history": str(args.msagcn_history),
            "msagcn_report_dir": str(args.msagcn_report_dir),
            "msagcn_label2idx": str(args.msagcn_label2idx),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved figures to: {out_dir}")
    print(f"Summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
