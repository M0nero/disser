#!/usr/bin/env python
import json
from pathlib import Path

import matplotlib.pyplot as plt

# Пути до файлов относительно самого скрипта
ROOT = Path(__file__).resolve().parent
HISTORY_PATH = ROOT / "history.json"
REPORT_PATH = ROOT / "report_ep055.json"  # берём финальный репорт
LABEL2IDX_PATH = ROOT / "label2idx.json"
PLOTS_DIR = ROOT / "plots"


def load_history(path: Path = HISTORY_PATH):
    with open(path, "r", encoding="utf-8") as f:
        history = json.load(f)

    # На всякий случай сортируем по эпохе
    history = sorted(history, key=lambda x: x["epoch"])
    return history


def plot_learning_curves(history, out_dir: Path = PLOTS_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_acc = [h["val_acc"] for h in history]
    val_f1 = [h["val_f1"] for h in history]

    # Loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, label="Train loss")
    plt.plot(epochs, val_loss, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "learning_curves_loss.png", dpi=200)

    # Validation metrics (accuracy + F1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, val_acc, label="Val accuracy")
    plt.plot(epochs, val_f1, label="Val F1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation accuracy and macro-F1")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "learning_curves_val_metrics.png", dpi=200)


def load_per_class_f1(report_path: Path = REPORT_PATH):
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    f1_by_idx = {}
    for k, v in report.items():
        if k in ("accuracy", "macro avg", "weighted avg"):
            continue
        try:
            idx = int(k)
        except ValueError:
            continue
        f1 = float(v.get("f1-score", 0.0))
        f1_by_idx[idx] = f1

    return f1_by_idx


def plot_f1_distribution(f1_by_idx, out_dir: Path = PLOTS_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)

    values = list(f1_by_idx.values())

    # Histogram
    plt.figure(figsize=(8, 5))
    # бины 0.0–1.0 с шагом 0.1
    bins = [x / 10.0 for x in range(0, 11)]
    plt.hist(values, bins=bins, edgecolor="black")
    plt.xlabel("Per-class F1")
    plt.ylabel("Number of classes")
    plt.title("Distribution of per-class F1 (epoch 55)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "f1_histogram.png", dpi=200)

    # Sorted curve
    sorted_vals = sorted(values)
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(sorted_vals)), sorted_vals)
    plt.xlabel("Class rank (sorted by F1)")
    plt.ylabel("F1")
    plt.title("Per-class F1 (sorted, epoch 55)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "f1_sorted_curve.png", dpi=200)


def load_idx2label(path: Path = LABEL2IDX_PATH):
    with open(path, "r", encoding="utf-8") as f:
        label2idx = json.load(f)

    # инвертируем маппинг: idx -> label
    idx2label = {int(idx): label for label, idx in label2idx.items()}
    return idx2label


def plot_top_bottom_classes(
    f1_by_idx,
    idx2label,
    out_dir: Path = PLOTS_DIR,
    top_n: int = 100,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    items = sorted(f1_by_idx.items(), key=lambda x: x[1])
    bottom = items[:top_n]
    top = items[-top_n:]

    def _plot(items_local, filename: str, title: str):
        # сортируем внутри списка по F1, чтобы на графике шло снизу вверх
        items_local = sorted(items_local, key=lambda x: x[1])
        indices = [idx for idx, _ in items_local]
        scores = [score for _, score in items_local]
        labels = [idx2label.get(idx, str(idx)) for idx in indices]

        # динамическая высота фигуры под количество баров
        height = max(4.0, 0.4 * len(labels) + 1.0)
        plt.figure(figsize=(10, height))
        y_pos = range(len(labels))
        plt.barh(y_pos, scores)
        plt.yticks(y_pos, labels)
        plt.xlabel("F1")
        plt.title(title)
        plt.xlim(0.0, 1.0)

        # подписи значений F1 справа от баров
        for i, v in enumerate(scores):
            plt.text(min(v + 0.01, 0.99), i, f"{v:.2f}", va="center")

        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=200)

    _plot(bottom, "bottom20_f1.png", "Bottom-20 classes by F1 (epoch 55)")
    _plot(top, "top20_f1.png", "Top-20 classes by F1 (epoch 55)")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Learning curves
    history = load_history()
    plot_learning_curves(history, PLOTS_DIR)

    # 2) Per-class F1 distribution
    f1_by_idx = load_per_class_f1()
    plot_f1_distribution(f1_by_idx, PLOTS_DIR)

    # 3) Top/Bottom-20 classes with decoded labels
    idx2label = load_idx2label()
    plot_top_bottom_classes(f1_by_idx, idx2label, PLOTS_DIR)


if __name__ == "__main__":
    main()
