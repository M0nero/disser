from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix


def _sanitize_label(label: str) -> str:
    return str(label).replace("/", "_").replace(" ", "_")


def _short_label(label: str, max_len: int = 12) -> str:
    label = str(label)
    if len(label) <= max_len:
        return label
    return label[: max_len - 1] + "..."


def build_confusion_image(
    y_true: List[int],
    y_pred: List[int],
    idx2label: Dict[int, str],
    topk: int = 50,
    *,
    normalize: bool = True,
) -> Tuple[np.ndarray, List[int]]:
    if not y_true:
        return np.zeros((1, 1, 3), dtype=np.uint8), []

    num_classes = max(max(y_true), max(y_pred)) + 1
    support = np.bincount(np.array(y_true), minlength=num_classes)
    valid = np.where(support > 0)[0]
    if valid.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8), []
    order = valid[np.argsort(-support[valid])]
    topk = int(max(1, topk))
    topk_idx = order[: min(topk, len(order))]

    mapping = {cls: i for i, cls in enumerate(topk_idx)}
    other_idx = len(topk_idx)

    def map_label(x: int) -> int:
        return mapping.get(x, other_idx)

    y_true_m = [map_label(x) for x in y_true]
    y_pred_m = [map_label(x) for x in y_pred]
    labels = list(range(len(topk_idx) + 1))
    cm = confusion_matrix(y_true_m, y_pred_m, labels=labels)
    cm = cm.astype(np.float32)
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        cm = cm / row_sum

    label_names = [_short_label(idx2label.get(i, str(i))) for i in topk_idx] + ["other"]

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(label_names)))
    ax.set_yticks(np.arange(len(label_names)))
    ax.set_xticklabels(label_names, rotation=90, fontsize=6)
    ax.set_yticklabels(label_names, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion (top-K + other)")
    fig.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()

    # Matplotlib backends differ here: TkAgg may not expose tostring_rgb(),
    # while Agg-like canvases expose buffer_rgba(). Use the most portable path.
    if hasattr(fig.canvas, "buffer_rgba"):
        rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = np.ascontiguousarray(rgba[..., :3])
    elif hasattr(fig.canvas, "tostring_rgb"):
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    elif hasattr(fig.canvas, "tostring_argb"):
        argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
        img = np.ascontiguousarray(argb[..., 1:])
    else:
        raise RuntimeError(f"Unsupported matplotlib canvas for confusion image: {type(fig.canvas).__name__}")

    plt.close(fig)
    return img, topk_idx.tolist()


def format_examples_text(examples, idx2label: Dict[int, str], max_items: int) -> str:
    if not examples:
        return "No examples collected."
    max_items = int(max(1, max_items))
    lines = ["rank\tvideo\tclip\ttrue\tpred\tconf\tcorrect"]
    for i, ex in enumerate(examples[:max_items], start=1):
        video = ex.get("video", "")
        t0 = ex.get("t0", "")
        t1 = ex.get("t1", "")
        clip = f"{t0}-{t1}" if t0 != "" and t1 != "" else ""
        true_idx = int(ex.get("true", -1))
        pred_idx = int(ex.get("pred", -1))
        true_lbl = idx2label.get(true_idx, str(true_idx))
        pred_lbl = idx2label.get(pred_idx, str(pred_idx))
        conf = float(ex.get("conf", 0.0))
        correct = int(true_idx == pred_idx)
        lines.append(f"{i}\t{video}\t{clip}\t{true_lbl}\t{pred_lbl}\t{conf:.3f}\t{correct}")
    return "\n".join(lines)


def push_lowest_conf(examples: List[Dict[str, object]], item: Dict[str, object], max_k: int) -> None:
    if max_k <= 0:
        return
    examples.append(item)
    examples.sort(key=lambda x: float(x.get("conf", 0.0)))
    if len(examples) > max_k:
        del examples[max_k:]
