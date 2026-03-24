from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

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
    class_ids: Sequence[int] | None = None,
) -> Tuple[np.ndarray, List[int]]:
    if not y_true:
        return np.zeros((1, 1, 3), dtype=np.uint8), []

    num_classes = max(max(y_true), max(y_pred)) + 1
    support = np.bincount(np.array(y_true), minlength=num_classes)
    if class_ids is not None:
        class_ids = [int(x) for x in class_ids]
        focus = set(class_ids)
        filtered = [(int(t), int(p)) for t, p in zip(y_true, y_pred) if int(t) in focus]
        if not filtered:
            return np.zeros((1, 1, 3), dtype=np.uint8), []
        y_true = [t for t, _ in filtered]
        y_pred = [p for _, p in filtered]
        valid = np.array([idx for idx in class_ids if idx < support.shape[0] and support[idx] > 0], dtype=np.int64)
    else:
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

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(8, 8), dpi=120)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
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
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    img = np.ascontiguousarray(rgba[..., :3])
    fig.clear()
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


def compute_calibration_metrics(
    conf: Sequence[float],
    correct: Sequence[float | int | bool],
    true_prob: Sequence[float],
    prob_sq_sum: Sequence[float],
    *,
    num_bins: int = 15,
) -> Dict[str, float]:
    if not conf:
        return {"ece_15bin": 0.0, "brier": 0.0, "nll": 0.0}

    conf_arr = np.asarray(conf, dtype=np.float64)
    correct_arr = np.asarray(correct, dtype=np.float64)
    true_prob_arr = np.clip(np.asarray(true_prob, dtype=np.float64), 1e-9, 1.0)
    sq_sum_arr = np.asarray(prob_sq_sum, dtype=np.float64)

    bin_edges = np.linspace(0.0, 1.0, int(max(1, num_bins)) + 1)
    ece = 0.0
    for i in range(len(bin_edges) - 1):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        if i == len(bin_edges) - 2:
            mask = (conf_arr >= lo) & (conf_arr <= hi)
        else:
            mask = (conf_arr >= lo) & (conf_arr < hi)
        if not np.any(mask):
            continue
        acc_bin = float(correct_arr[mask].mean())
        conf_bin = float(conf_arr[mask].mean())
        ece += float(mask.mean()) * abs(acc_bin - conf_bin)

    brier = float(np.mean(sq_sum_arr + 1.0 - 2.0 * true_prob_arr))
    nll = float(np.mean(-np.log(true_prob_arr)))
    return {"ece_15bin": float(ece), "brier": brier, "nll": nll}


def build_confusion_pairs(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    idx2label: Mapping[int, str],
    *,
    support_true: Sequence[int] | None = None,
    bucket_by_class: Mapping[int, str] | None = None,
    limit: int | None = None,
) -> List[Dict[str, object]]:
    if not y_true:
        return []
    if support_true is None:
        max_cls = max(max(y_true), max(y_pred)) + 1
        support_true = np.bincount(np.asarray(y_true, dtype=np.int64), minlength=max_cls).tolist()
    support_true = list(support_true)
    pairs = Counter(
        (int(t), int(p))
        for t, p in zip(y_true, y_pred)
        if int(t) != int(p)
    )
    rows: List[Dict[str, object]] = []
    for (true_id, pred_id), count in pairs.items():
        support = int(support_true[true_id]) if true_id < len(support_true) else 0
        rows.append(
            {
                "true_id": int(true_id),
                "pred_id": int(pred_id),
                "true_label": str(idx2label.get(int(true_id), str(true_id))),
                "pred_label": str(idx2label.get(int(pred_id), str(pred_id))),
                "count": int(count),
                "rate_within_true": (float(count) / float(max(1, support))),
                "support_true": int(support),
                "bucket_true": str(bucket_by_class.get(int(true_id), "none")) if bucket_by_class else "none",
                "bucket_pred": str(bucket_by_class.get(int(pred_id), "none")) if bucket_by_class else "none",
            }
        )
    rows.sort(key=lambda row: (-int(row["count"]), -float(row["rate_within_true"]), int(row["true_id"]), int(row["pred_id"])))
    if limit is not None and limit > 0:
        rows = rows[: int(limit)]
    return rows


def format_table_text(rows: Sequence[Mapping[str, object]], columns: Sequence[str], *, max_rows: int = 50) -> str:
    if not rows:
        return "No rows."
    cols = [str(c) for c in columns]
    lines = ["\t".join(cols)]
    for row in rows[: max(1, int(max_rows))]:
        vals = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                vals.append(f"{value:.6f}")
            else:
                vals.append(str(value))
        lines.append("\t".join(vals))
    return "\n".join(lines)


def format_confusion_pairs_text(rows: Sequence[Mapping[str, object]], *, max_rows: int = 50) -> str:
    return format_table_text(
        rows,
        ("true_label", "pred_label", "count", "rate_within_true", "support_true", "bucket_true", "bucket_pred"),
        max_rows=max_rows,
    )


def format_prediction_rows_text(rows: Sequence[Mapping[str, object]], *, max_rows: int = 50) -> str:
    return format_table_text(
        rows,
        ("video", "t0", "t1", "true_label", "pred_label", "conf", "margin", "entropy", "bucket_true"),
        max_rows=max_rows,
    )


def format_per_class_rows_text(rows: Sequence[Mapping[str, object]], *, max_rows: int = 50) -> str:
    return format_table_text(
        rows,
        (
            "class_id",
            "label",
            "bucket",
            "support_train",
            "support_val",
            "f1",
            "precision",
            "recall",
            "conf_mean",
            "margin_mean",
        ),
        max_rows=max_rows,
    )
