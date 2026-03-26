from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from msagcn.training.oof_storage import read_predictions_table, read_sharded_array


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a family_map.json from train-only OOF artifacts.")
    p.add_argument("--oof_dir", required=True, type=str)
    p.add_argument("--out", required=True, type=str, help="Output family_map.json path")
    p.add_argument("--num_families", type=int, default=128)
    p.add_argument(
        "--auto_num_families",
        action="store_true",
        help="Automatically choose num_families from train-only OOF diagnostics instead of using --num_families directly.",
    )
    p.add_argument(
        "--auto_num_families_candidates",
        type=str,
        default="",
        help="Optional comma-separated candidate family counts for --auto_num_families (default: auto ladder from num_classes).",
    )
    p.add_argument("--proto_weight", type=float, default=0.50)
    p.add_argument("--conf_weight", type=float, default=0.30)
    p.add_argument("--kin_weight", type=float, default=0.20)
    p.add_argument("--top_neighbors", type=int, default=10)
    return p.parse_args()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.where(norms > 0.0, matrix / np.clip(norms, 1e-12, None), 0.0)
    sim = safe @ safe.T
    sim = np.clip(sim, -1.0, 1.0)
    return (sim + 1.0) * 0.5


def _fit_agglomerative(distance: np.ndarray, num_families: int) -> np.ndarray:
    kwargs = {"n_clusters": int(num_families), "linkage": "average"}
    try:
        model = AgglomerativeClustering(metric="precomputed", **kwargs)
    except TypeError:
        model = AgglomerativeClustering(affinity="precomputed", **kwargs)
    return model.fit_predict(distance)


def _round_candidate(raw: float, *, num_classes: int) -> int:
    if num_classes >= 128:
        step = 8
    elif num_classes >= 64:
        step = 4
    else:
        step = 2
    rounded = int(step * round(float(raw) / float(step)))
    return int(min(num_classes, max(2, rounded)))


def _default_auto_family_candidates(num_classes: int) -> list[int]:
    ratios = [1.0 / 16.0, 3.0 / 32.0, 1.0 / 8.0, 5.0 / 32.0, 3.0 / 16.0]
    candidates = {_round_candidate(num_classes * ratio, num_classes=num_classes) for ratio in ratios}
    sqrt_candidate = _round_candidate(np.sqrt(float(num_classes)) * 4.0, num_classes=num_classes)
    candidates.add(sqrt_candidate)
    return sorted(candidate for candidate in candidates if 1 <= int(candidate) <= int(num_classes))


def _parse_auto_candidates(text: str, *, num_classes: int) -> list[int]:
    values: list[int] = []
    for chunk in str(text).split(","):
        item = chunk.strip()
        if not item:
            continue
        value = int(item)
        if value < 1 or value > int(num_classes):
            raise ValueError(f"auto family candidate {value} is outside [1, {num_classes}]")
        values.append(int(value))
    uniq = sorted(set(values))
    if not uniq:
        raise ValueError("--auto_num_families_candidates must contain at least one valid integer")
    return uniq


def _remap_family_ids(labels: np.ndarray) -> np.ndarray:
    unique_labels = sorted(np.unique(labels).tolist())
    remap = {int(old): int(new) for new, old in enumerate(unique_labels)}
    return np.asarray([remap[int(v)] for v in labels], dtype=np.int64)


def _compute_partition_metrics(sim: np.ndarray, family_ids: np.ndarray) -> dict[str, Any]:
    num_classes = int(len(family_ids))
    unique_families, counts = np.unique(family_ids, return_counts=True)
    num_families = int(len(unique_families))
    count_map = {int(fid): int(cnt) for fid, cnt in zip(unique_families.tolist(), counts.tolist())}
    family_sizes = [int(cnt) for cnt in counts.tolist()]
    target_avg = float(num_classes) / float(max(1, num_families))
    same_mask = family_ids[:, None] == family_ids[None, :]
    diag_mask = np.eye(num_classes, dtype=bool)
    intra_mask = same_mask & ~diag_mask
    inter_mask = ~same_mask
    intra_vals = sim[intra_mask]
    inter_vals = sim[inter_mask]
    mean_intra = float(np.mean(intra_vals)) if intra_vals.size > 0 else 0.0
    mean_inter = float(np.mean(inter_vals)) if inter_vals.size > 0 else 0.0
    singleton_family_count = int(sum(1 for size in family_sizes if int(size) == 1))
    singleton_ratio = float(singleton_family_count) / float(max(1, num_families))
    large_family_threshold = max(8, int(np.ceil(target_avg * 2.5)))
    large_family_class_count = int(sum(size for size in family_sizes if int(size) > int(large_family_threshold)))
    large_family_ratio = float(large_family_class_count) / float(max(1, num_classes))
    score = float((mean_intra - mean_inter) - 0.10 * singleton_ratio - 0.08 * large_family_ratio)
    size_hist: dict[str, int] = {}
    for size in family_sizes:
        key = str(int(size))
        size_hist[key] = size_hist.get(key, 0) + 1
    return {
        "num_families": int(num_families),
        "mean_intra_family_similarity": mean_intra,
        "mean_inter_family_similarity": mean_inter,
        "singleton_family_count": singleton_family_count,
        "singleton_family_ratio": singleton_ratio,
        "largest_family_size": int(max(family_sizes) if family_sizes else 0),
        "large_family_threshold": int(large_family_threshold),
        "large_family_class_ratio": large_family_ratio,
        "family_size_histogram": size_hist,
        "score": score,
        "family_count_map": count_map,
    }


def _build_neighbor_exports(
    *,
    sim: np.ndarray,
    proto_sim: np.ndarray,
    conf_sim: np.ndarray,
    kin_sim: np.ndarray,
    family_ids: np.ndarray,
    class_names: dict[int, str],
    top_neighbors_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    num_classes = int(len(family_ids))
    top_cross_family: list[dict[str, Any]] = []
    neighbor_rows: list[dict[str, Any]] = []
    for class_id in range(num_classes):
        order = np.argsort(-sim[class_id])
        top_neighbors = [idx for idx in order.tolist() if idx != class_id][: max(1, int(top_neighbors_k))]
        for rank, neighbor_id in enumerate(top_neighbors, start=1):
            neighbor_rows.append(
                {
                    "class_id": int(class_id),
                    "label": class_names.get(int(class_id), str(class_id)),
                    "family_id": int(family_ids[class_id]),
                    "neighbor_id": int(neighbor_id),
                    "neighbor_label": class_names.get(int(neighbor_id), str(neighbor_id)),
                    "neighbor_family_id": int(family_ids[neighbor_id]),
                    "rank": int(rank),
                    "similarity": float(sim[class_id, neighbor_id]),
                    "proto_similarity": float(proto_sim[class_id, neighbor_id]),
                    "conf_similarity": float(conf_sim[class_id, neighbor_id]),
                    "kin_similarity": float(kin_sim[class_id, neighbor_id]),
                }
            )
            if family_ids[class_id] != family_ids[neighbor_id]:
                top_cross_family.append(
                    {
                        "class_id": int(class_id),
                        "label": class_names.get(int(class_id), str(class_id)),
                        "neighbor_id": int(neighbor_id),
                        "neighbor_label": class_names.get(int(neighbor_id), str(neighbor_id)),
                        "family_id": int(family_ids[class_id]),
                        "neighbor_family_id": int(family_ids[neighbor_id]),
                        "similarity": float(sim[class_id, neighbor_id]),
                        "conf_similarity": float(conf_sim[class_id, neighbor_id]),
                    }
                )
    top_cross_family.sort(key=lambda row: (-float(row["conf_similarity"]), -float(row["similarity"]), int(row["class_id"])))
    return neighbor_rows, top_cross_family


def main() -> None:
    args = parse_args()
    oof_dir = Path(args.oof_dir).expanduser()
    out_path = Path(args.out).expanduser()
    meta_path = oof_dir / "oof_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError("Expected oof_meta.json in the OOF directory")

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    pred_rows = read_predictions_table(oof_dir, basename="oof_predictions")
    try:
        features = read_sharded_array(oof_dir, "oof_features")
    except FileNotFoundError:
        features = np.load(oof_dir / "oof_features.npy")
    try:
        kinematics = read_sharded_array(oof_dir, "oof_kinematics")
    except FileNotFoundError:
        kinematics = np.load(oof_dir / "oof_kinematics.npy")
    if len(pred_rows) != int(features.shape[0]) or len(pred_rows) != int(kinematics.shape[0]):
        raise RuntimeError("OOF predictions/features/kinematics are misaligned")

    num_classes = int(meta.get("num_classes", 0))
    if num_classes <= 0:
        raise ValueError("oof_meta.json must contain num_classes > 0")
    if not bool(args.auto_num_families) and (int(args.num_families) <= 0 or int(args.num_families) > num_classes):
        raise ValueError(f"--num_families must be in [1, {num_classes}]")

    true_ids = np.asarray([int(row["true_class"]) for row in pred_rows], dtype=np.int64)
    pred_ids = np.asarray([int(row["pred_class"]) for row in pred_rows], dtype=np.int64)
    class_names = {int(k): str(v) for k, v in meta.get("class_names", {}).items()}

    supports = np.bincount(true_ids, minlength=num_classes).astype(np.int64)
    prototypes = np.zeros((num_classes, features.shape[1]), dtype=np.float32)
    kin_prototypes = np.zeros((num_classes, kinematics.shape[1]), dtype=np.float32)
    for class_id in range(num_classes):
        mask = true_ids == class_id
        if np.any(mask):
            prototypes[class_id] = features[mask].mean(axis=0)
            kin_prototypes[class_id] = kinematics[mask].mean(axis=0)

    confusion_counts = np.zeros((num_classes, num_classes), dtype=np.float64)
    for t, p in zip(true_ids.tolist(), pred_ids.tolist()):
        confusion_counts[int(t), int(p)] += 1.0
    row_sums = confusion_counts.sum(axis=1, keepdims=True)
    confusion_prob = np.divide(confusion_counts, np.clip(row_sums, 1.0, None), out=np.zeros_like(confusion_counts), where=row_sums > 0)
    conf_sim = 0.5 * (confusion_prob + confusion_prob.T)
    np.fill_diagonal(conf_sim, 1.0)

    proto_sim = _cosine_similarity_matrix(prototypes)
    kin_sim = _cosine_similarity_matrix(kin_prototypes)
    proto_w = float(args.proto_weight)
    conf_w = float(args.conf_weight)
    kin_w = float(args.kin_weight)
    total_w = max(1e-12, proto_w + conf_w + kin_w)
    proto_w, conf_w, kin_w = proto_w / total_w, conf_w / total_w, kin_w / total_w
    sim = proto_w * proto_sim + conf_w * conf_sim + kin_w * kin_sim
    sim = np.clip(sim, 0.0, 1.0)
    np.fill_diagonal(sim, 1.0)
    distance = np.clip(1.0 - sim, 0.0, 1.0)
    auto_search_rows: list[dict[str, Any]] = []
    if bool(args.auto_num_families):
        candidate_counts = (
            _parse_auto_candidates(str(args.auto_num_families_candidates), num_classes=num_classes)
            if str(args.auto_num_families_candidates).strip()
            else _default_auto_family_candidates(num_classes)
        )
    else:
        candidate_counts = [int(args.num_families)]

    candidate_payloads: list[tuple[int, np.ndarray, dict[str, Any]]] = []
    for candidate_num in candidate_counts:
        labels = _fit_agglomerative(distance, int(candidate_num))
        family_ids = _remap_family_ids(labels)
        metrics = _compute_partition_metrics(sim, family_ids)
        auto_row = {
            "candidate_num_families": int(candidate_num),
            "effective_num_families": int(metrics["num_families"]),
            "score": float(metrics["score"]),
            "mean_intra_family_similarity": float(metrics["mean_intra_family_similarity"]),
            "mean_inter_family_similarity": float(metrics["mean_inter_family_similarity"]),
            "singleton_family_count": int(metrics["singleton_family_count"]),
            "singleton_family_ratio": float(metrics["singleton_family_ratio"]),
            "largest_family_size": int(metrics["largest_family_size"]),
            "large_family_class_ratio": float(metrics["large_family_class_ratio"]),
        }
        auto_search_rows.append(auto_row)
        candidate_payloads.append((int(candidate_num), family_ids, metrics))

    if bool(args.auto_num_families):
        best_score = max(float(row["score"]) for row in auto_search_rows)
        tolerance = max(1e-4, abs(best_score) * 0.02)
        eligible_counts = {
            int(row["candidate_num_families"])
            for row in auto_search_rows
            if float(row["score"]) >= float(best_score) - float(tolerance)
        }
        selected_num_families = min(eligible_counts)
    else:
        selected_num_families = int(args.num_families)

    selected_payload = next(payload for payload in candidate_payloads if int(payload[0]) == int(selected_num_families))
    _, family_ids, selected_metrics = selected_payload
    unique_labels = sorted(np.unique(family_ids).tolist())
    neighbor_rows, top_cross_family = _build_neighbor_exports(
        sim=sim,
        proto_sim=proto_sim,
        conf_sim=conf_sim,
        kin_sim=kin_sim,
        family_ids=family_ids,
        class_names=class_names,
        top_neighbors_k=int(args.top_neighbors),
    )
    family_map = {
        "version": 1,
        "num_classes": int(num_classes),
        "num_families": int(len(unique_labels)),
        "class_to_family": {str(class_id): int(family_ids[class_id]) for class_id in range(num_classes)},
        "metadata": {
            "method": "oof_proto_conf_kin_agglomerative",
            "weights": {
                "proto": float(proto_w),
                "conf": float(conf_w),
                "kin": float(kin_w),
            },
            "source_oof_dir": str(oof_dir.resolve()),
            "target_num_families": int(selected_num_families),
            "auto_num_families": bool(args.auto_num_families),
            "auto_candidate_counts": [int(v) for v in candidate_counts],
        },
    }
    stats = {
        "num_classes": int(num_classes),
        "num_families": int(len(unique_labels)),
        "selected_num_families": int(selected_num_families),
        "auto_num_families": bool(args.auto_num_families),
        "auto_num_families_candidates": [int(v) for v in candidate_counts],
        "singleton_family_count": int(selected_metrics["singleton_family_count"]),
        "family_size_histogram": dict(selected_metrics["family_size_histogram"]),
        "mean_intra_family_similarity": float(selected_metrics["mean_intra_family_similarity"]),
        "mean_inter_family_similarity": float(selected_metrics["mean_inter_family_similarity"]),
        "largest_family_size": int(selected_metrics["largest_family_size"]),
        "large_family_class_ratio": float(selected_metrics["large_family_class_ratio"]),
        "selection_score": float(selected_metrics["score"]),
        "top_confusing_cross_family_pairs": top_cross_family[:50],
    }

    _write_json(out_path, family_map)
    _write_json(out_path.with_name("family_stats.json"), stats)
    if bool(args.auto_num_families):
        _write_json(
            out_path.with_name("family_search.json"),
            {
                "version": 1,
                "method": "auto_num_families",
                "selected_num_families": int(selected_num_families),
                "selection_tolerance": float(max(1e-4, abs(max(float(row['score']) for row in auto_search_rows)) * 0.02)),
                "candidates": auto_search_rows,
            },
        )
    _write_csv(
        out_path.with_name("class_neighbors.csv"),
        [
            "class_id",
            "label",
            "family_id",
            "neighbor_id",
            "neighbor_label",
            "neighbor_family_id",
            "rank",
            "similarity",
            "proto_similarity",
            "conf_similarity",
            "kin_similarity",
        ],
        neighbor_rows,
    )
    diagnostics = [
        "# OOF Family Diagnostics",
        "",
        f"- num_classes: {num_classes}",
        f"- num_families: {len(unique_labels)}",
        f"- singleton_family_count: {stats['singleton_family_count']}",
        f"- mean_intra_family_similarity: {stats['mean_intra_family_similarity']:.4f}",
        f"- mean_inter_family_similarity: {stats['mean_inter_family_similarity']:.4f}",
        f"- auto_num_families: {int(bool(args.auto_num_families))}",
        f"- selected_num_families: {int(selected_num_families)}",
    ]
    (out_path.with_name("family_diagnostics.md")).write_text("\n".join(diagnostics) + "\n", encoding="utf-8")
    if bool(args.auto_num_families):
        print(f"Auto-selected num_families={int(selected_num_families)} from candidates {candidate_counts}")
    print(f"Done. Wrote family map to {out_path}")


if __name__ == "__main__":
    main()
