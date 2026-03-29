#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
import numpy as np
import pyarrow.parquet as pq

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402


def _safe_float(value, default=np.nan) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _load_manifest(path: Path) -> List[Dict]:
    if path.suffix.lower() == ".parquet":
        return [dict(row) for row in pq.read_table(path).to_pylist()]
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise RuntimeError(f"Input must be a JSON array or parquet table: {path}")
    return data


def _series(manifest: List[Dict], key: str) -> np.ndarray:
    vals = [_safe_float(row.get(key)) for row in manifest]
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise RuntimeError(f"No numeric values found for key '{key}'")
    return arr


def _ecdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.sort(values)
    y = np.arange(1, x.size + 1, dtype=float) / float(x.size)
    return x, y


def _save_figure(fig: plt.Figure, out_dir: Path, stem: str, formats: Sequence[str], dpi: int) -> List[Path]:
    out_paths: List[Path] = []
    for ext in formats:
        ext_norm = ext.strip().lower()
        if not ext_norm:
            continue
        out_path = out_dir / f"{stem}.{ext_norm}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        out_paths.append(out_path)
    plt.close(fig)
    return out_paths


def _add_vertical_markers(
    ax: plt.Axes,
    percentiles: Iterable[int],
    values: np.ndarray,
    color: str,
    label_prefix: str = "P",
) -> Dict[str, float]:
    pct_arr = np.asarray(list(percentiles), dtype=float)
    pct_vals = np.percentile(values, pct_arr)
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    out: Dict[str, float] = {}
    for idx, (p, x) in enumerate(zip(pct_arr, pct_vals)):
        ax.axvline(x, color=color, linestyle="--", linewidth=1.2, alpha=0.9)
        y_pos = 0.04 + idx * 0.08
        ax.text(
            x,
            y_pos,
            f"{label_prefix}{int(p)}={x:.3f}",
            transform=trans,
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=9,
            color=color,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": color, "alpha": 0.7},
        )
        out[f"p{int(p)}"] = float(x)
    return out


def _figure5(manifest: List[Dict], out_dir: Path, formats: Sequence[str], dpi: int) -> Dict[str, float]:
    vals = _series(manifest, "hands_coverage")
    x, y = _ecdf(vals)
    p10, p50, p90 = np.percentile(vals, [10, 50, 90])
    n_videos = int(vals.size)
    frac_eq_1 = float(np.mean(np.isclose(vals, 1.0, atol=1e-6)))
    frac_lt_090 = float(np.mean(vals < 0.90))
    frac_lt_050 = float(np.mean(vals < 0.50))

    fig, (ax, ax_zoom) = plt.subplots(
        1,
        2,
        figsize=(10.2, 4.9),
        gridspec_kw={"width_ratios": [1.9, 1.1], "wspace": 0.14},
    )

    # Main ECDF.
    ax.step(x, y, where="post", linewidth=2.3, color="#0B84A5")
    ax.axvline(float(p10), color="#3E3E3E", linestyle="--", linewidth=1.3, alpha=0.9)
    ax.axvline(1.0, color="#3E3E3E", linestyle=":", linewidth=1.2, alpha=0.9)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("hands_coverage")
    ax.set_ylabel("Fraction of videos")
    ax.set_title("Figure 5. ECDF of hand detection coverage (Slovo)")
    ax.grid(True, alpha=0.25)
    ax.text(
        0.01,
        0.98,
        (
            f"n={n_videos} | P10={p10:.3f} | coverage=1.0: {frac_eq_1:.1%} | "
            f"<0.90: {frac_lt_090:.1%} | <0.50: {frac_lt_050:.1%}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.6,
        color="#3A3A3A",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#777777", "alpha": 0.8},
    )

    # High-coverage zoom panel.
    mask = x >= 0.90
    x_tail = x[mask] if np.any(mask) else x
    y_tail = y[mask] if np.any(mask) else y
    # Exclude the x==1.0 jump when computing zoom y-limits.
    pre_one = x_tail < (1.0 - 1e-9)
    if np.any(pre_one):
        x_tail_plot = x_tail[pre_one]
        y_tail_plot = y_tail[pre_one]
    else:
        x_tail_plot = x_tail
        y_tail_plot = y_tail
    if x_tail_plot.size > 0 and x_tail_plot[-1] < (1.0 - 1e-9):
        # Keep the jump at x=1.0 excluded from y-limits, but extend the
        # drawn step to the right border to avoid an empty tail segment.
        x_tail_draw = np.append(x_tail_plot, 1.0)
        y_tail_draw = np.append(y_tail_plot, y_tail_plot[-1])
    else:
        x_tail_draw = x_tail_plot
        y_tail_draw = y_tail_plot
    y_lo = float(np.min(y_tail_plot)) if y_tail_plot.size else 0.0
    y_hi = float(np.max(y_tail_plot)) if y_tail_plot.size else 1.0
    y_pad = max(0.01, (y_hi - y_lo) * 0.08)
    ax_zoom.step(x_tail_draw, y_tail_draw, where="post", linewidth=2.1, color="#0B84A5")
    ax_zoom.axvline(1.0, color="#3E3E3E", linestyle=":", linewidth=1.1, alpha=0.85)
    if p10 >= 0.90:
        ax_zoom.axvline(float(p10), color="#3E3E3E", linestyle="--", linewidth=1.1, alpha=0.85)
        trans_zoom = blended_transform_factory(ax_zoom.transData, ax_zoom.transAxes)
        ax_zoom.text(
            float(p10),
            0.97,
            f"P10={p10:.3f}",
            transform=trans_zoom,
            rotation=90,
            ha="right",
            va="top",
            fontsize=8,
            color="#3E3E3E",
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "#777777",
                "alpha": 0.8,
            },
        )
    for y_ref in (0.10, 0.30):
        ax_zoom.axhline(y_ref, color="#9A9A9A", linestyle=":", linewidth=1.0, alpha=0.55)
    ax_zoom.set_xlim(0.90, 1.0)
    ax_zoom.set_ylim(max(0.0, y_lo - y_pad), min(1.0, y_hi + y_pad))
    ax_zoom.set_title("High-coverage zoom", fontsize=10)
    ax_zoom.set_xlabel("hands_coverage")
    ax_zoom.grid(True, alpha=0.22)

    _save_figure(fig, out_dir, "figure5_hands_coverage_ecdf", formats, dpi)
    return {
        "p10": float(p10),
        "p50": float(p50),
        "p90": float(p90),
        "n": float(n_videos),
        "coverage_eq_1_frac": frac_eq_1,
        "coverage_lt_090_frac": frac_lt_090,
        "coverage_lt_050_frac": frac_lt_050,
    }


def _figure6(manifest: List[Dict], out_dir: Path, formats: Sequence[str], dpi: int) -> Dict[str, float]:
    left = _series(manifest, "left_coverage")
    right = _series(manifest, "right_coverage")
    both = _series(manifest, "both_coverage")
    n_videos = int(both.size)
    both_p50 = float(np.percentile(both, 50))
    both_p75 = float(np.percentile(both, 75))
    both_zero = float(np.mean(both <= 0.0))
    both_one = float(np.mean(both >= 1.0))

    fig, (ax, ax_zoom) = plt.subplots(
        1,
        2,
        figsize=(10.4, 4.9),
        gridspec_kw={"width_ratios": [1.9, 1.1], "wspace": 0.14},
    )
    for vals, label, color in [
        (left, "left_coverage", "#4C78A8"),
        (right, "right_coverage", "#F58518"),
        (both, "both_coverage", "#54A24B"),
    ]:
        x, y = _ecdf(vals)
        ax.step(x, y, where="post", linewidth=2.2, label=label, color=color)

    ax.axvline(both_p50, color="#2F6B2F", linestyle="--", linewidth=1.25, alpha=0.9)
    ax.axvline(both_p75, color="#2F6B2F", linestyle=":", linewidth=1.15, alpha=0.9)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("coverage")
    ax.set_ylabel("Fraction of videos")
    ax.set_title("Figure 6. ECDFs of left/right/both hand coverage (Slovo)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.text(
        0.01,
        0.02,
        f"n={n_videos} | both P50={both_p50:.3f} | both P75={both_p75:.3f} | both=0: {both_zero:.1%} | both=1: {both_one:.1%}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#3A3A3A",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#777777", "alpha": 0.8},
    )

    # Right-tail zoom panel for high-coverage separation.
    for vals, color in [
        (left, "#4C78A8"),
        (right, "#F58518"),
        (both, "#54A24B"),
    ]:
        x, y = _ecdf(vals)
        mask = x >= 0.85
        x_tail = x[mask] if np.any(mask) else x
        y_tail = y[mask] if np.any(mask) else y
        ax_zoom.step(x_tail, y_tail, where="post", linewidth=2.0, color=color)
    ax_zoom.axvline(both_p50, color="#2F6B2F", linestyle="--", linewidth=1.1, alpha=0.85)
    ax_zoom.axvline(both_p75, color="#2F6B2F", linestyle=":", linewidth=1.0, alpha=0.85)
    ax_zoom.set_xlim(0.85, 1.0)
    ax_zoom.set_ylim(0.0, 1.0)
    ax_zoom.grid(True, alpha=0.22)
    ax_zoom.set_title("Right-tail zoom", fontsize=10)
    ax_zoom.set_xlabel("coverage")

    _save_figure(fig, out_dir, "figure6_left_right_both_ecdf", formats, dpi)
    return {
        "p10": float(np.percentile(both, 10)),
        "p50": both_p50,
        "p75": both_p75,
        "p90": float(np.percentile(both, 90)),
        "both_zero_frac": both_zero,
        "both_one_frac": both_one,
        "n": float(n_videos),
    }


def _figure7(manifest: List[Dict], out_dir: Path, formats: Sequence[str], dpi: int) -> Dict[str, float]:
    vals = _series(manifest, "quality_score")
    x, y = _ecdf(vals)
    p25, p50, p75 = np.percentile(vals, [25, 50, 75])
    n_videos = int(vals.size)
    frac_ge_090 = float(np.mean(vals >= 0.90))
    frac_ge_095 = float(np.mean(vals >= 0.95))
    frac_le_040 = float(np.mean(vals <= 0.40))
    jump_share = float(np.mean((vals >= 0.395) & (vals <= 0.405)))

    fig, (ax_main, ax_zoom) = plt.subplots(
        1,
        2,
        figsize=(10.2, 4.9),
        gridspec_kw={"width_ratios": [1.9, 1.1], "wspace": 0.14},
    )

    # Main ECDF.
    ax_main.step(x, y, where="post", linewidth=2.3, color="#7A5195")
    for x_ref in (float(p25), float(p50), float(p75)):
        ax_main.axvline(x_ref, color="#5E3C99", linestyle="--", linewidth=1.25, alpha=0.9)
    x_min = max(0.0, float(np.nanmin(vals)) - 0.02)
    x_max = min(1.0, float(np.nanmax(vals)) + 0.01)
    if x_max <= x_min:
        x_min, x_max = 0.0, 1.0
    ax_main.set_xlim(x_min, x_max)
    ax_main.set_ylim(0.0, 1.0)
    ax_main.set_xlabel(r"Composite quality score $Q$")
    ax_main.set_ylabel("Fraction of videos")
    ax_main.grid(True, alpha=0.25)

    # Single compact stats line (no clutter blocks).
    ax_main.text(
        0.01,
        0.02,
        f"n={n_videos} | P25={p25:.3f} | P50={p50:.3f} | P75={p75:.3f}",
        transform=ax_main.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#3A3A3A",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#777777", "alpha": 0.8},
    )

    # Right-tail zoom panel.
    mask = x >= 0.85
    x_tail = x[mask] if np.any(mask) else x
    y_tail = y[mask] if np.any(mask) else y
    y_lo = float(np.min(y_tail)) if y_tail.size else 0.0
    ax_zoom.step(x_tail, y_tail, where="post", linewidth=2.1, color="#7A5195")
    for x_ref in (float(p50), float(p75)):
        ax_zoom.axvline(x_ref, color="#5E3C99", linestyle="--", linewidth=1.15, alpha=0.85)
    ax_zoom.set_xlim(0.85, 1.0)
    ax_zoom.set_ylim(max(0.0, y_lo - 0.03), 1.0)
    ax_zoom.grid(True, alpha=0.22)
    ax_zoom.set_title("Right-tail zoom", fontsize=10)
    ax_zoom.set_xlabel(r"Composite quality score $Q$")

    _save_figure(fig, out_dir, "figure7_quality_score_ecdf", formats, dpi)
    return {
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
        "n": float(n_videos),
        "frac_ge_090": frac_ge_090,
        "frac_ge_095": frac_ge_095,
        "frac_le_040": frac_le_040,
        "jump_share_0p395_0p405": jump_share,
    }


def _per_video_failure_rates(manifest: List[Dict]) -> Dict[str, np.ndarray]:
    missing: List[float] = []
    occluded: List[float] = []
    swap: List[float] = []
    outlier: List[float] = []

    for row in manifest:
        num_frames = _safe_float(row.get("num_frames"), 0.0)
        if not np.isfinite(num_frames) or num_frames <= 0.0:
            continue

        m1 = _safe_float(row.get("missing_frames_1"))
        m2 = _safe_float(row.get("missing_frames_2"))
        o1 = _safe_float(row.get("occluded_frames_1"))
        o2 = _safe_float(row.get("occluded_frames_2"))
        sw = _safe_float(row.get("swap_frames"))
        out1 = _safe_float(row.get("outlier_frames_1"))
        out2 = _safe_float(row.get("outlier_frames_2"))

        vals = [m1, m2, o1, o2, sw, out1, out2]
        if not all(np.isfinite(v) for v in vals):
            continue

        # Formulas requested by user text:
        # missing_rate = (missing_frames_1 + missing_frames_2) / num_frames
        # occluded_rate = (occluded_frames_1 + occluded_frames_2) / num_frames
        # swap_rate = swap_frames / num_frames
        # outlier_rate = (outlier_frames_1 + outlier_frames_2) / num_frames
        missing.append((m1 + m2) / num_frames)
        occluded.append((o1 + o2) / num_frames)
        swap.append(sw / num_frames)
        outlier.append((out1 + out2) / num_frames)

    rates = {
        "Missing": np.asarray(missing, dtype=float),
        "Occluded": np.asarray(occluded, dtype=float),
        "Swap": np.asarray(swap, dtype=float),
        "Outlier": np.asarray(outlier, dtype=float),
    }
    for key, arr in rates.items():
        if arr.size == 0:
            raise RuntimeError(
                f"No values for '{key}' rate. Rebuild manifest with eval fields first."
            )
    return rates


def _figure8(manifest: List[Dict], out_dir: Path, formats: Sequence[str], dpi: int) -> Dict[str, Dict[str, float]]:
    rates = _per_video_failure_rates(manifest)
    labels = list(rates.keys())

    medians = np.array([np.percentile(rates[k], 50) for k in labels], dtype=float)
    p25 = np.array([np.percentile(rates[k], 25) for k in labels], dtype=float)
    p75 = np.array([np.percentile(rates[k], 75) for k in labels], dtype=float)

    err_low = medians - p25
    err_high = p75 - medians

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    x = np.arange(len(labels))
    colors = ["#D62728", "#FF7F0E", "#1F77B4", "#2CA02C"]

    bars = ax.bar(x, medians, color=colors, alpha=0.9, width=0.62, edgecolor="black", linewidth=0.6)
    ax.errorbar(
        x,
        medians,
        yerr=np.vstack([err_low, err_high]),
        fmt="none",
        ecolor="black",
        elinewidth=1.2,
        capsize=4,
        capthick=1.2,
        zorder=3,
    )

    for rect, m in zip(bars, medians):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height(),
            f"{m:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rate (fraction of frames)")
    ax.grid(axis="y", alpha=0.25)
    y_max = float(np.max(p75)) if np.max(p75) > 0 else 1.0
    ax.set_ylim(0.0, y_max * 1.18)

    _save_figure(fig, out_dir, "figure8_failure_modes_median_iqr", formats, dpi)

    out_stats: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(labels):
        out_stats[name.lower()] = {
            "p25": float(p25[i]),
            "median": float(medians[i]),
            "p75": float(p75[i]),
        }
    return out_stats


def _ccdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.sort(values)
    n = float(x.size)
    # Strict empirical survival: S(x_i) = P(rate > x_i).
    # For log-scale plotting, drop the terminal zero point.
    y = (n - np.arange(1, x.size + 1, dtype=float)) / n
    keep = y > 0.0
    return x[keep], y[keep]


def _figure8_ccdf(
    manifest: List[Dict], out_dir: Path, formats: Sequence[str], dpi: int
) -> Dict[str, Dict[str, float]]:
    rates = _per_video_failure_rates(manifest)
    order = ["Missing", "Occluded", "Swap", "Outlier"]
    display_names = {
        "Missing": "MissingRate",
        "Occluded": "OcclusionRate",
        "Swap": "SwapRate",
        "Outlier": "OutlierRate",
    }
    colors = {
        "Missing": "#D62728",
        "Occluded": "#FF7F0E",
        "Swap": "#1F77B4",
        "Outlier": "#2CA02C",
    }

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    y_ref = 0.10
    ax.axhline(y_ref, color="#777777", linestyle="--", linewidth=1.1, alpha=0.8)
    ax.text(
        0.995,
        y_ref * 1.08,
        "P(rate > x) = 0.10",
        transform=blended_transform_factory(ax.transAxes, ax.transData),
        ha="right",
        va="bottom",
        fontsize=9,
        color="#555555",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#999999", "alpha": 0.7},
    )

    stats: Dict[str, Dict[str, float]] = {}
    max_x = 0.0
    for name in order:
        vals = rates[name]
        max_x = max(max_x, float(np.max(vals)))
        x, y = _ccdf(vals)
        nz = float(np.mean(vals > 0.0))
        p90 = float(np.percentile(vals, 90))
        p95 = float(np.percentile(vals, 95))
        ax.step(
            x,
            y,
            where="post",
            linewidth=2.0,
            color=colors[name],
            label=f"{display_names[name]} (non-zero={nz:.1%}, P90={p90:.3f})",
        )
        ax.scatter([p90], [y_ref], color=colors[name], s=18, zorder=4)
        stats[name.lower()] = {
            "nonzero_frac": nz,
            "p90": p90,
            "p95": p95,
            "max": float(np.max(vals)),
        }

    x_lim_hi = max(1.0, min(2.0, max_x * 1.03))
    ax.set_xlim(0.0, x_lim_hi)
    n = len(next(iter(rates.values())))
    min_survival = 1.0 / float(max(1, n))
    ax.set_ylim(min_survival, 1.0)
    ax.set_yscale("log")
    ax.set_xlabel("Rate (fraction of frames)")
    ax.set_ylabel("Survival fraction P(rate > x)")
    ax.grid(True, which="both", axis="both", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    _save_figure(fig, out_dir, "figure8_failure_modes_ccdf", formats, dpi)
    return stats


def _figure9(manifest: List[Dict], out_dir: Path, formats: Sequence[str], dpi: int) -> Dict[str, Dict[str, float]]:
    series_map = {
        "SP-left": _series(manifest, "sp_recovered_left_frac"),
        "SP-right": _series(manifest, "sp_recovered_right_frac"),
        "Track-left": _series(manifest, "track_recovered_left_frac"),
        "Track-right": _series(manifest, "track_recovered_right_frac"),
    }

    labels = list(series_map.keys())
    data = [series_map[k] for k in labels]

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    bp = ax.boxplot(
        data,
        tick_labels=labels,
        showfliers=False,
        widths=0.58,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.6},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
    )
    box_colors = ["#5DA5DA", "#60BD68", "#FAA43A", "#F17CB0"]
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    ax.set_ylabel("Recovered fraction")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)

    _save_figure(fig, out_dir, "figure9_recovery_boxplots", formats, dpi)

    out_stats: Dict[str, Dict[str, float]] = {}
    for name, arr in series_map.items():
        out_stats[name] = {
            "p25": float(np.percentile(arr, 25)),
            "median": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
        }
    return out_stats


def _figure9_inset(
    manifest: List[Dict], out_dir: Path, formats: Sequence[str], dpi: int
) -> Dict[str, Dict[str, float]]:
    series_map = {
        "SP-left": _series(manifest, "sp_recovered_left_frac"),
        "SP-right": _series(manifest, "sp_recovered_right_frac"),
        "Track-left": _series(manifest, "track_recovered_left_frac"),
        "Track-right": _series(manifest, "track_recovered_right_frac"),
    }
    order = ["SP-left", "SP-right", "Track-left", "Track-right"]
    display_labels = {
        "SP-left": "ROI 2nd-pass\nleft hand",
        "SP-right": "ROI 2nd-pass\nright hand",
        "Track-left": "Tracking bridge\nleft hand",
        "Track-right": "Tracking bridge\nright hand",
    }
    data = [series_map[k] for k in order]
    tick_labels = [display_labels[k] for k in order]

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    bp = ax.boxplot(
        data,
        tick_labels=tick_labels,
        showfliers=False,
        widths=0.58,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.7},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
    )
    box_colors = ["#5DA5DA", "#60BD68", "#FAA43A", "#F17CB0"]
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    ax.set_ylabel("Recovered fraction per clip\n(recovered / total frames)")
    ax.set_xlabel("Recovery mechanism + hand")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)

    # Compact per-series summary directly on the plot for quick reading.
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    out_stats: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(order, 1):
        arr = series_map[name]
        p25 = float(np.percentile(arr, 25))
        med = float(np.percentile(arr, 50))
        p75 = float(np.percentile(arr, 75))
        nz = float(np.mean(arr > 0.0))
        if name.startswith("SP-"):
            ax.text(
                i,
                0.98,
                f"Median={med:.3f}\nIQR={p25:.3f}-{p75:.3f}\nNon-zero={nz:.1%}",
                transform=trans,
                ha="center",
                va="top",
                fontsize=8,
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "edgecolor": "#666666",
                    "alpha": 0.85,
                },
            )
        else:
            ax.text(
                i,
                0.035,
                f"Non-zero={nz:.1%}",
                transform=trans,
                ha="center",
                va="bottom",
                fontsize=8,
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "white",
                    "edgecolor": "#777777",
                    "alpha": 0.85,
                },
            )
        out_stats[name] = {
            "p25": p25,
            "median": med,
            "p75": p75,
            "nonzero_frac": nz,
        }

    # Inset to resolve low-magnitude tracking distributions.
    track_data = [series_map["Track-left"], series_map["Track-right"]]
    track_labels = ["Left hand", "Right hand"]
    track_p99 = max(float(np.percentile(arr, 99)) for arr in track_data)
    track_ylim = min(0.12, max(0.04, track_p99 * 1.10))

    axins = inset_axes(ax, width="44%", height="46%", loc="upper right", borderpad=1.2)
    bp2 = axins.boxplot(
        track_data,
        tick_labels=track_labels,
        showfliers=False,
        widths=0.58,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.4},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
    )
    for patch, color in zip(bp2["boxes"], box_colors[2:]):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)
    axins.set_ylim(0.0, track_ylim)
    axins.grid(axis="y", alpha=0.25)
    axins.set_title("Tracking bridge (zoomed)", fontsize=9)
    axins.tick_params(axis="x", labelsize=8)
    axins.tick_params(axis="y", labelsize=8)

    # Detailed stats for tracking stay in the inset to avoid overlap.
    for idx, key in enumerate(["Track-left", "Track-right"]):
        arr = series_map[key]
        p25 = float(np.percentile(arr, 25))
        med = float(np.percentile(arr, 50))
        p75 = float(np.percentile(arr, 75))
        nz = float(np.mean(arr > 0.0))
        x_pos = 0.03 if idx == 0 else 0.53
        axins.text(
            x_pos,
            0.97,
            f"Median={med:.3f}\nIQR={p25:.3f}-{p75:.3f}\nNon-zero={nz:.1%}",
            transform=axins.transAxes,
            ha="left",
            va="top",
            fontsize=6.5,
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "#666666",
                "alpha": 0.85,
            },
        )

    _save_figure(fig, out_dir, "figure9_recovery_boxplots_inset", formats, dpi)
    return out_stats


def main() -> int:
    ap = argparse.ArgumentParser("Build publication-ready figures from Slovo videos.parquet.")
    ap.add_argument("--manifest", default="datasets/skeletons/Slovo/videos.parquet")
    ap.add_argument("--out-dir", default="outputs/Slovo/paper_figures")
    ap.add_argument("--formats", default="png,pdf", help="Comma-separated, e.g. png,pdf,svg")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = [x.strip() for x in str(args.formats).split(",") if x.strip()]

    manifest = _load_manifest(manifest_path)
    print(f"[INFO] Loaded manifest: {manifest_path} ({len(manifest)} videos)")

    summary: Dict[str, Dict] = {}
    summary["figure5"] = _figure5(manifest, out_dir, formats, args.dpi)
    summary["figure6_both_coverage_markers"] = _figure6(manifest, out_dir, formats, args.dpi)
    summary["figure7"] = _figure7(manifest, out_dir, formats, args.dpi)
    summary["figure8"] = _figure8(manifest, out_dir, formats, args.dpi)
    summary["figure8_ccdf"] = _figure8_ccdf(manifest, out_dir, formats, args.dpi)
    summary["figure9"] = _figure9(manifest, out_dir, formats, args.dpi)
    summary["figure9_inset"] = _figure9_inset(manifest, out_dir, formats, args.dpi)

    summary_path = out_dir / "figure_stats.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] Figures saved to: {out_dir}")
    print(f"[OK] Stats saved to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
