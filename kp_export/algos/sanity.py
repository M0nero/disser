from __future__ import annotations

from typing import List, Dict, Optional, Tuple
import math

HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

def _wrist_xy(pts: Optional[List[Dict[str, float]]]) -> Optional[Tuple[float, float]]:
    if not pts or len(pts) == 0:
        return None
    try:
        return float(pts[0]["x"]), float(pts[0]["y"])
    except Exception:
        return None

def _hand_scale(pts: Optional[List[Dict[str, float]]]) -> float:
    if not pts or len(pts) < 2:
        return 0.0
    w = _wrist_xy(pts)
    if w is None:
        return 0.0
    wx, wy = w
    dists = []
    for p in pts[1:]:
        try:
            dx = float(p["x"]) - wx
            dy = float(p["y"]) - wy
        except Exception:
            continue
        dists.append(math.hypot(dx, dy))
    if not dists:
        return 0.0
    dists.sort()
    return dists[len(dists) // 2]

def check_hand_sanity(
    pts,
    *,
    prev_anchor=None,
    prev_pred=None,
    world_coords=False,
    scale_range=(0.70, 1.35),
    wrist_k=2.0,
    bone_tol=0.30,
    enable_wrist_check=True,
    debug_out: Optional[Dict[str, float]] = None,
) -> tuple[bool, list[str]]:
    reasons: List[str] = []
    if not pts:
        return True, reasons

    curr_scale = _hand_scale(pts)
    if debug_out is not None:
        debug_out["curr_scale"] = float(curr_scale)
    anchor_scale = None
    if prev_anchor:
        anchor_scale = _hand_scale(prev_anchor)
        if anchor_scale <= 0.0:
            anchor_scale = None
    if debug_out is not None:
        debug_out["anchor_scale"] = float(anchor_scale) if anchor_scale is not None else None

    # A) scale_jump
    ratio = None
    if anchor_scale is not None and anchor_scale > 0.0:
        ratio = curr_scale / anchor_scale if anchor_scale > 0.0 else None
        if ratio is not None:
            lo, hi = float(scale_range[0]), float(scale_range[1])
            if ratio < lo or ratio > hi:
                reasons.append(f"scale_jump ratio={ratio:.2f} range=[{lo:.2f},{hi:.2f}]")
    if debug_out is not None:
        debug_out["scale_ratio"] = float(ratio) if ratio is not None else None
        debug_out["scale_lo"] = float(scale_range[0])
        debug_out["scale_hi"] = float(scale_range[1])

    # B) wrist_jump
    if enable_wrist_check and prev_pred:
        cur_xy = _wrist_xy(pts)
        prev_xy = _wrist_xy(prev_pred)
        scale_ref = anchor_scale if anchor_scale is not None else curr_scale
        if cur_xy is not None and prev_xy is not None and scale_ref > 0.0:
            dist = math.hypot(cur_xy[0] - prev_xy[0], cur_xy[1] - prev_xy[1])
            thr = wrist_k * scale_ref
            if dist > thr:
                reasons.append(
                    f"wrist_jump dist={dist:.3f} thr={thr:.3f} (k={wrist_k:.1f} scale={scale_ref:.3f})"
                )
            if debug_out is not None:
                debug_out["wrist_jump_dist"] = float(dist)
                debug_out["wrist_jump_thr"] = float(thr)
                debug_out["wrist_jump_k"] = float(wrist_k)
        elif debug_out is not None:
            debug_out["wrist_jump_dist"] = None
            debug_out["wrist_jump_thr"] = None
            debug_out["wrist_jump_k"] = float(wrist_k)
    elif debug_out is not None:
        debug_out["wrist_jump_dist"] = None
        debug_out["wrist_jump_thr"] = None
        debug_out["wrist_jump_k"] = float(wrist_k)

    # C) bone_ratio / bone_shape
    if prev_anchor and anchor_scale is not None and anchor_scale > 0.0 and curr_scale > 0.0:
        eps = 1e-6
        max_rel_err = 0.0
        worst = None
        for i, j in HAND_BONES:
            if i >= len(pts) or j >= len(pts) or i >= len(prev_anchor) or j >= len(prev_anchor):
                continue
            try:
                cur_dx = float(pts[i]["x"]) - float(pts[j]["x"])
                cur_dy = float(pts[i]["y"]) - float(pts[j]["y"])
                anc_dx = float(prev_anchor[i]["x"]) - float(prev_anchor[j]["x"])
                anc_dy = float(prev_anchor[i]["y"]) - float(prev_anchor[j]["y"])
            except Exception:
                continue
            cur_len = math.hypot(cur_dx, cur_dy) / curr_scale
            anc_len = math.hypot(anc_dx, anc_dy) / anchor_scale
            rel_err = abs(cur_len - anc_len) / max(anc_len, eps)
            if rel_err > max_rel_err:
                max_rel_err = rel_err
                worst = (i, j)
        if worst is not None and max_rel_err > bone_tol:
            reasons.append(f"bone_ratio max_rel_err={max_rel_err:.2f} tol={bone_tol:.2f} worst={worst}")
        if debug_out is not None:
            debug_out["bone_max_rel_err"] = float(max_rel_err)
            debug_out["bone_worst"] = worst
            debug_out["bone_tol"] = float(bone_tol)
    elif debug_out is not None:
        debug_out["bone_max_rel_err"] = None
        debug_out["bone_worst"] = None
        debug_out["bone_tol"] = float(bone_tol)

    return (len(reasons) == 0), reasons
