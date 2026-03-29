from __future__ import annotations
from typing import List, Dict, Optional, Tuple
from ..core.utils import bbox_from_pts_px, clip_rect, px_to_norm
from ..process.contracts import SecondPassResult
import cv2


def hands_up_gate(hand: str, pose_img_landmarks, W: int, H: int) -> bool:
    if pose_img_landmarks is None or len(pose_img_landmarks) < 17:
        return True
    wrist_idx = 15 if hand == "left" else 16
    elbow_idx = 13 if hand == "left" else 14
    wy = float(pose_img_landmarks[wrist_idx].y) * H
    ey = float(pose_img_landmarks[elbow_idx].y) * H
    return wy < ey


def roi_center_for(hand: str, W: int, H: int, pose_img_landmarks, last_left_px, last_right_px):
    cx = cy = None
    if pose_img_landmarks is not None and len(pose_img_landmarks) >= 17:
        idx = 15 if hand == "left" else 16
        lm = pose_img_landmarks[idx]
        cx = float(lm.x) * W
        cy = float(lm.y) * H
    if cx is None:
        prev_px = last_left_px if hand == "left" else last_right_px
        if prev_px is not None:
            x0, y0, x1, y1 = bbox_from_pts_px(prev_px)
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)
    if cx is None:
        cx, cy = W / 2.0, H / 2.0
    return cx, cy


def run_second_pass_for(
    hand: str, *,
    bgr, proc_w, proc_h, world_coords,
    pose_img_landmarks, last_left_px, last_right_px,
    hands_sp,
    sp_trigger_below: float,
    sp_roi_frac: float,
    sp_margin: float,
    sp_escalate_step: float,
    sp_escalate_max: float,
    sp_hands_up_only: bool,
    center_penalty_lambda: float = 0.30,
    label_relax_margin: float = 0.15,
    require_label_match: bool = False,
    max_center_dist_norm: float = 1.25,
    debug_out: Optional[dict] = None,
    cur_pts,
    cur_score,
    # NEW jitter params (optional)
    sp_jitter_px: int = 0,
    sp_jitter_rings: int = 1,
    # Optional external hint for ROI center (e.g., Kalman prediction)
    center_hint: Optional[Tuple[float, float]] = None,
    debug_return_roi: bool = False,
):
    """
    Second pass with jitter-grid around ROI center.
    On each scale we try base center plus offsets in a square grid.
    The first scale that yields a valid hand wins. Within a scale we keep
    the best-scoring candidate.
    """
    trigger = (cur_pts is None) or (cur_score is None) or (cur_score < sp_trigger_below)
    if not (hands_sp is not None and trigger):
        return SecondPassResult(landmarks=cur_pts, score=cur_score, recovered=False, roi=None)

    if sp_hands_up_only and not hands_up_gate(hand, pose_img_landmarks, proc_w, proc_h):
        return SecondPassResult(landmarks=cur_pts, score=cur_score, recovered=False, roi=None)

    H, W = proc_h, proc_w
    short = min(W, H)

    import math
    n_steps = max(1, int(round(sp_escalate_max / max(1e-6, sp_escalate_step))))
    base_size = max(32.0, sp_roi_frac * short)

    # center selection
    if center_hint is not None:
        cx, cy = center_hint
        cx = float(min(max(cx, 1.0), W - 2.0))
        cy = float(min(max(cy, 1.0), H - 2.0))
    else:
        cx, cy = roi_center_for(hand, W, H, pose_img_landmarks, last_left_px, last_right_px)

    if debug_out is not None:
        debug_out["center_hint_px"] = [float(cx), float(cy)]
        debug_out["candidates"] = []

    best_pts = None
    best_score = cur_score if cur_score is not None else 0.0
    roi_selected = None
    need_roi = bool(debug_return_roi or debug_out is not None)
    return_roi = bool(debug_return_roi)

    from ..core.utils import xyz_list_from_lms

    def _dist_norm_to_center(px_x: float, px_y: float, roi_size: float) -> float:
        roi_size = max(1.0, float(roi_size))
        return math.hypot(px_x - cx, px_y - cy) / roi_size

    def _dist_norm_from_lms(lms_norm, x0: float, y0: float, x1: float, y1: float, roi_size: float) -> Optional[float]:
        if not lms_norm:
            return None
        wx = float(lms_norm[0]["x"]) * (x1 - x0) + x0
        wy = float(lms_norm[0]["y"]) * (y1 - y0) + y0
        return _dist_norm_to_center(wx, wy, roi_size)

    dist_norm_current = None
    if cur_score is not None and cur_pts is not None:
        if not world_coords:
            if len(cur_pts) > 0:
                dist_norm_current = _dist_norm_to_center(
                    float(cur_pts[0]["x"]) * W,
                    float(cur_pts[0]["y"]) * H,
                    base_size,
                )
        else:
            prev_px = last_left_px if hand == "left" else last_right_px
            if prev_px:
                dist_norm_current = _dist_norm_to_center(
                    float(prev_px[0]["x"]),
                    float(prev_px[0]["y"]),
                    base_size,
                )

    best_combined = float("-inf")
    if cur_score is not None and cur_pts is not None:
        best_combined = float(cur_score)
        if dist_norm_current is not None:
            best_combined = float(cur_score) - center_penalty_lambda * dist_norm_current

    def _jitter_offsets(size: float):
        step = int(max(6, round(0.04 * size))) if sp_jitter_px <= 0 else int(sp_jitter_px)
        rings = max(1, int(sp_jitter_rings))
        offs = [(0, 0)]
        for r in range(1, rings + 1):
            d = r * step
            for dx in (-d, 0, d):
                for dy in (-d, 0, d):
                    if dx == 0 and dy == 0:
                        continue
                    offs.append((dx, dy))
        return offs

    for k in range(n_steps):
        scale = 1.0 + k * sp_escalate_step
        size = base_size * scale

        local_best_pref_pts = None
        local_best_pref_score = None
        local_best_pref_combined = best_combined
        local_best_pref_roi = None
        local_best_pref_debug = None

        local_best_any_pts = None
        local_best_any_score = None
        local_best_any_combined = best_combined
        local_best_any_roi = None
        local_best_any_debug = None

        for dx_j, dy_j in _jitter_offsets(size):
            x0 = cx - size / 2.0 + dx_j
            y0 = cy - size / 2.0 + dy_j
            x1 = cx + size / 2.0 + dx_j
            y1 = cy + size / 2.0 + dy_j

            dx = (x1 - x0) * sp_margin
            dy = (y1 - y0) * sp_margin
            x0, y0, x1, y1 = clip_rect(x0 - dx, y0 - dy, x1 + dx, y1 + dy, W, H)
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            roi_tuple = (x0, y0, x1, y1) if need_roi else None
            roi_size = float(max(x1 - x0, y1 - y0))

            rgb_roi = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
            rh2 = hands_sp.process(rgb_roi)
            if not rh2 or not getattr(rh2, "multi_hand_landmarks", None):
                continue

            for idx_hd, hd in enumerate(rh2.multi_handedness or []):
                label = str(hd.classification[0].label).lower()
                score = float(hd.classification[0].score)
                want = "left" if hand == "left" else "right"
                label_match = want in label
                if require_label_match and not label_match:
                    continue

                lms = xyz_list_from_lms(rh2.multi_hand_landmarks[idx_hd]) or []
                dist_norm = _dist_norm_from_lms(lms, x0, y0, x1, y1, roi_size)
                if dist_norm is not None and dist_norm > max_center_dist_norm:
                    continue
                dist_penalty = center_penalty_lambda * (dist_norm if dist_norm is not None else 0.0)
                combined_score = score - dist_penalty
                cand_debug = None
                if debug_out is not None:
                    cand_debug = {
                        "score": float(score),
                        "dist_norm": float(dist_norm) if dist_norm is not None else None,
                        "penalty": float(dist_penalty),
                        "combined": float(combined_score),
                        "label": label,
                        "match": bool(label_match),
                        "roi": [int(x0), int(y0), int(x1), int(y1)],
                        "scale": float(scale),
                    }
                    debug_out["candidates"].append(cand_debug)

                if world_coords and getattr(rh2, "multi_hand_world_landmarks", None):
                    cand = xyz_list_from_lms(rh2.multi_hand_world_landmarks[idx_hd])
                else:
                    px = [dict(
                        x=(x0 + p["x"] * (x1 - x0)),
                        y=(y0 + p["y"] * (y1 - y0)),
                        z=p["z"]
                    ) for p in lms]
                    cand = px_to_norm(px, W, H)

                if cand is None:
                    continue

                if label_match and combined_score >= local_best_pref_combined:
                    local_best_pref_pts = cand
                    local_best_pref_score = score
                    local_best_pref_combined = combined_score
                    if need_roi:
                        local_best_pref_roi = roi_tuple
                    local_best_pref_debug = cand_debug

                if combined_score >= local_best_any_combined:
                    local_best_any_pts = cand
                    local_best_any_score = score
                    local_best_any_combined = combined_score
                    if need_roi:
                        local_best_any_roi = roi_tuple
                    local_best_any_debug = cand_debug

        selected_pts = None
        selected_score = None
        selected_roi = None
        selected_debug = None
        if local_best_pref_pts is not None and local_best_any_pts is not None:
            if local_best_any_combined > local_best_pref_combined + label_relax_margin:
                selected_pts = local_best_any_pts
                selected_score = local_best_any_score
                selected_roi = local_best_any_roi
                selected_debug = local_best_any_debug
            else:
                selected_pts = local_best_pref_pts
                selected_score = local_best_pref_score
                selected_roi = local_best_pref_roi
                selected_debug = local_best_pref_debug
        elif local_best_pref_pts is not None:
            selected_pts = local_best_pref_pts
            selected_score = local_best_pref_score
            selected_roi = local_best_pref_roi
            selected_debug = local_best_pref_debug
        elif local_best_any_pts is not None:
            selected_pts = local_best_any_pts
            selected_score = local_best_any_score
            selected_roi = local_best_any_roi
            selected_debug = local_best_any_debug

        if selected_pts is not None:
            best_pts = selected_pts
            best_score = float(selected_score or 0.0)
            if need_roi:
                roi_selected = selected_roi
            if debug_out is not None:
                debug_out["selected"] = selected_debug
                if selected_debug is not None and selected_debug.get("dist_norm") is not None:
                    debug_out["selected_dist_norm"] = selected_debug.get("dist_norm")
                if roi_selected is not None:
                    debug_out["selected_roi_px"] = list(roi_selected)
            break

    if debug_out is not None:
        cands = debug_out.get("candidates") or []
        if len(cands) > 50:
            cands.sort(key=lambda c: c.get("combined", float("-inf")), reverse=True)
            debug_out["candidates"] = cands[:50]

    if best_pts is not None:
        if debug_out is not None and "selected" not in debug_out:
            debug_out["selected"] = None
        return SecondPassResult(
            landmarks=best_pts,
            score=best_score,
            recovered=True,
            roi=(roi_selected if return_roi else None),
            debug=debug_out,
        )
    if debug_out is not None and "selected" not in debug_out:
        debug_out["selected"] = None
    return SecondPassResult(
        landmarks=cur_pts,
        score=cur_score,
        recovered=False,
        roi=(roi_selected if return_roi else None),
        debug=debug_out,
    )
