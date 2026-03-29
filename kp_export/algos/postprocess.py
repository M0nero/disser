from __future__ import annotations

from copy import deepcopy
import math
import statistics
from typing import Any, Dict, List, Optional, Tuple

from ..core.geometry import hand_scale

MP_HAND_NUM_LANDMARKS = 21

EMA_ALPHA = 0.5
RTS_Q = 1e-4
SIGMA_ANCHOR = 0.01
SIGMA_OBS = 0.02
SIGMA_INTERP = 0.03
SIGMA_PRED = 0.06


def _copy_pts(pts: List[Dict[str, float]]) -> List[Dict[str, float]]:
    return [dict(p) for p in pts]


def _is_frame_record(frame: Any) -> bool:
    return hasattr(frame, "hand_1") and hasattr(frame, "hand_2") and hasattr(frame, "diagnostics")


def _legacy_hand_key(hand_idx: int) -> str:
    return f"hand {hand_idx}"


def _diag_key(hand_idx: int, field: str) -> str:
    return f"hand_{hand_idx}_{field}"


def _hand_obs(frame: Any, hand_idx: int):
    return frame.hand_1 if hand_idx == 1 else frame.hand_2


def _get_hand_pts(frame: Any, hand_idx: int):
    if _is_frame_record(frame):
        return _hand_obs(frame, hand_idx).landmarks
    return frame.get(_legacy_hand_key(hand_idx))


def _set_hand_pts(frame: Any, hand_idx: int, pts) -> None:
    if _is_frame_record(frame):
        _hand_obs(frame, hand_idx).landmarks = pts
        frame.both_hands = bool(frame.hand_1.landmarks is not None and frame.hand_2.landmarks is not None)
        return
    frame[_legacy_hand_key(hand_idx)] = pts
    frame["both_hands"] = 1 if (frame.get("hand 1") is not None and frame.get("hand 2") is not None) else 0


def _get_hand_score(frame: Any, hand_idx: int):
    if _is_frame_record(frame):
        return _hand_obs(frame, hand_idx).score
    return frame.get(f"hand {hand_idx}_score")


def _get_hand_source(frame: Any, hand_idx: int):
    if _is_frame_record(frame):
        return _hand_obs(frame, hand_idx).source
    return frame.get(f"hand {hand_idx}_source")


def _set_hand_source(frame: Any, hand_idx: int, value) -> None:
    if _is_frame_record(frame):
        _hand_obs(frame, hand_idx).source = value
        frame.diagnostics.values[_diag_key(hand_idx, "source")] = value
        return
    frame[f"hand {hand_idx}_source"] = value


def _get_hand_state(frame: Any, hand_idx: int):
    if _is_frame_record(frame):
        return _hand_obs(frame, hand_idx).state
    return frame.get(f"hand {hand_idx}_state")


def _set_hand_state(frame: Any, hand_idx: int, value) -> None:
    if _is_frame_record(frame):
        _hand_obs(frame, hand_idx).state = value
        frame.diagnostics.values[_diag_key(hand_idx, "state")] = value
        return
    frame[f"hand {hand_idx}_state"] = value


def _get_hand_reject(frame: Any, hand_idx: int):
    if _is_frame_record(frame):
        return _hand_obs(frame, hand_idx).reject_reason
    return frame.get(f"hand {hand_idx}_reject_reason")


def _get_hand_is_anchor(frame: Any, hand_idx: int):
    if _is_frame_record(frame):
        return _hand_obs(frame, hand_idx).is_anchor
    return frame.get(f"hand {hand_idx}_is_anchor")


def _get_ts(frame: Any):
    if _is_frame_record(frame):
        return frame.ts_ms
    return frame.get("ts")


def _set_extra(frame: Any, key: str, value: Any) -> None:
    if _is_frame_record(frame):
        frame.extras[key] = value
        return
    frame[key] = value


def _copy_frame(fr: Any) -> Any:
    return deepcopy(fr)


def _is_sanity_reject(reason: Optional[str]) -> bool:
    if not reason:
        return False
    return "sanity" in str(reason).lower()


def _is_anchor(fr: Dict[str, Any], hand_idx: int, hi: float) -> bool:
    pts = _get_hand_pts(fr, hand_idx)
    if pts is None:
        return False
    if _is_sanity_reject(_get_hand_reject(fr, hand_idx)):
        return False
    if _get_hand_is_anchor(fr, hand_idx):
        return True
    score = _get_hand_score(fr, hand_idx)
    source = _get_hand_source(fr, hand_idx)
    return score is not None and score >= hi and source in ("pass1", "pass2")


def _should_replace(fr: Dict[str, Any], hand_idx: int) -> bool:
    if _get_hand_pts(fr, hand_idx) is None:
        return True
    source = _get_hand_source(fr, hand_idx)
    if source in ("tracked", "occluded", "hold"):
        return True
    state = _get_hand_state(fr, hand_idx)
    if state is None or state != "observed":
        return True
    return False


def _anchor_repr(pts: List[Dict[str, float]]) -> Optional[Tuple[Tuple[float, float, float], float, List[Tuple[float, float, float]]]]:
    if not pts:
        return None
    n = min(MP_HAND_NUM_LANDMARKS, len(pts))
    wx = float(pts[0]["x"])
    wy = float(pts[0]["y"])
    wz = float(pts[0]["z"])
    scale = hand_scale(pts, min_points=min(2, MP_HAND_NUM_LANDMARKS))
    scale = max(scale, 1e-6)
    offsets: List[Tuple[float, float, float]] = []
    for j in range(n):
        px = float(pts[j]["x"])
        py = float(pts[j]["y"])
        pz = float(pts[j]["z"])
        offsets.append(((px - wx) / scale, (py - wy) / scale, (pz - wz) / scale))
    return (wx, wy, wz), scale, offsets


def _interp_pts(a0, a1, t: float) -> List[Dict[str, float]]:
    (w0, s0, off0) = a0
    (w1, s1, off1) = a1
    n = min(len(off0), len(off1))
    if n <= 0:
        return []
    wx = w0[0] * (1.0 - t) + w1[0] * t
    wy = w0[1] * (1.0 - t) + w1[1] * t
    wz = w0[2] * (1.0 - t) + w1[2] * t
    scale = s0 * (1.0 - t) + s1 * t
    pts: List[Dict[str, float]] = []
    for j in range(n):
        ox = off0[j][0] * (1.0 - t) + off1[j][0] * t
        oy = off0[j][1] * (1.0 - t) + off1[j][1] * t
        oz = off0[j][2] * (1.0 - t) + off1[j][2] * t
        x = wx + scale * ox
        y = wy + scale * oy
        z = wz + scale * oz
        x = min(1.0, max(0.0, x))
        y = min(1.0, max(0.0, y))
        pts.append({"x": float(x), "y": float(y), "z": float(z)})
    return pts


def _compute_dt_list(frames: List[Any]) -> List[float]:
    if not frames:
        return []
    ts_list = [_get_ts(fr) for fr in frames]
    if any(ts is None for ts in ts_list):
        return [1.0] * len(frames)
    diffs = []
    for i in range(1, len(ts_list)):
        dt = float(ts_list[i]) - float(ts_list[i - 1])
        if dt > 0:
            diffs.append(dt)
    if not diffs:
        return [1.0] * len(frames)
    med = float(statistics.median(diffs))
    if med <= 0:
        return [1.0] * len(frames)
    out = [1.0]
    for i in range(1, len(ts_list)):
        dt = (float(ts_list[i]) - float(ts_list[i - 1])) / med
        out.append(max(1e-6, float(dt)))
    return out


def _ema_bidirectional(seq: List[Optional[List[List[float]]]], alpha: float) -> List[Optional[List[List[float]]]]:
    n = len(seq)
    fwd: List[Optional[List[List[float]]]] = [None] * n
    prev = None
    for i in range(n):
        cur = seq[i]
        if cur is None:
            prev = None
            continue
        if prev is None:
            fwd[i] = [p[:] for p in cur]
        else:
            fwd[i] = [
                [
                    alpha * cur[j][0] + (1.0 - alpha) * prev[j][0],
                    alpha * cur[j][1] + (1.0 - alpha) * prev[j][1],
                    alpha * cur[j][2] + (1.0 - alpha) * prev[j][2],
                ]
                for j in range(len(cur))
            ]
        prev = fwd[i]
    bwd: List[Optional[List[List[float]]]] = [None] * n
    prev = None
    for i in range(n - 1, -1, -1):
        cur = seq[i]
        if cur is None:
            prev = None
            continue
        if prev is None:
            bwd[i] = [p[:] for p in cur]
        else:
            bwd[i] = [
                [
                    alpha * cur[j][0] + (1.0 - alpha) * prev[j][0],
                    alpha * cur[j][1] + (1.0 - alpha) * prev[j][1],
                    alpha * cur[j][2] + (1.0 - alpha) * prev[j][2],
                ]
                for j in range(len(cur))
            ]
        prev = bwd[i]
    out: List[Optional[List[List[float]]]] = [None] * n
    for i in range(n):
        if fwd[i] is None and bwd[i] is None:
            continue
        if fwd[i] is None:
            out[i] = bwd[i]
            continue
        if bwd[i] is None:
            out[i] = fwd[i]
            continue
        out[i] = [
            [
                0.5 * (fwd[i][j][0] + bwd[i][j][0]),
                0.5 * (fwd[i][j][1] + bwd[i][j][1]),
                0.5 * (fwd[i][j][2] + bwd[i][j][2]),
            ]
            for j in range(len(fwd[i]))
        ]
    return out


def _rts_smooth_1d(obs: List[Optional[float]], dt_list: List[float], r_list: List[float], q: float) -> List[Optional[float]]:
    n = len(obs)
    if n == 0:
        return []
    x_filt: List[Optional[List[float]]] = [None] * n
    P_filt: List[Optional[List[List[float]]]] = [None] * n
    x_pred: List[Optional[List[float]]] = [None] * n
    P_pred: List[Optional[List[List[float]]]] = [None] * n
    x: Optional[List[float]] = None
    P: Optional[List[List[float]]] = None
    for i in range(n):
        dt = dt_list[i] if i < len(dt_list) else 1.0
        if x is None:
            if obs[i] is None:
                continue
            x = [float(obs[i]), 0.0]
            P = [[1.0, 0.0], [0.0, 1.0]]
            x_filt[i] = x[:]
            P_filt[i] = [P[0][:], P[1][:]]
            x_pred[i] = x[:]
            P_pred[i] = [P[0][:], P[1][:]]
            continue
        x0 = x[0] + dt * x[1]
        x1 = x[1]
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        q11 = 0.25 * dt4 * q
        q12 = 0.5 * dt3 * q
        q22 = dt2 * q
        P00 = P[0][0] + dt * (P[1][0] + P[0][1]) + dt2 * P[1][1]
        P01 = P[0][1] + dt * P[1][1]
        P10 = P[1][0] + dt * P[1][1]
        P11 = P[1][1]
        P = [[P00 + q11, P01 + q12], [P10 + q12, P11 + q22]]
        x = [x0, x1]
        x_pred[i] = x[:]
        P_pred[i] = [P[0][:], P[1][:]]
        if obs[i] is not None:
            R = r_list[i] if i < len(r_list) else (SIGMA_OBS * SIGMA_OBS)
            y = float(obs[i]) - x[0]
            S = P[0][0] + R
            if S <= 0:
                S = 1e-6
            K0 = P[0][0] / S
            K1 = P[1][0] / S
            x[0] = x[0] + K0 * y
            x[1] = x[1] + K1 * y
            P00 = (1.0 - K0) * P[0][0]
            P01 = (1.0 - K0) * P[0][1]
            P10 = P[1][0] - K1 * P[0][0]
            P11 = P[1][1] - K1 * P[0][1]
            P = [[P00, P01], [P10, P11]]
        x_filt[i] = x[:]
        P_filt[i] = [P[0][:], P[1][:]]
    last = None
    for i in range(n - 1, -1, -1):
        if x_filt[i] is not None:
            last = i
            break
    if last is None:
        return [None] * n
    x_smooth: List[Optional[List[float]]] = [None] * n
    x_smooth[last] = x_filt[last]
    for i in range(last - 1, -1, -1):
        if x_filt[i] is None or x_pred[i + 1] is None or P_filt[i] is None or P_pred[i + 1] is None:
            x_smooth[i] = None
            continue
        dt = dt_list[i + 1] if (i + 1) < len(dt_list) else 1.0
        F00, F01, F10, F11 = 1.0, dt, 0.0, 1.0
        P0 = P_filt[i]
        Pp = P_pred[i + 1]
        det = Pp[0][0] * Pp[1][1] - Pp[0][1] * Pp[1][0]
        if abs(det) < 1e-12:
            x_smooth[i] = x_filt[i]
            continue
        inv00 = Pp[1][1] / det
        inv01 = -Pp[0][1] / det
        inv10 = -Pp[1][0] / det
        inv11 = Pp[0][0] / det
        PFt00 = P0[0][0] * F00 + P0[0][1] * F01
        PFt01 = P0[0][0] * F10 + P0[0][1] * F11
        PFt10 = P0[1][0] * F00 + P0[1][1] * F01
        PFt11 = P0[1][0] * F10 + P0[1][1] * F11
        C00 = PFt00 * inv00 + PFt01 * inv10
        C01 = PFt00 * inv01 + PFt01 * inv11
        C10 = PFt10 * inv00 + PFt11 * inv10
        C11 = PFt10 * inv01 + PFt11 * inv11
        dx0 = x_smooth[i + 1][0] - x_pred[i + 1][0]
        dx1 = x_smooth[i + 1][1] - x_pred[i + 1][1]
        x0 = x_filt[i][0] + C00 * dx0 + C01 * dx1
        x1 = x_filt[i][1] + C10 * dx0 + C11 * dx1
        x_smooth[i] = [x0, x1]
    out: List[Optional[float]] = [None] * n
    for i in range(n):
        if obs[i] is None:
            out[i] = None
        elif x_smooth[i] is not None:
            out[i] = float(x_smooth[i][0])
        elif x_filt[i] is not None:
            out[i] = float(x_filt[i][0])
    return out


def _measurement_variance(fr: Any, hand_idx: int, anchor_mask: bool) -> float:
    if anchor_mask:
        sigma = SIGMA_ANCHOR
    else:
        source = _get_hand_source(fr, hand_idx)
        state = _get_hand_state(fr, hand_idx)
        if source in ("pass1", "pass2") and state == "observed":
            sigma = SIGMA_OBS
        elif source == "interp":
            sigma = SIGMA_INTERP
        else:
            sigma = SIGMA_PRED
    return float(sigma * sigma)


def _extract_hand_arrays(frames: List[Any], hand_idx: int) -> List[Optional[List[List[float]]]]:
    out: List[Optional[List[List[float]]]] = []
    for fr in frames:
        pts = _get_hand_pts(fr, hand_idx)
        if not pts:
            out.append(None)
            continue
        arr = [[float(p["x"]), float(p["y"]), float(p["z"])] for p in pts]
        out.append(arr)
    return out


def _apply_smoothed(frames: List[Any], hand_idx: int, smoothed: List[Optional[List[List[float]]]]) -> None:
    for i, arr in enumerate(smoothed):
        if arr is None:
            continue
        _set_hand_pts(frames[i], hand_idx, [
            {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
            for p in arr
        ])


def _wrist(pts: Optional[List[Dict[str, float]]]) -> Optional[Tuple[float, float, float]]:
    if not pts:
        return None
    return float(pts[0]["x"]), float(pts[0]["y"]), float(pts[0]["z"])


def postprocess_sequence(
    frames: List[Any],
    *,
    hi: float,
    max_gap: int,
    smoother: str,
    only_anchors: bool,
    world_coords: bool,
) -> Tuple[List[Any], Dict[str, Any]]:
    frames_pp = [_copy_frame(fr) for fr in frames]
    stats = {
        "pp_filled_left": 0,
        "pp_filled_right": 0,
        "pp_gaps_filled_left": 0,
        "pp_gaps_filled_right": 0,
        "pp_smoothing_delta_left": 0.0,
        "pp_smoothing_delta_right": 0.0,
    }
    if world_coords:
        stats["skipped"] = True
        stats["reason"] = "world_coords"
        return frames_pp, stats
    if not frames_pp:
        return frames_pp, stats

    anchor_info = {}
    anchor_mask = {}
    for hand_idx in (1, 2):
        anchors: List[Tuple[int, List[Dict[str, float]]]] = []
        mask: List[bool] = []
        for i, fr in enumerate(frames):
            is_anchor = _is_anchor(fr, hand_idx, hi)
            mask.append(is_anchor)
            if is_anchor:
                pts = _get_hand_pts(fr, hand_idx)
                if isinstance(pts, list):
                    anchors.append((i, pts))
        anchor_info[hand_idx] = anchors
        anchor_mask[hand_idx] = mask

    for hand_idx in (1, 2):
        anchors = anchor_info[hand_idx]
        if len(anchors) < 2:
            continue
        filled = 0
        gaps_filled = 0
        for (i0, pts0), (i1, pts1) in zip(anchors, anchors[1:]):
            gap_len = i1 - i0 - 1
            if gap_len <= 0 or gap_len > max_gap:
                continue
            a0 = _anchor_repr(pts0)
            a1 = _anchor_repr(pts1)
            if a0 is None or a1 is None:
                continue
            filled_any = False
            span = float(i1 - i0)
            for k in range(i0 + 1, i1):
                if not only_anchors and not _should_replace(frames_pp[k], hand_idx):
                    continue
                t = (k - i0) / span
                interp_pts = _interp_pts(a0, a1, t)
                if not interp_pts:
                    continue
                _set_hand_pts(frames_pp[k], hand_idx, interp_pts)
                _set_hand_source(frames_pp[k], hand_idx, "interp")
                _set_hand_state(frames_pp[k], hand_idx, "predicted")
                _set_extra(frames_pp[k], f"hand {hand_idx}_pp_applied", True)
                _set_extra(frames_pp[k], f"hand {hand_idx}_pp_reason", "gap_fill")
                if only_anchors:
                    raw_fr = frames[k]
                    raw_pts = _get_hand_pts(raw_fr, hand_idx)
                    raw_src = _get_hand_source(raw_fr, hand_idx)
                    if raw_pts is not None and raw_src in ("pass1", "pass2") and not anchor_mask[hand_idx][k]:
                        _set_extra(frames_pp[k], f"hand {hand_idx}_pp_overrode", True)
                        _set_extra(frames_pp[k], f"hand {hand_idx}_pp_overrode_reason", "only_anchors")
                filled += 1
                filled_any = True
            if filled_any:
                gaps_filled += 1
        if hand_idx == 1:
            stats["pp_filled_left"] = filled
            stats["pp_gaps_filled_left"] = gaps_filled
        else:
            stats["pp_filled_right"] = filled
            stats["pp_gaps_filled_right"] = gaps_filled

    if smoother not in ("none", "ema", "rts"):
        smoother = "none"

    if smoother != "none":
        dt_list = _compute_dt_list(frames_pp)
        for hand_idx in (1, 2):
            before_wrist = [
                _wrist(_get_hand_pts(fr, hand_idx))
                for fr in frames_pp
            ]
            seq = _extract_hand_arrays(frames_pp, hand_idx)
            if smoother == "ema":
                smoothed = _ema_bidirectional(seq, EMA_ALPHA)
            else:
                r_list = [
                    _measurement_variance(fr, hand_idx, anchor_mask[hand_idx][i])
                    for i, fr in enumerate(frames_pp)
                ]
                smoothed = [None] * len(seq)
                if seq:
                    n_points = None
                    for arr in seq:
                        if arr is not None:
                            n_points = len(arr)
                            break
                    if n_points is not None:
                        smoothed = [
                            [p[:] for p in arr] if arr is not None else None
                            for arr in seq
                        ]
                        for j in range(n_points):
                            for dim in range(3):
                                obs: List[Optional[float]] = []
                                for arr in seq:
                                    if arr is None or j >= len(arr):
                                        obs.append(None)
                                    else:
                                        obs.append(arr[j][dim])
                                out = _rts_smooth_1d(obs, dt_list, r_list, RTS_Q)
                                for i, val in enumerate(out):
                                    if val is not None and smoothed[i] is not None:
                                        smoothed[i][j][dim] = float(val)
            _apply_smoothed(frames_pp, hand_idx, smoothed)
            for i, pts in anchor_info[hand_idx]:
                if pts is not None:
                    _set_hand_pts(frames_pp[i], hand_idx, _copy_pts(pts))
            delta_sum = 0.0
            delta_count = 0
            for i, fr in enumerate(frames_pp):
                if anchor_mask[hand_idx][i]:
                    continue
                b = before_wrist[i]
                a = _wrist(_get_hand_pts(fr, hand_idx))
                if b is None or a is None:
                    continue
                dx = a[0] - b[0]
                dy = a[1] - b[1]
                dz = a[2] - b[2]
                delta_sum += math.sqrt(dx * dx + dy * dy + dz * dz)
                delta_count += 1
            delta = delta_sum / float(delta_count) if delta_count else 0.0
            if hand_idx == 1:
                stats["pp_smoothing_delta_left"] = float(delta)
            else:
                stats["pp_smoothing_delta_right"] = float(delta)

    return frames_pp, stats
