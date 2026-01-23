
from __future__ import annotations
from typing import List, Dict, Optional
import numpy as np
from .. import _env  # noqa: F401  ensure caps applied before cv2 import
import cv2

def xyz_list_from_lms(lms) -> Optional[List[Dict[str, float]]]:
    if lms is None:
        return None
    return [dict(x=float(p.x), y=float(p.y), z=float(p.z)) for p in lms.landmark]

def pick_pose_indices(xyz: Optional[List[Dict[str, float]]], keep: Optional[List[int]]) -> Optional[List[Dict[str, float]]]:
    if xyz is None:
        return None
    if keep is None:
        return xyz
    if len(keep) == 0:
        return []
    out: List[Dict[str, float]] = []
    L = len(xyz)
    for idx in keep:
        if 0 <= idx < L:
            out.append(xyz[idx])
        else:
            out.append(dict(x=0.0, y=0.0, z=0.0))
    return out

def parse_keep_indices(s: str) -> Optional[List[int]]:
    s = (s or "").strip().lower()
    if s in ("", "none"):
        return []
    if s == "all":
        return None
    return [int(x) for x in s.replace(" ", "").split(",") if x != ""]

def resize_short_side(bgr, short_side: Optional[int]):
    if not short_side or short_side <= 0:
        return bgr
    h, w = bgr.shape[:2]
    ss = min(h, w)
    if ss <= short_side:
        return bgr
    scale = short_side / float(ss)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

def clip_rect(x0, y0, x1, y1, W, H):
    x0 = max(0, min(W-1, int(round(x0)))); y0 = max(0, min(H-1, int(round(y0))))
    x1 = max(0, min(W,   int(round(x1)))); y1 = max(0, min(H,   int(round(y1))))
    if x1 <= x0: x1 = min(W, x0+1)
    if y1 <= y0: y1 = min(H, y0+1)
    return x0, y0, x1, y1

def bbox_from_pts_px(hand_pts_px: List[Dict[str, float]]):
    xs = [p['x'] for p in hand_pts_px]; ys = [p['y'] for p in hand_pts_px]
    return (min(xs), min(ys), max(xs), max(ys))

def norm_to_px(pts_norm: List[Dict[str, float]], W: int, H: int) -> List[Dict[str, float]]:
    return [dict(x=p['x']*W, y=p['y']*H, z=p['z']) for p in pts_norm]

def px_to_norm(pts_px: List[Dict[str, float]], W: int, H: int) -> List[Dict[str, float]]:
    invW = 1.0/max(1, W); invH = 1.0/max(1, H)
    return [dict(x=p['x']*invW, y=p['y']*invH, z=p['z']) for p in pts_px]
