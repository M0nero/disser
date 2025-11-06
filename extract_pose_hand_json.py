import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import cv2
import mediapipe as mp

# ----------------------------
# Helpers
# ----------------------------

def _to_xyz_list_world(lms) -> Optional[List[Dict[str, float]]]:
    if lms is None:
        return None
    # mediapipe world_landmarks: absolute (approx meters) coordinates
    return [dict(x=p.x, y=p.y, z=p.z) for p in lms.landmark]

def _to_xyz_list_image(lms) -> Optional[List[Dict[str, float]]]:
    if lms is None:
        return None
    # mediapipe image landmarks: normalized [0..1] image coords + relative z
    return [dict(x=p.x, y=p.y, z=p.z) for p in lms.landmark]

def _pick_pose_indices(xyz: Optional[List[Dict[str, float]]],
                       keep: Optional[List[int]]) -> Optional[List[Dict[str, float]]]:
    if xyz is None:
        return None
    if not keep:
        return xyz
    out = []
    L = len(xyz)
    for idx in keep:
        if 0 <= idx < L:
            out.append(xyz[idx])
        else:
            out.append(dict(x=0.0, y=0.0, z=0.0))
    return out

def _parse_keep_indices(s: str) -> Optional[List[int]]:
    if s.strip().lower() in ("", "all", "none", "null"):
        return None if s.lower() == "all" else []
    return [int(x) for x in s.replace(" ", "").split(",") if x != ""]

def _safe_int(x, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default

# ----------------------------
# Core per-video processing
# ----------------------------

def process_video(
    path: Path,
    out_path: Path,
    world_coords: bool,
    keep_pose_indices: Optional[List[int]],
    stride: int,
    short_side: Optional[int],
    min_det: float,
    min_track: float,
) -> Dict[str, Any]:
    """
    Processes single video and writes <out_path>.json:
    {
      "meta": {...},
      "frames": [
          { "ts": int(ms), "hand 1": [21{x,y,z}]|null, "hand 2": [...], "pose": [N{x,y,z}]|null },
          ...
      ]
    }
    Returns meta info for manifest.
    """
    mp_hands = mp.solutions.hands
    mp_pose  = mp.solutions.pose

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0:
            fps = 30.0  # fallback
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        # optional resize factor for speed (keep aspect by short side)
        def maybe_resize(bgr):
            nonlocal width, height
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

        # set up mediapipe
        hands_opts = dict(static_image_mode=False,
                          max_num_hands=2,
                          min_detection_confidence=min_det,
                          min_tracking_confidence=min_track)
        pose_opts  = dict(model_complexity=1,
                          enable_segmentation=False,
                          min_detection_confidence=min_det,
                          min_tracking_confidence=min_track)

        frames = []
        i = 0
        with mp_hands.Hands(**hands_opts) as hands, mp_pose.Pose(**pose_opts) as pose:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                if stride > 1 and (i % stride) != 0:
                    i += 1
                    continue

                bgr = maybe_resize(bgr)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                # Pose first (можно позже использовать ROI под кисти)
                rp = pose.process(rgb)

                # Hands next
                rh = hands.process(rgb)

                # Collect hands (Left/Right by handedness)
                left = right = None

                # Prefer world coords when available
                if world_coords and getattr(rh, "multi_hand_world_landmarks", None):
                    lm_list = rh.multi_hand_world_landmarks
                else:
                    lm_list = rh.multi_hand_landmarks

                if lm_list and rh.multi_handedness:
                    # mp гарантирует выравнивание landmarks <-> handedness по индексу
                    for lm, hd in zip(lm_list, rh.multi_handedness):
                        label = hd.classification[0].label.lower()
                        if "left" in label:
                            left = _to_xyz_list_world(lm) if world_coords else _to_xyz_list_image(lm)
                        elif "right" in label:
                            right = _to_xyz_list_world(lm) if world_coords else _to_xyz_list_image(lm)

                # Pose (33) → pick subset indices (или все)
                if world_coords and getattr(rp, "pose_world_landmarks", None):
                    pose_xyz = _to_xyz_list_world(rp.pose_world_landmarks)
                elif getattr(rp, "pose_landmarks", None):
                    pose_xyz = _to_xyz_list_image(rp.pose_landmarks)
                else:
                    pose_xyz = None

                pose_xyz = _pick_pose_indices(pose_xyz, keep_pose_indices)

                # timestamp (ms) согласно исходному fps и stride
                ts_ms = int(round(1000.0 * i / fps))
                frames.append({
                    "ts": ts_ms,
                    "hand 1": left,   # Левая рука
                    "hand 2": right,  # Правая рука
                    "pose": pose_xyz  # 33 или сабсет
                })
                i += 1

        # write JSON per video
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "meta": {
                    "video": path.name,
                    "fps": fps,
                    "size": [width, height],
                    "version": 2,
                    "coords": "world" if world_coords else "image",
                    "pose_indices": keep_pose_indices if keep_pose_indices is not None else "all"
                },
                "frames": frames
            }, f, ensure_ascii=False)

        return {
            "id": path.stem,
            "file": str(out_path),
            "num_frames": len(frames),
            "fps": fps
        }

    finally:
        cap.release()

# ----------------------------
# Combine to single JSON (streaming, memory-safe)
# ----------------------------

def combine_to_single_json(per_video_files: List[Path], combined_path: Path):
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with open(combined_path, "w", encoding="utf-8") as out:
        out.write("{")
        first = True
        for p in per_video_files:
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            # формат, совместимый с твоим датасетом:
            # "<video_id>": [ ...frames... ]
            vid = p.stem
            frames = obj.get("frames", [])
            if not first:
                out.write(",")
            first = False
            out.write(json.dumps(vid))
            out.write(":")
            out.write(json.dumps(frames, ensure_ascii=False))
        out.write("}")

# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Extract Pose+Hand landmarks to JSON (memory-safe, parallel).")
    ap.add_argument("--in-dir", type=str, required=True, help="Directory with input videos")
    ap.add_argument("--pattern", type=str, default="*.mp4", help="Glob pattern (e.g., *.mp4)")
    ap.add_argument("--out-dir", type=str, required=True, help="Directory to store per-video JSONs")
    ap.add_argument("--combined-json", type=str, default="", help="Optional path for combined single JSON (legacy format)")
    ap.add_argument("--keep-pose-indices", type=str, default="0,9,10,11,12,13,14,23,24",
                    help='Comma-separated indices to keep from Pose (use "all" to keep all 33)')
    ap.add_argument("--world-coords", action="store_true", help="Use world coordinates when available")
    ap.add_argument("--image-coords", action="store_true", help="Force image coords")
    ap.add_argument("--stride", type=int, default=1, help="Process every Nth frame")
    ap.add_argument("--short-side", type=int, default=0, help="Resize so that short side == value (0 = no resize)")
    ap.add_argument("--min-det", type=float, default=0.5, help="min_detection_confidence")
    ap.add_argument("--min-track", type=float, default=0.5, help="min_tracking_confidence")
    ap.add_argument("--jobs", type=int, default=1, help="Parallel processes over videos")
    args = ap.parse_args()

    if args.image_coords:
        world_coords = False
    else:
        world_coords = True if args.world_coords else True  # default to world

    keep = _parse_keep_indices(args.keep_pose_indices)
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(in_dir.glob(args.pattern))
    if not videos:
        raise SystemExit(f"No videos found by pattern: {in_dir}/{args.pattern}")

    manifest = []
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = []
            for v in videos:
                out_path = out_dir / f"{v.stem}.json"
                futs.append(ex.submit(
                    process_video, v, out_path, world_coords, keep,
                    max(1, args.stride), args.short_side, args.min_det, args.min_track
                ))
            for fut in as_completed(futs):
                meta = fut.result()
                manifest.append(meta)
    else:
        for v in videos:
            out_path = out_dir / f"{v.stem}.json"
            meta = process_video(v, out_path, world_coords, keep,
                                 max(1, args.stride), args.short_side, args.min_det, args.min_track)
            manifest.append(meta)

    # write manifest
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({"videos": manifest}, f, ensure_ascii=False, indent=2)

    # optional combined JSON (legacy format)
    if args.combined_json:
        combine_to_single_json([out_dir / f"{m['id']}.json" for m in manifest], Path(args.combined_json))

if __name__ == "__main__":
    main()