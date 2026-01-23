
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional
import json

def _as_paths(per_video_files: Iterable[Path] | Path | str) -> list[Path]:
    if isinstance(per_video_files, (str, Path)):
        root = Path(per_video_files)
        if root.is_dir():
            return sorted(root.glob("*.json"))
        return [root]
    return [Path(p) for p in per_video_files]


def _select_prefer_pp(files: Iterable[Path], prefer_pp: bool) -> list[Path]:
    by_id: dict[str, dict[str, Path]] = {}
    for f in files:
        p = Path(f)
        stem = p.stem
        if stem.endswith("_pp"):
            key = stem[:-3]
            by_id.setdefault(key, {})["pp"] = p
        else:
            by_id.setdefault(stem, {})["raw"] = p

    chosen: list[Path] = []
    for entry in by_id.values():
        if prefer_pp and "pp" in entry:
            chosen.append(entry["pp"])
        elif "raw" in entry:
            chosen.append(entry["raw"])
        elif "pp" in entry:
            chosen.append(entry["pp"])
    return sorted(chosen, key=lambda p: p.name)


def _vid_from_path(path: Path) -> str:
    stem = path.stem
    return stem[:-3] if stem.endswith("_pp") else stem


def _looks_like_frames_list(data: list) -> bool:
    if not data:
        return True
    first = data[0]
    if not isinstance(first, dict):
        return False
    keys = set(first.keys())
    expected = {
        "hand 1",
        "hand 2",
        "left_hand",
        "right_hand",
        "hands",
        "pose",
        "pose_landmarks",
        "landmarks",
    }
    return bool(keys & expected)


def combine_to_single_json(
    per_video_files: Iterable[Path] | Path | str,
    combined_path: Path | str,
    include_meta: bool = False,
    prefer_pp: bool = True,
    with_meta: Optional[bool] = None,
) -> None:
    if with_meta is not None:
        include_meta = bool(with_meta)
    files = _select_prefer_pp(_as_paths(per_video_files), prefer_pp=prefer_pp)
    combined_path = Path(combined_path)
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with combined_path.open("w", encoding="utf-8") as out:
        out.write("{\"videos\":{" if include_meta else "{")
        first = True
        for jf in files:
            try:
                with Path(jf).open("r", encoding="utf-8") as f:
                    data = json.load(f)
                vid = _vid_from_path(Path(jf))
                if isinstance(data, dict):
                    if "frames" not in data or not isinstance(data.get("frames"), list):
                        raise ValueError("Unsupported JSON payload")
                    frames = data.get("frames", [])
                    meta = data.get("meta", {})
                elif isinstance(data, list):
                    if not _looks_like_frames_list(data):
                        raise ValueError("Unsupported JSON payload")
                    frames = data
                    meta = {}
                else:
                    raise ValueError("Unsupported JSON payload")
                item = {"meta": meta, "frames": frames} if include_meta else frames
            except Exception as e:
                print(f"[WARN] skip {jf}: {e}")
                continue
            if not first:
                out.write(",")
            first = False
            out.write(json.dumps(vid))
            out.write(":")
            out.write(json.dumps(item, ensure_ascii=False))
        out.write("}" if not include_meta else "}}")
        out.flush()
