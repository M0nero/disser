from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import orjson as _fastjson

    def _loads_bytes(b: bytes):
        return _fastjson.loads(b)
except Exception:  # pragma: no cover - optional dependency

    def _loads_bytes(b: bytes):
        return json.loads(b.decode("utf-8"))


def read_video_file_nocache(path_str: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Read per-video JSON -> (frames, meta), no global cache."""
    p = Path(path_str)
    with p.open("rb") as f:
        blob = _loads_bytes(f.read())
    if isinstance(blob, dict) and "frames" in blob:
        return blob["frames"], blob.get("meta", {})
    # legacy: file is a list of frames
    return blob, {}


def frames_from_combined(skel: Dict[str, Any], vid: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if "videos" in skel:
        v = skel["videos"].get(vid, {})
        return v.get("frames", []), v.get("meta", {})
    return skel.get(vid, []), {}

