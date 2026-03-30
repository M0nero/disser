from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any, Iterator, Optional

import cv2

from ...core.utils import resize_short_side


@dataclass
class DecodedFrame:
    frame_index: int
    rel_index: int
    ts_ms: int
    dt_ms: int
    bgr: Any
    rgb: Any
    proc_w: int
    proc_h: int


@dataclass
class _DecodeEnd:
    error: Optional[BaseException] = None


def open_video_capture(path: Path, *, frame_start: int = 0):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    if frame_start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    width_src = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_src = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    return cap, width_src, height_src, fps


def iter_decoded_frames(
    cap,
    *,
    frame_start: int,
    frame_end: Optional[int],
    stride: int,
    short_side: Optional[int],
    fps: float,
    ts_source: str,
) -> Iterator[DecodedFrame]:
    last_ts: Optional[int] = None
    i = int(frame_start)
    while True:
        if frame_end is not None and i >= frame_end:
            break
        ok, bgr = cap.read()
        if not ok:
            break

        rel_i = i - frame_start
        if (rel_i % max(1, stride)) != 0:
            i += 1
            continue

        bgr = resize_short_side(bgr, short_side)
        proc_h, proc_w = bgr.shape[:2]

        ts = None
        if ts_source in ("auto", "pos_msec"):
            t = float(cap.get(cv2.CAP_PROP_POS_MSEC))
            ts = int(round(t)) if t > 0 else None
        if ts is None:
            ts = int(round((i / max(fps, 1e-6)) * 1000.0))
        dt = 0 if last_ts is None else max(0, ts - last_ts)
        last_ts = ts

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        yield DecodedFrame(
            frame_index=i,
            rel_index=rel_i,
            ts_ms=int(ts),
            dt_ms=int(dt),
            bgr=bgr,
            rgb=rgb,
            proc_w=int(proc_w),
            proc_h=int(proc_h),
        )
        i += 1


def iter_prefetched_decoded_frames(
    cap,
    *,
    frame_start: int,
    frame_end: Optional[int],
    stride: int,
    short_side: Optional[int],
    fps: float,
    ts_source: str,
    prefetch_frames: int,
) -> Iterator[DecodedFrame]:
    depth = max(1, int(prefetch_frames))
    q: Queue[DecodedFrame | _DecodeEnd] = Queue(maxsize=depth)

    def _producer() -> None:
        try:
            for item in iter_decoded_frames(
                cap,
                frame_start=frame_start,
                frame_end=frame_end,
                stride=stride,
                short_side=short_side,
                fps=fps,
                ts_source=ts_source,
            ):
                q.put(item)
            q.put(_DecodeEnd())
        except BaseException as exc:  # pragma: no cover - exercised via consumer path
            q.put(_DecodeEnd(error=exc))

    thread = Thread(target=_producer, name="kp-export-decode-prefetch", daemon=True)
    thread.start()

    while True:
        item = q.get()
        if isinstance(item, _DecodeEnd):
            thread.join(timeout=1.0)
            if item.error is not None:
                raise item.error
            break
        yield item
