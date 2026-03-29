from __future__ import annotations

import json
from pathlib import Path
from typing import IO, Any, List, Optional

from ..contracts import FrameRecord
from ..records.legacy import legacy_frame_from_record


def emit_ndjson_line(handle: Optional[IO[str]], payload: dict[str, Any]) -> None:
    if handle is None:
        return
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_smoothed_ndjson(frame_records: List[FrameRecord], ndjson_path: Optional[Path]) -> None:
    if ndjson_path is None:
        return
    smoothed_ndjson_path = ndjson_path.with_name(f"{ndjson_path.stem}_smoothed.ndjson")
    with open(smoothed_ndjson_path, "wt", encoding="utf-8") as smoothed_f:
        for record in frame_records:
            emit_ndjson_line(smoothed_f, legacy_frame_from_record(record))
