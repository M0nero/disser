from __future__ import annotations

import json
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    tmp.replace(path)


class ShardStatusReporter:
    def __init__(
        self,
        *,
        run_id: str,
        total: int,
        shard_index: Optional[int] = None,
        num_shards: int = 1,
        status_path: str | Path = "",
        events_path: str | Path = "",
        failures_path: str | Path = "",
        pod_id: str = "",
        gpu_count: Optional[int] = None,
    ) -> None:
        self.run_id = str(run_id)
        self.total = int(total)
        self.shard_index = int(shard_index) if shard_index is not None else None
        self.num_shards = int(max(1, num_shards))
        self.status_path = Path(status_path) if str(status_path or "").strip() else None
        self.events_path = Path(events_path) if str(events_path or "").strip() else None
        self.failures_path = Path(failures_path) if str(failures_path or "").strip() else None
        self.hostname = socket.gethostname()
        self.pod_id = str(pod_id or "")
        self.gpu_count = int(gpu_count) if gpu_count is not None else None
        self.started_at = _utc_now()
        self.updated_at = self.started_at
        self._error_counts: Dict[str, int] = {}
        self._state = "created"

    def emit_event(self, event: str, **payload: Any) -> None:
        if self.events_path is None:
            return
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": _utc_now(),
            "run_id": self.run_id,
            "event": str(event),
            "shard_index": self.shard_index,
            "num_shards": self.num_shards,
            **payload,
        }
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def note_failure(self, sample_id: str, error: str) -> None:
        key = str(error or "").strip()
        if key:
            self._error_counts[key] = self._error_counts.get(key, 0) + 1
        if self.failures_path is not None:
            self.failures_path.parent.mkdir(parents=True, exist_ok=True)
            with self.failures_path.open("a", encoding="utf-8") as f:
                f.write(f"{sample_id}\n")
        self.emit_event("sample_failed", sample_id=str(sample_id), error=key)

    def update(
        self,
        *,
        state: str,
        processed: int,
        failed: int,
        remaining: int,
        videos_per_sec: Optional[float] = None,
        avg_sec_per_video: Optional[float] = None,
        eta_sec: Optional[float] = None,
        last_sample_id: str = "",
    ) -> None:
        self._state = str(state)
        self.updated_at = _utc_now()
        payload = {
            "run_id": self.run_id,
            "shard_index": self.shard_index,
            "num_shards": self.num_shards,
            "state": self._state,
            "selected": int(self.total),
            "processed": int(processed),
            "failed": int(failed),
            "remaining": int(remaining),
            "videos_per_sec": float(videos_per_sec) if videos_per_sec is not None else None,
            "avg_sec_per_video": float(avg_sec_per_video) if avg_sec_per_video is not None else None,
            "eta_sec": float(eta_sec) if eta_sec is not None else None,
            "hostname": self.hostname,
            "pod_id": self.pod_id,
            "gpu_count": self.gpu_count,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "last_sample_id": str(last_sample_id or ""),
            "error_summary": [
                {"error": error, "count": count}
                for error, count in sorted(self._error_counts.items(), key=lambda item: (-item[1], item[0]))
            ],
        }
        if self.status_path is not None:
            _atomic_write_json(self.status_path, payload)

