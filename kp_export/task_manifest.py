from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .tasks import TaskSpec


TASK_MANIFEST_SCHEMA = "kp_extract_tasks"
TASK_MANIFEST_VERSION = 1


def shard_index_for_sample(sample_id: str, num_shards: int) -> int:
    shards = max(1, int(num_shards))
    if shards == 1:
        return 0
    digest = hashlib.sha1(str(sample_id).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % shards


def filter_tasks_for_shard(
    tasks: Sequence[TaskSpec],
    *,
    num_shards: int,
    shard_index: int,
) -> List[TaskSpec]:
    shards = max(1, int(num_shards))
    index = int(shard_index)
    if index < 0 or index >= shards:
        raise ValueError(f"shard_index must be in [0, {shards - 1}]")
    if shards == 1:
        return list(tasks)
    return [task for task in tasks if shard_index_for_sample(task.sample_id, shards) == index]


def split_tasks(tasks: Sequence[TaskSpec], *, num_shards: int) -> List[List[TaskSpec]]:
    shards = [[] for _ in range(max(1, int(num_shards)))]
    for task in tasks:
        shards[shard_index_for_sample(task.sample_id, len(shards))].append(task)
    return shards


def _manifest_row(task: TaskSpec) -> Dict[str, object]:
    row = task.to_payload()
    row["manifest_schema"] = TASK_MANIFEST_SCHEMA
    row["manifest_version"] = TASK_MANIFEST_VERSION
    return row


def write_task_manifest(path: str | Path, tasks: Iterable[TaskSpec]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(_manifest_row(task), ensure_ascii=False, sort_keys=True) + "\n")
    return out_path


def load_task_manifest(path: str | Path) -> List[TaskSpec]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Task manifest not found: {manifest_path}")
    tasks: List[TaskSpec] = []
    for line_no, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = line.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except Exception as exc:
            raise RuntimeError(f"Invalid JSON on line {line_no} in {manifest_path}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"Invalid manifest row on line {line_no} in {manifest_path}: expected object")
        payload = dict(payload)
        payload.pop("manifest_schema", None)
        payload.pop("manifest_version", None)
        tasks.append(TaskSpec.from_payload(payload))
    if not tasks:
        raise RuntimeError(f"Task manifest is empty: {manifest_path}")
    return tasks
