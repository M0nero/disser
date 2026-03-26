from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def load_family_map(path: str | Path, *, num_classes: int) -> dict[str, Any]:
    family_path = Path(path).expanduser()
    if not family_path.exists():
        raise FileNotFoundError(f"Family map not found: {family_path}")
    with family_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("family_map.json must contain a JSON object")

    class_to_family_raw = payload.get("class_to_family")
    if not isinstance(class_to_family_raw, dict):
        raise ValueError("family_map.json must contain a 'class_to_family' object mapping class_id -> family_id")

    mapping = [-1] * int(num_classes)
    seen: set[int] = set()
    for key, value in class_to_family_raw.items():
        try:
            class_id = int(key)
            family_id = int(value)
        except Exception as exc:
            raise ValueError(f"Invalid class_to_family entry {key!r}: {value!r}") from exc
        if not (0 <= class_id < int(num_classes)):
            raise ValueError(f"class_to_family contains out-of-range class_id={class_id} for num_classes={num_classes}")
        if family_id < 0:
            raise ValueError(f"class_to_family contains negative family_id={family_id} for class_id={class_id}")
        mapping[class_id] = family_id
        seen.add(family_id)

    missing = [idx for idx, fam in enumerate(mapping) if fam < 0]
    if missing:
        preview = ", ".join(str(idx) for idx in missing[:8])
        suffix = " ..." if len(missing) > 8 else ""
        raise ValueError(f"family_map.json is missing assignments for class ids: {preview}{suffix}")

    sorted_ids = sorted(seen)
    expected = list(range(len(sorted_ids)))
    if sorted_ids != expected:
        raise ValueError(
            "family ids must be contiguous in [0, num_families-1]; "
            f"got ids={sorted_ids[:12]}{' ...' if len(sorted_ids) > 12 else ''}"
        )

    declared_num_classes = payload.get("num_classes")
    if declared_num_classes is not None and int(declared_num_classes) != int(num_classes):
        raise ValueError(
            f"family_map.json num_classes={declared_num_classes} does not match current num_classes={num_classes}"
        )

    declared_num_families = payload.get("num_families")
    num_families = len(sorted_ids)
    if declared_num_families is not None and int(declared_num_families) != int(num_families):
        raise ValueError(
            f"family_map.json num_families={declared_num_families} does not match derived num_families={num_families}"
        )

    return {
        "path": str(family_path.resolve()),
        "version": int(payload.get("version", 1)),
        "num_classes": int(num_classes),
        "num_families": int(num_families),
        "class_to_family": mapping,
        "metadata": payload.get("metadata", {}),
    }


def build_class_to_family_tensor(family_map: dict[str, Any], *, device: torch.device | None = None) -> torch.Tensor:
    tensor = torch.tensor(family_map["class_to_family"], dtype=torch.long)
    if device is not None:
        tensor = tensor.to(device)
    return tensor
