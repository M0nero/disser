from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, List


def load_labels(path: Optional[Path]) -> Optional[List[str]]:
    if not path:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Labels file not found: {path}")
    if path.suffix.lower() in {".txt"}:
        return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if path.suffix.lower() in {".json"}:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(x) for x in data]
        raise ValueError("JSON labels must be an array of strings.")
    raise ValueError("Unsupported labels format. Use .txt (one per line) or .json (array).")
