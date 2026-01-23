from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


def load_config_section(config_path: Optional[str], section: str) -> Dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}
    if section and isinstance(raw.get(section), dict):
        return dict(raw.get(section, {}))
    return dict(raw)


def write_run_config(
    out_dir: Path,
    args: Any,
    *,
    config_path: Optional[str] = None,
    section: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "ts": time.time(),
        "section": section or "",
        "config_path": str(config_path or ""),
        "args": dict(getattr(args, "__dict__", {}) or {}),
    }
    if extra:
        payload["extra"] = dict(extra)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, default=str),
        encoding="utf-8",
    )
