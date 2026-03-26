from __future__ import annotations

import json
import hashlib
import subprocess
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


def git_sha(repo_root: Optional[Path] = None) -> str:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True)
    except Exception:
        return ""
    return out.strip()


def stable_sha(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def write_dataset_manifest(
    out_dir: Path,
    *,
    stage: str,
    args: Any,
    config_path: Optional[str] = None,
    section: Optional[str] = None,
    inputs: Optional[Dict[str, Any]] = None,
    counts: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if isinstance(args, dict):
        args_payload = dict(args)
    else:
        args_payload = dict(getattr(args, "__dict__", {}) or {})
    sha_payload = {
        "stage": str(stage),
        "section": str(section or ""),
        "config_path": str(config_path or ""),
        "args": args_payload,
        "inputs": dict(inputs or {}),
        "counts": dict(counts or {}),
        "extra": dict(extra or {}),
    }
    manifest: Dict[str, Any] = {
        "ts": time.time(),
        "stage": str(stage),
        "section": str(section or ""),
        "config_path": str(config_path or ""),
        "config_sha": stable_sha(sha_payload),
        "git_sha": git_sha(),
        "args": args_payload,
        "inputs": dict(inputs or {}),
        "counts": dict(counts or {}),
    }
    if extra:
        manifest["extra"] = dict(extra)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, default=str),
        encoding="utf-8",
    )
    return manifest
