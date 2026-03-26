from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict


MANIFEST_NAME = "runtime_manifest.json"
MANIFEST_VERSION = 1


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def manifest_path(bundle_dir: str | Path) -> Path:
    root = Path(bundle_dir)
    if root.is_file():
        return root
    return root / MANIFEST_NAME


def write_runtime_manifest(bundle_dir: str | Path, payload: Dict[str, Any]) -> Path:
    root = Path(bundle_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "manifest_version": int(MANIFEST_VERSION),
        "created_ts": float(time.time()),
        **dict(payload or {}),
    }
    path = root / MANIFEST_NAME
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return path


def load_runtime_manifest(bundle_dir_or_manifest: str | Path) -> Dict[str, Any]:
    path = manifest_path(bundle_dir_or_manifest)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Runtime manifest must be a JSON object: {path}")
    payload.setdefault("manifest_version", MANIFEST_VERSION)
    payload["_manifest_path"] = str(path.resolve())
    payload["_bundle_dir"] = str(path.parent.resolve())
    return payload


def resolve_bundle_path(manifest: Dict[str, Any], rel_or_abs: str) -> Path:
    value = Path(str(rel_or_abs))
    if value.is_absolute():
        return value
    bundle_dir = Path(str(manifest.get("_bundle_dir", "")))
    if not bundle_dir:
        raise ValueError("Manifest does not contain _bundle_dir for relative path resolution")
    return (bundle_dir / value).resolve()


def copy_into_bundle(bundle_dir: str | Path, source: str | Path, *, rel_path: str) -> Path:
    src = Path(source).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    dst = Path(bundle_dir) / rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst
