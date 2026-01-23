#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare no_event skeleton JSONs from a list of IDs or a whole source folder.

Reads IDs (one per line), copies JSONs from src_dir to dst_dir, and optionally
compacts payloads to the legacy structure expected by older code.
If --ids is not provided, processes all JSONs found in src_dir.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


META_KEYS = {
    "video",
    "fps",
    "size_src",
    "size_proc",
    "version",
    "coords",
    "mp_backend",
    "mp_models",
    "mp_tasks_delegate",
    "pose_indices",
    "hand_mapping",
    "second_pass",
    "second_pass_params",
    "sp_debug_roi",
    "interp_hold",
}

FRAME_KEYS = {
    "ts",
    "dt",
    "hand 1",
    "hand 1_score",
    "hand 1_source",
    "hand 2",
    "hand 2_score",
    "hand 2_source",
    "pose",
    "pose_vis",
    "pose_interpolated",
}


def _read_lines(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _compact_meta(meta: Dict[str, Any], stem: str, alt_used: bool) -> Dict[str, Any]:
    out = {k: meta.get(k) for k in META_KEYS if k in meta}
    if alt_used and "video" in out and isinstance(out["video"], str):
        suffix = Path(out["video"]).suffix or ".mp4"
        out["video"] = f"{stem}{suffix}"
    return out


def _compact_frame(fr: Dict[str, Any]) -> Dict[str, Any]:
    return {k: fr.get(k) for k in FRAME_KEYS if k in fr}


def _compact_payload(payload: Any, stem: str, alt_used: bool) -> Dict[str, Any]:
    if isinstance(payload, dict):
        frames = payload.get("frames", [])
        meta = payload.get("meta", {})
    elif isinstance(payload, list):
        frames = payload
        meta = {}
    else:
        raise ValueError("Unsupported JSON payload shape")
    meta_out = _compact_meta(meta if isinstance(meta, dict) else {}, stem, alt_used)
    frames_out = [_compact_frame(fr) for fr in frames if isinstance(fr, dict)]
    return {"meta": meta_out, "frames": frames_out}


def _iter_ids(lines: Iterable[str]) -> Iterable[str]:
    for raw in lines:
        stem = Path(raw).stem
        if stem:
            yield stem


def _gather_sources_from_dir(src_dir: Path, use_pp: bool) -> List[Tuple[str, Path]]:
    base: Dict[str, Path] = {}
    pp: Dict[str, Path] = {}
    for p in src_dir.glob("*.json"):
        name = p.name
        if name.endswith("_pp.json"):
            stem = name[:-8]
            pp[stem] = p
        else:
            base[p.stem] = p
    stems = sorted(set(base.keys()) | set(pp.keys()))
    out: List[Tuple[str, Path]] = []
    for stem in stems:
        src = pp.get(stem) if use_pp else base.get(stem)
        if src is None:
            src = base.get(stem) if use_pp else pp.get(stem)
        if src is not None:
            out.append((stem, src))
    return out


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser("Prepare no_event JSONs with optional legacy compaction")
    ap.add_argument("--ids", type=str, default="", help="Path to removed_ids.txt (one id per line)")
    ap.add_argument("--src-dir", type=str, required=True, help="Directory with source JSONs")
    ap.add_argument("--dst-dir", type=str, required=True, help="Output directory for no_event JSONs")
    ap.add_argument("--compact", action="store_true", default=True, help="Drop extra keys to legacy format")
    ap.add_argument("--no-compact", dest="compact", action="store_false")
    ap.add_argument("--use-pp", action="store_true",
                    help="Prefer *_pp.json files from src-dir (fallback to base JSON if missing)")
    ap.add_argument("--strip-no-prefix", action="store_true",
                    help="If id starts with 'no' and file is missing, try without the prefix")
    ap.add_argument("--dry-run", action="store_true", help="List planned actions without writing files")
    args = ap.parse_args(argv)

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)

    if not src_dir.exists():
        raise FileNotFoundError(src_dir)

    ids: List[str] = []
    sources: List[Tuple[str, Path]] = []
    if args.ids:
        ids_path = Path(args.ids)
        if not ids_path.exists():
            raise FileNotFoundError(ids_path)
        ids = list(_iter_ids(_read_lines(ids_path)))
        if not ids:
            raise RuntimeError("No IDs found in list.")
    else:
        sources = _gather_sources_from_dir(src_dir, use_pp=bool(args.use_pp))
        if not sources:
            raise RuntimeError("No JSON files found in src-dir.")

    missing: List[str] = []
    written = 0

    if not args.dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)

    if ids:
        iter_list = [(stem, None) for stem in ids]
    else:
        iter_list = sources

    for stem, preset_src in iter_list:
        src_path = preset_src
        alt_used = False
        if src_path is None:
            pp_path = src_dir / f"{stem}_pp.json"
            src_path = pp_path if args.use_pp and pp_path.exists() else (src_dir / f"{stem}.json")
            if not src_path.exists() and args.strip_no_prefix and stem.startswith("no"):
                alt = stem[2:]
                alt_pp = src_dir / f"{alt}_pp.json"
                alt_path = alt_pp if args.use_pp and alt_pp.exists() else (src_dir / f"{alt}.json")
                if alt_path.exists():
                    src_path = alt_path
                    alt_used = True
                else:
                    missing.append(stem)
                    continue
            elif not src_path.exists():
                missing.append(stem)
                continue

        if args.dry_run:
            print(f"[DRY] {src_path} -> {dst_dir / (stem + '.json')}")
            written += 1
            continue

        payload = _load_json(src_path)
        out_payload = _compact_payload(payload, stem, alt_used) if args.compact else payload

        out_path = dst_dir / f"{stem}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(out_payload, f, ensure_ascii=True, indent=2)
        written += 1

    print(f"[OK] Written: {written}")
    if missing:
        print(f"[WARN] Missing: {len(missing)}")
        for m in missing[:20]:
            print("  -", m)
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")


if __name__ == "__main__":
    main()
