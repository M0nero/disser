from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Copy a deterministic subset of videos while preserving relative paths.")
    ap.add_argument("--in-dir", required=True, help="Root directory containing source videos.")
    ap.add_argument("--out-dir", required=True, help="Destination root for the copied subset.")
    ap.add_argument("--pattern", default="**/*.mp4", help="Glob pattern relative to --in-dir.")
    ap.add_argument("--count", type=int, default=100, help="Number of videos to copy.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed used for deterministic sampling.")
    ap.add_argument("--manifest", default="", help="Optional path to write the selected relative file paths.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output directory contents.")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    in_dir = Path(args.in_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if not in_dir.exists():
        raise SystemExit(f"Input directory not found: {in_dir}")

    videos = sorted(p for p in in_dir.glob(args.pattern) if p.is_file())
    if not videos:
        raise SystemExit(f"No videos found: {in_dir}/{args.pattern}")

    count = max(0, min(int(args.count), len(videos)))
    rng = random.Random(int(args.seed))
    selected = sorted(rng.sample(videos, count), key=lambda p: str(p.relative_to(in_dir)).lower())

    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rel_paths: list[str] = []
    for src in selected:
        rel = src.relative_to(in_dir)
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        rel_paths.append(rel.as_posix())

    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else (out_dir / "subset_manifest.txt")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(rel_paths) + ("\n" if rel_paths else ""), encoding="utf-8")

    print(f"[OK] copied {len(rel_paths)} video(s) to {out_dir}")
    print(f"[OK] manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
