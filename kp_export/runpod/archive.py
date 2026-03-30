from __future__ import annotations

import shutil
import subprocess
import tarfile
from pathlib import Path


def archive_directory_to_tar_zst(source_dir: str | Path, out_path: str | Path) -> Path:
    src = Path(source_dir)
    dst = Path(out_path)
    if not src.exists():
        raise FileNotFoundError(f"Archive source not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tar_path = dst.with_suffix("").with_suffix(".tar")
    with tarfile.open(tar_path, "w") as tar:
        tar.add(src, arcname=src.name)
    zstd_bin = shutil.which("zstd")
    if not zstd_bin:
        raise RuntimeError("zstd command not found. Install zstd in the runtime image before archiving shards.")
    subprocess.run([zstd_bin, "-q", "-f", str(tar_path), "-o", str(dst)], check=True)
    tar_path.unlink(missing_ok=True)
    return dst
