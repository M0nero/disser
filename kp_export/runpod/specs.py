from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    input_root: str
    output_root: str
    run_root: str
    scratch_root: str
    pod_count: int
    gpu_type: str
    secure_cloud: bool
    network_volume_id: str
    container_image: str
    extractor_args: Dict[str, Any]
    shard_manifests: List[str]
    status_interval_sec: int = 30
    archive: Dict[str, Any] = field(default_factory=dict)
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    data_center_ids: List[str] = field(default_factory=list)
    ports: List[str] = field(default_factory=lambda: ["22/tcp"])
    docker_start_cmd: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    container_disk_gb: int = 50
    volume_mount_path: str = "/workspace"
    gpu_count: int = 1
    support_public_ip: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunSpec":
        return cls(
            run_id=str(data["run_id"]),
            input_root=str(data["input_root"]),
            output_root=str(data["output_root"]),
            run_root=str(data["run_root"]),
            scratch_root=str(data["scratch_root"]),
            pod_count=int(data["pod_count"]),
            gpu_type=str(data["gpu_type"]),
            secure_cloud=bool(data["secure_cloud"]),
            network_volume_id=str(data.get("network_volume_id") or ""),
            container_image=str(data["container_image"]),
            extractor_args=dict(data.get("extractor_args") or {}),
            shard_manifests=[str(x) for x in list(data.get("shard_manifests") or [])],
            status_interval_sec=int(data.get("status_interval_sec", 30)),
            archive=dict(data.get("archive") or {}),
            retry_policy=dict(data.get("retry_policy") or {}),
            data_center_ids=[str(x) for x in list(data.get("data_center_ids") or [])],
            ports=[str(x) for x in list(data.get("ports") or ["22/tcp"])],
            docker_start_cmd=[str(x) for x in list(data.get("docker_start_cmd") or [])],
            env={str(k): str(v) for k, v in dict(data.get("env") or {}).items()},
            container_disk_gb=int(data.get("container_disk_gb", 50)),
            volume_mount_path=str(data.get("volume_mount_path", "/workspace")),
            gpu_count=int(data.get("gpu_count", 1)),
            support_public_ip=bool(data.get("support_public_ip", True)),
        )

    def write_json(self, path: str | Path) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        return out_path

    @classmethod
    def read_json(cls, path: str | Path) -> "RunSpec":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
