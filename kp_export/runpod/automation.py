from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from ..task_manifest import split_tasks, write_task_manifest
from ..tasks import TaskSpec
from .client import RunpodClient
from .specs import RunSpec


def write_shard_manifests(
    *,
    run_root: str | Path,
    tasks: Iterable[TaskSpec],
    num_shards: int,
) -> Tuple[Path, List[Path]]:
    run_root_path = Path(run_root)
    manifest_dir = run_root_path / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    task_list = list(tasks)
    all_manifest = write_task_manifest(manifest_dir / "all_tasks.jsonl", task_list)
    shard_manifests: List[Path] = []
    for shard_index, shard_tasks in enumerate(split_tasks(task_list, num_shards=max(1, int(num_shards)))):
        shard_path = manifest_dir / f"shard-{shard_index:05d}.jsonl"
        write_task_manifest(shard_path, shard_tasks)
        shard_manifests.append(shard_path)
    return all_manifest, shard_manifests


def build_run_spec(
    *,
    run_id: str,
    input_root: str,
    output_root: str,
    scratch_root: str,
    pod_count: int,
    gpu_type: str,
    network_volume_id: str,
    container_image: str,
    extractor_args: Dict[str, Any],
    shard_manifests: List[str],
    data_center_ids: List[str] | None = None,
    archive: Dict[str, Any] | None = None,
    retry_policy: Dict[str, Any] | None = None,
    env: Dict[str, str] | None = None,
    docker_start_cmd: List[str] | None = None,
) -> RunSpec:
    run_root = Path(output_root) / run_id
    return RunSpec(
        run_id=str(run_id),
        input_root=str(input_root),
        output_root=str(output_root),
        run_root=str(run_root),
        scratch_root=str(scratch_root),
        pod_count=int(pod_count),
        gpu_type=str(gpu_type),
        secure_cloud=True,
        network_volume_id=str(network_volume_id),
        container_image=str(container_image),
        extractor_args=dict(extractor_args),
        shard_manifests=[str(x) for x in shard_manifests],
        archive=dict(archive or {}),
        retry_policy=dict(retry_policy or {}),
        data_center_ids=[str(x) for x in list(data_center_ids or [])],
        env={str(k): str(v) for k, v in dict(env or {}).items()},
        docker_start_cmd=[str(x) for x in list(docker_start_cmd or [])],
    )


def build_pod_create_payload(spec: RunSpec, *, shard_index: int) -> Dict[str, Any]:
    run_root = Path(spec.run_root)
    shard_manifest = spec.shard_manifests[int(shard_index)]
    shard_name = f"shard-{int(shard_index):05d}"
    shard_root = run_root / "shards" / shard_name
    status_root = run_root / "status"
    logs_root = run_root / "logs"
    archive_root = run_root / "archives"
    env = {
        **dict(spec.env),
        "RUN_SPEC_PATH": str(run_root / "run_spec.json"),
        "RUN_ID": str(spec.run_id),
        "SHARD_INDEX": str(int(shard_index)),
        "NUM_SHARDS": str(int(spec.pod_count)),
        "TASK_MANIFEST": str(shard_manifest),
        "OUT_DIR": str(shard_root),
        "SCRATCH_DIR": str(Path(spec.scratch_root) / spec.run_id / shard_name),
        "STATUS_PATH": str(status_root / f"{shard_name}.json"),
        "EVENTS_PATH": str(logs_root / f"{shard_name}.events.jsonl"),
        "FAILURES_PATH": str(logs_root / f"{shard_name}.failed_samples.txt"),
        "STDERR_LOG_PATH": str(logs_root / f"{shard_name}.stderr.log"),
        "ARCHIVE_OUT": str(archive_root / f"{shard_name}.tar.zst"),
    }
    payload = {
        "name": f"kp-extract-{spec.run_id}-{int(shard_index):05d}",
        "imageName": spec.container_image,
        "cloudType": "SECURE",
        "computeType": "GPU",
        "gpuCount": int(spec.gpu_count),
        "gpuTypeIds": [spec.gpu_type],
        "gpuTypePriority": "custom",
        "containerDiskInGb": int(spec.container_disk_gb),
        "volumeMountPath": str(spec.volume_mount_path),
        "networkVolumeId": str(spec.network_volume_id),
        "ports": list(spec.ports),
        "supportPublicIp": bool(spec.support_public_ip),
        "env": env,
    }
    if spec.data_center_ids:
        payload["dataCenterIds"] = list(spec.data_center_ids)
        payload["dataCenterPriority"] = "custom"
    payload["dockerStartCmd"] = list(spec.docker_start_cmd or ["bash", "/app/scripts/runpod_entrypoint.sh"])
    return payload


def ensure_network_volume(
    client: RunpodClient,
    *,
    volume_id: str = "",
    volume_name: str = "",
    size_gb: int = 0,
    data_center_id: str = "",
) -> Dict[str, Any]:
    if volume_id:
        return client.get_network_volume(volume_id)
    existing = client.list_network_volumes()
    items = existing if isinstance(existing, list) else existing.get("items") or existing.get("data") or []
    for item in items:
        if str(item.get("name", "")).strip() == str(volume_name).strip():
            return item
    if not volume_name or not size_gb or not data_center_id:
        raise RuntimeError("Need existing volume_id or volume_name + size_gb + data_center_id to provision a volume")
    return client.create_network_volume(name=volume_name, size_gb=int(size_gb), data_center_id=str(data_center_id))


def launch_run(client: RunpodClient, spec: RunSpec) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for shard_index in range(spec.pod_count):
        payload = build_pod_create_payload(spec, shard_index=shard_index)
        results.append(client.create_pod(payload))
    return results


def aggregate_status_files(run_root: str | Path) -> Dict[str, Any]:
    status_dir = Path(run_root) / "status"
    files = sorted(status_dir.glob("shard-*.json"))
    shards: List[Dict[str, Any]] = []
    for path in files:
        try:
            shards.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            shards.append({"path": str(path), "state": "invalid"})
    total_selected = sum(int(s.get("selected", 0) or 0) for s in shards)
    total_processed = sum(int(s.get("processed", 0) or 0) for s in shards)
    total_failed = sum(int(s.get("failed", 0) or 0) for s in shards)
    total_remaining = sum(int(s.get("remaining", 0) or 0) for s in shards)
    return {
        "shards": shards,
        "shard_count": len(shards),
        "selected": total_selected,
        "processed": total_processed,
        "failed": total_failed,
        "remaining": total_remaining,
    }
