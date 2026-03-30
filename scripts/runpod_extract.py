from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kp_export.runpod import (
    RunpodClient,
    aggregate_status_files,
    archive_directory_to_tar_zst,
    build_run_spec,
    ensure_network_volume,
    launch_run,
    merge_shard_outputs,
    validate_artifact_root,
    write_shard_manifests,
)
from kp_export.runpod.specs import RunSpec
from kp_export.task_manifest import load_task_manifest, write_task_manifest
from scripts.extract_keypoints import main as extract_main


def _strip_remainder_prefix(values: List[str]) -> List[str]:
    if values and values[0] == "--":
        return values[1:]
    return values


def _arg_value(argv: List[str], name: str, default: str = "") -> str:
    for idx, item in enumerate(argv):
        if item == name and idx + 1 < len(argv):
            return str(argv[idx + 1])
    return default


def _load_api_key(explicit: str) -> str:
    api_key = str(explicit or os.environ.get("RUNPOD_API_KEY") or "").strip()
    if not api_key:
        raise SystemExit("Runpod API key not found. Pass --api-key or set RUNPOD_API_KEY.")
    return api_key


def _load_pods_from_run_root(run_root: str | Path) -> List[Dict[str, Any]]:
    pods_path = Path(run_root) / "pods.json"
    if not pods_path.exists():
        return []
    data = json.loads(pods_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [dict(item) for item in data if isinstance(item, dict)]
    return []


def _pod_id(record: Dict[str, Any]) -> str:
    return str(record.get("id") or record.get("podId") or "").strip()


def _pod_name(record: Dict[str, Any]) -> str:
    return str(record.get("name") or record.get("podName") or "").strip()


def _pod_state(record: Dict[str, Any]) -> str:
    for key in ("desiredStatus", "status", "state"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return "unknown"


def _watch_payload(spec: RunSpec, *, pod_details: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    summary = aggregate_status_files(spec.run_root)
    pods = pod_details if pod_details is not None else _load_pods_from_run_root(spec.run_root)
    payload = {
        "run_id": spec.run_id,
        "run_root": spec.run_root,
        "selected": int(summary.get("selected", 0) or 0),
        "processed": int(summary.get("processed", 0) or 0),
        "failed": int(summary.get("failed", 0) or 0),
        "remaining": int(summary.get("remaining", 0) or 0),
        "shard_count": int(summary.get("shard_count", 0) or 0),
        "shards": list(summary.get("shards", [])),
        "pods": [
            {
                "id": _pod_id(pod),
                "name": _pod_name(pod),
                "state": _pod_state(pod),
            }
            for pod in pods
            if isinstance(pod, dict)
        ],
    }
    return payload


def _print_watch_payload(payload: Dict[str, Any], *, compact: bool = False) -> None:
    if not compact:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
        return
    run_id = str(payload.get("run_id", ""))
    processed = int(payload.get("processed", 0) or 0)
    selected = int(payload.get("selected", 0) or 0)
    failed = int(payload.get("failed", 0) or 0)
    remaining = int(payload.get("remaining", 0) or 0)
    states = {}
    for pod in payload.get("pods", []):
        state = str(pod.get("state", "unknown"))
        states[state] = states.get(state, 0) + 1
    state_text = ",".join(f"{name}:{count}" for name, count in sorted(states.items())) or "no-pod-api"
    line = f"[WATCH] run={run_id} done={processed}/{selected} failed={failed} remaining={remaining} | pods={state_text}"
    if sys.stdout.isatty():
        print(line.ljust(140), end="\r", flush=True)
    else:
        print(line)


def _all_shards_finished(payload: Dict[str, Any]) -> bool:
    shards = list(payload.get("shards", []))
    if not shards:
        return False
    terminal = {"completed", "completed_with_failures", "failed", "interrupted"}
    return all(str(shard.get("state", "")) in terminal for shard in shards)


def cmd_prepare(args: argparse.Namespace) -> int:
    extractor_args = _strip_remainder_prefix(list(args.extractor_args or []))
    if not extractor_args:
        raise SystemExit("prepare requires extractor args after '--'")

    run_root = Path(args.output_root) / args.run_id
    manifest_dir = run_root / "manifests"
    all_manifest = manifest_dir / "all_tasks.jsonl"
    prepare_artifact = run_root / "_prepare_artifact"

    prep_args = list(extractor_args)
    if "--out-dir" not in prep_args:
        prep_args.extend(["--out-dir", str(prepare_artifact)])
    prep_args.extend(["--write-task-manifest", str(all_manifest), "--prepare-only"])
    result = extract_main(prep_args, _print_errors=True)
    if not result.get("ok", False):
        return int(result.get("code", 1))

    tasks = load_task_manifest(all_manifest)
    _, shard_manifests = write_shard_manifests(run_root=run_root, tasks=tasks, num_shards=int(args.pod_count))

    spec = build_run_spec(
        run_id=str(args.run_id),
        input_root=_arg_value(prep_args, "--in-dir"),
        output_root=str(args.output_root),
        scratch_root=str(args.scratch_root),
        pod_count=int(args.pod_count),
        gpu_type=str(args.gpu_type),
        network_volume_id=str(args.network_volume_id or ""),
        container_image=str(args.container_image),
        extractor_args={"argv": extractor_args},
        shard_manifests=[str(path) for path in shard_manifests],
        data_center_ids=[str(x) for x in list(args.data_center_id or [])],
        archive={
            "enabled": bool(args.archive_s3_bucket or args.archive_local),
            "local": bool(args.archive_local),
            "s3_bucket": str(args.archive_s3_bucket or ""),
            "s3_prefix": str(args.archive_s3_prefix or ""),
        },
        retry_policy={"max_retries": int(args.max_retries)},
        env={},
    )
    spec_path = spec.write_json(run_root / "run_spec.json")
    print(f"[OK] all_tasks manifest: {all_manifest}")
    print(f"[OK] shard manifests: {len(shard_manifests)} under {manifest_dir}")
    print(f"[OK] run spec: {spec_path}")
    return 0


def cmd_launch(args: argparse.Namespace) -> int:
    spec_path = Path(args.run_spec)
    spec = RunSpec.read_json(spec_path)
    client = RunpodClient(api_key=_load_api_key(args.api_key))
    volume = ensure_network_volume(
        client,
        volume_id=spec.network_volume_id,
        volume_name=str(args.volume_name or ""),
        size_gb=int(args.volume_size_gb or 0),
        data_center_id=str(args.volume_data_center_id or ""),
    )
    volume_id = str(volume.get("id") or volume.get("networkVolumeId") or spec.network_volume_id)
    if volume_id != spec.network_volume_id:
        spec = RunSpec.from_dict({**spec.to_dict(), "network_volume_id": volume_id})
        spec.write_json(spec_path)

    pods = launch_run(client, spec)
    pods_path = Path(spec.run_root) / "pods.json"
    pods_path.parent.mkdir(parents=True, exist_ok=True)
    pods_path.write_text(json.dumps(pods, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    print(f"[OK] launched {len(pods)} pod(s)")
    print(f"[OK] pod metadata written to {pods_path}")
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    spec = RunSpec.read_json(args.run_spec)
    client = RunpodClient(api_key=_load_api_key(args.api_key)) if args.with_api else None
    exit_code = 0
    while True:
        pod_details: List[Dict[str, Any]] = []
        if client is not None:
            for pod in _load_pods_from_run_root(spec.run_root):
                pod_id = _pod_id(pod)
                if pod_id:
                    pod_details.append(client.get_pod(pod_id))
        payload = _watch_payload(spec, pod_details=pod_details or None)
        _print_watch_payload(payload, compact=bool(args.compact))
        if args.fail_on_errors and int(payload.get("failed", 0) or 0) > 0:
            exit_code = 1
        if not args.follow:
            break
        if _all_shards_finished(payload):
            if args.compact and sys.stdout.isatty():
                print()
            break
        time.sleep(max(1.0, float(args.interval)))
    return int(exit_code)


def cmd_merge(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root)
    shard_roots = sorted((run_root / "shards").glob("shard-*"))
    out_dir = Path(args.out_dir) if args.out_dir else (run_root / "merged")
    merged = merge_shard_outputs(shard_roots, out_dir=out_dir, run_id=args.run_id or "")
    print(json.dumps(merged, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    roots: List[Path] = []
    if args.artifact_root:
        roots = [Path(args.artifact_root)]
    else:
        run_root = Path(args.run_root)
        if args.include_shards:
            roots.extend(sorted((run_root / "shards").glob("shard-*")))
        if args.include_merged:
            roots.append(run_root / "merged")
    if not roots:
        raise SystemExit("No artifact roots selected for validation.")
    failed = False
    for root in roots:
        summary, errors = validate_artifact_root(root)
        print(json.dumps({"summary": summary, "errors": errors}, ensure_ascii=False, sort_keys=True, indent=2))
        if errors:
            failed = True
    return 1 if failed else 0


def cmd_retry(args: argparse.Namespace) -> int:
    all_tasks = load_task_manifest(args.all_tasks)
    wanted = set()
    for failure_path in [Path(p) for p in list(args.failure_file or [])]:
        if not failure_path.exists():
            continue
        for line in failure_path.read_text(encoding="utf-8").splitlines():
            sample_id = str(line.strip())
            if sample_id:
                wanted.add(sample_id)
    if not wanted:
        raise SystemExit("No failed sample ids found.")
    retry_tasks = [task for task in all_tasks if task.sample_id in wanted]
    if not retry_tasks:
        raise SystemExit("No tasks matched the failure set.")
    out_path = write_task_manifest(args.out_manifest, retry_tasks)
    print(f"[OK] retry manifest written to {out_path} ({len(retry_tasks)} tasks)")
    return 0


def cmd_terminate(args: argparse.Namespace) -> int:
    spec = RunSpec.read_json(args.run_spec)
    pods = _load_pods_from_run_root(spec.run_root)
    if not pods:
        raise SystemExit(f"No pods.json found under {spec.run_root}")
    client = RunpodClient(api_key=_load_api_key(args.api_key))
    failures = 0
    action = "stop" if args.stop_only else "delete"
    for pod in pods:
        pod_id = _pod_id(pod)
        if not pod_id:
            continue
        try:
            if args.stop_only:
                client.stop_pod(pod_id)
            else:
                client.delete_pod(pod_id)
            print(f"[OK] {action} {pod_id} {_pod_name(pod)}")
        except Exception as exc:
            failures += 1
            print(f"[ERROR] {action} {pod_id}: {exc}")
            if not args.best_effort:
                return 1
    return 1 if failures > 0 else 0


def cmd_pod_run(args: argparse.Namespace) -> int:
    raw_run_spec = str(args.run_spec or os.environ.get("RUN_SPEC_PATH", "")).strip()
    if not raw_run_spec:
        raise SystemExit("RUN_SPEC_PATH is required for pod-run.")
    run_spec_path = Path(raw_run_spec)
    spec = RunSpec.read_json(run_spec_path)
    shard_index = int(args.shard_index if args.shard_index is not None else os.environ.get("SHARD_INDEX", "0"))
    extractor_args = list(spec.extractor_args.get("argv") or [])
    task_manifest = str(args.task_manifest or os.environ.get("TASK_MANIFEST", "")).strip()
    out_dir = str(args.out_dir or os.environ.get("OUT_DIR", "")).strip()
    scratch_dir = str(args.scratch_dir or os.environ.get("SCRATCH_DIR", "")).strip()
    status_path = str(args.status_path or os.environ.get("STATUS_PATH", "")).strip()
    events_path = str(args.events_path or os.environ.get("EVENTS_PATH", "")).strip()
    failures_path = str(args.failures_path or os.environ.get("FAILURES_PATH", "")).strip()
    archive_out = str(os.environ.get("ARCHIVE_OUT", "")).strip()

    argv = list(extractor_args)
    argv.extend(
        [
            "--task-manifest", task_manifest,
            "--out-dir", out_dir,
            "--scratch-dir", scratch_dir,
            "--status-path", status_path,
            "--events-path", events_path,
            "--failures-path", failures_path,
            "--num-shards", str(spec.pod_count),
            "--shard-index", str(shard_index),
        ]
    )
    result = extract_main(argv, _print_errors=True)
    code = int(result.get("code", 1))
    if code == 0 and spec.archive.get("enabled") and archive_out:
        archive_directory_to_tar_zst(out_dir, archive_out)
    return code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Runpod automation for kp_export extraction.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ap_prepare = sub.add_parser("prepare")
    ap_prepare.add_argument("--run-id", required=True)
    ap_prepare.add_argument("--output-root", required=True)
    ap_prepare.add_argument("--scratch-root", default="/tmp/kp_export")
    ap_prepare.add_argument("--pod-count", type=int, required=True)
    ap_prepare.add_argument("--gpu-type", required=True)
    ap_prepare.add_argument("--container-image", required=True)
    ap_prepare.add_argument("--network-volume-id", default="")
    ap_prepare.add_argument("--data-center-id", action="append", default=[])
    ap_prepare.add_argument("--archive-s3-bucket", default="")
    ap_prepare.add_argument("--archive-s3-prefix", default="")
    ap_prepare.add_argument("--archive-local", action="store_true")
    ap_prepare.add_argument("--max-retries", type=int, default=1)
    ap_prepare.add_argument("extractor_args", nargs=argparse.REMAINDER)
    ap_prepare.set_defaults(func=cmd_prepare)

    ap_launch = sub.add_parser("launch")
    ap_launch.add_argument("--run-spec", required=True)
    ap_launch.add_argument("--api-key", default="")
    ap_launch.add_argument("--volume-name", default="")
    ap_launch.add_argument("--volume-size-gb", type=int, default=0)
    ap_launch.add_argument("--volume-data-center-id", default="")
    ap_launch.set_defaults(func=cmd_launch)

    ap_watch = sub.add_parser("watch")
    ap_watch.add_argument("--run-spec", required=True)
    ap_watch.add_argument("--with-api", action="store_true")
    ap_watch.add_argument("--api-key", default="")
    ap_watch.add_argument("--follow", action="store_true")
    ap_watch.add_argument("--interval", type=float, default=15.0)
    ap_watch.add_argument("--compact", action="store_true")
    ap_watch.add_argument("--fail-on-errors", action="store_true")
    ap_watch.set_defaults(func=cmd_watch)

    ap_merge = sub.add_parser("merge")
    ap_merge.add_argument("--run-root", required=True)
    ap_merge.add_argument("--out-dir", default="")
    ap_merge.add_argument("--run-id", default="")
    ap_merge.set_defaults(func=cmd_merge)

    ap_validate = sub.add_parser("validate")
    ap_validate.add_argument("--artifact-root", default="")
    ap_validate.add_argument("--run-root", default="")
    ap_validate.add_argument("--include-shards", action="store_true")
    ap_validate.add_argument("--include-merged", action="store_true")
    ap_validate.set_defaults(func=cmd_validate)

    ap_retry = sub.add_parser("retry")
    ap_retry.add_argument("--all-tasks", required=True)
    ap_retry.add_argument("--failure-file", action="append", required=True)
    ap_retry.add_argument("--out-manifest", required=True)
    ap_retry.set_defaults(func=cmd_retry)

    ap_terminate = sub.add_parser("terminate")
    ap_terminate.add_argument("--run-spec", required=True)
    ap_terminate.add_argument("--api-key", default="")
    ap_terminate.add_argument("--stop-only", action="store_true")
    ap_terminate.add_argument("--best-effort", action="store_true")
    ap_terminate.set_defaults(func=cmd_terminate)

    ap_pod = sub.add_parser("pod-run")
    ap_pod.add_argument("--run-spec", default="")
    ap_pod.add_argument("--task-manifest", default="")
    ap_pod.add_argument("--out-dir", default="")
    ap_pod.add_argument("--scratch-dir", default="")
    ap_pod.add_argument("--status-path", default="")
    ap_pod.add_argument("--events-path", default="")
    ap_pod.add_argument("--failures-path", default="")
    ap_pod.add_argument("--shard-index", type=int)
    ap_pod.set_defaults(func=cmd_pod_run)
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
