from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

from kp_export.output.staging import write_staged_payload
from kp_export.output.writer import ExtractorOutputWriter
from kp_export.runpod.automation import aggregate_status_files, build_pod_create_payload, build_run_spec, write_shard_manifests
from kp_export.runpod.client import RunpodClient
from kp_export.runpod.merge import merge_shard_outputs
from kp_export.runpod.specs import RunSpec
from kp_export.runpod.status import ShardStatusReporter
from kp_export.runpod.validate import validate_artifact_root
from kp_export.task_manifest import filter_tasks_for_shard, load_task_manifest, shard_index_for_sample
from kp_export.tasks import TaskSpec
from scripts.runpod_extract import _all_shards_finished, _watch_payload
from tests.test_kp_output_writer import _sample_payload


class _FakeHTTPResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class RunpodSupportTests(unittest.TestCase):
    def test_task_manifest_sharding_is_deterministic(self) -> None:
        tasks = [
            TaskSpec(sample_id=f"sample_{idx}", slug=f"sample_{idx}", source_video=f"/tmp/{idx}.mp4", config_dict={})
            for idx in range(20)
        ]
        shard_a = [task.sample_id for task in filter_tasks_for_shard(tasks, num_shards=4, shard_index=1)]
        shard_b = [task.sample_id for task in filter_tasks_for_shard(tasks, num_shards=4, shard_index=1)]
        self.assertEqual(shard_a, shard_b)
        self.assertTrue(all(shard_index_for_sample(sample_id, 4) == 1 for sample_id in shard_a))

    def test_write_shard_manifests_and_load(self) -> None:
        tasks = [
            TaskSpec(sample_id=f"sample_{idx}", slug=f"sample_{idx}", source_video=f"/tmp/{idx}.mp4", config_dict={})
            for idx in range(6)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            all_manifest, shard_manifests = write_shard_manifests(run_root=tmp, tasks=tasks, num_shards=3)
            self.assertTrue(all_manifest.exists())
            self.assertEqual(len(shard_manifests), 3)
            loaded = load_task_manifest(all_manifest)
            self.assertEqual(len(loaded), 6)

    def test_status_reporter_writes_status_events_and_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reporter = ShardStatusReporter(
                run_id="run1",
                total=10,
                shard_index=2,
                num_shards=4,
                status_path=root / "status" / "shard-00002.json",
                events_path=root / "logs" / "shard-00002.events.jsonl",
                failures_path=root / "logs" / "shard-00002.failed.txt",
                pod_id="pod_123",
                gpu_count=1,
            )
            reporter.emit_event("run_started", selected=10)
            reporter.note_failure("sample_x", "BrokenProcessPool: boom")
            reporter.update(state="running", processed=3, failed=1, remaining=6, videos_per_sec=0.5, eta_sec=12.0)

            status = json.loads((root / "status" / "shard-00002.json").read_text(encoding="utf-8"))
            events = (root / "logs" / "shard-00002.events.jsonl").read_text(encoding="utf-8").splitlines()
            failures = (root / "logs" / "shard-00002.failed.txt").read_text(encoding="utf-8").splitlines()
            self.assertEqual(status["processed"], 3)
            self.assertEqual(status["failed"], 1)
            self.assertEqual(len(events), 2)
            self.assertEqual(failures, ["sample_x"])

    def test_merge_and_validate_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_roots = []
            for idx, sample_id in enumerate(["sample_a", "sample_b"]):
                shard_dir = root / f"shard_{idx}"
                writer = ExtractorOutputWriter(
                    out_dir=shard_dir,
                    scratch_dir=root / "scratch",
                    run_id=f"run_{idx}",
                    args_snapshot={"jobs": 1},
                    versions={"python": "test"},
                )
                staged = write_staged_payload(writer.current_run_dir, sample_id, _sample_payload(sample_id, with_pp=bool(idx)))
                writer.commit_staged_sample(staged)
                writer.finalize(
                    scheduled_count=1,
                    skipped_count=0,
                    failed_count=0,
                    processed_count=1,
                    segments_mode=False,
                    jobs=1,
                    seed=0,
                    mp_backend="tasks",
                    aggregate={"quality_score": 0.75},
                )
                shard_roots.append(shard_dir)

            merged = merge_shard_outputs(shard_roots, out_dir=root / "merged", run_id="merged_run")
            self.assertEqual(merged["samples"], 2)
            summary, errors = validate_artifact_root(root / "merged")
            self.assertEqual(errors, [])
            self.assertEqual(summary["sample_count_videos"], 2)
            runs = pq.read_table(root / "merged" / "runs.parquet").to_pylist()
            self.assertEqual(len(runs), 1)

    def test_build_pod_create_payload(self) -> None:
        spec = build_run_spec(
            run_id="run123",
            input_root="/workspace/datasets/in",
            output_root="/workspace/kp_export_runs",
            scratch_root="/tmp/kp_export",
            pod_count=2,
            gpu_type="NVIDIA RTX 4090",
            network_volume_id="vol_1",
            container_image="repo/image:latest",
            extractor_args={"argv": ["--image-coords"]},
            shard_manifests=["/workspace/kp_export_runs/run123/manifests/shard-00000.jsonl", "/workspace/kp_export_runs/run123/manifests/shard-00001.jsonl"],
        )
        payload = build_pod_create_payload(spec, shard_index=1)
        self.assertEqual(payload["name"], "kp-extract-run123-00001")
        self.assertEqual(payload["networkVolumeId"], "vol_1")
        self.assertIn("TASK_MANIFEST", payload["env"])

    def test_aggregate_status_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            status_dir = run_root / "status"
            status_dir.mkdir(parents=True)
            for idx in range(2):
                (status_dir / f"shard-{idx:05d}.json").write_text(
                    json.dumps({"selected": 10, "processed": idx + 1, "failed": 0, "remaining": 9 - idx}),
                    encoding="utf-8",
                )
            summary = aggregate_status_files(run_root)
            self.assertEqual(summary["selected"], 20)
            self.assertEqual(summary["processed"], 3)

    def test_watch_payload_and_terminal_detection(self) -> None:
        spec = build_run_spec(
            run_id="run123",
            input_root="/in",
            output_root="/tmp/out",
            scratch_root="/tmp/kp_export",
            pod_count=2,
            gpu_type="NVIDIA RTX 4090",
            network_volume_id="vol_1",
            container_image="repo/image:latest",
            extractor_args={"argv": ["--image-coords"]},
            shard_manifests=["a.jsonl", "b.jsonl"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run123"
            status_dir = run_root / "status"
            status_dir.mkdir(parents=True)
            (run_root / "pods.json").write_text(
                json.dumps([
                    {"id": "pod_a", "name": "pod-a", "status": "RUNNING"},
                    {"id": "pod_b", "name": "pod-b", "status": "RUNNING"},
                ]),
                encoding="utf-8",
            )
            for idx in range(2):
                (status_dir / f"shard-{idx:05d}.json").write_text(
                    json.dumps({"selected": 10, "processed": 10, "failed": 0, "remaining": 0, "state": "completed"}),
                    encoding="utf-8",
                )
            spec = RunSpec.from_dict({**spec.to_dict(), "run_root": str(run_root)})
            payload = _watch_payload(spec)
            self.assertEqual(payload["selected"], 20)
            self.assertTrue(_all_shards_finished(payload))

    def test_runpod_client_uses_rest_paths(self) -> None:
        calls = []

        def _fake_opener(request, timeout=0):
            calls.append((request.method, request.full_url, request.data))
            return _FakeHTTPResponse(json.dumps({"ok": True}).encode("utf-8"))

        client = RunpodClient(api_key="token", opener=_fake_opener)
        client.create_pod({"name": "x"})
        client.list_network_volumes()
        client.stop_pod("pod_1")
        client.delete_pod("pod_1")
        self.assertEqual(calls[0][0], "POST")
        self.assertTrue(calls[0][1].endswith("/pods"))
        self.assertEqual(calls[1][0], "GET")
        self.assertTrue(calls[1][1].endswith("/networkvolumes"))
        self.assertEqual(calls[2][0], "POST")
        self.assertTrue(calls[2][1].endswith("/pods/pod_1/stop"))
        self.assertEqual(calls[3][0], "DELETE")
        self.assertTrue(calls[3][1].endswith("/pods/pod_1"))

    def test_run_spec_roundtrip(self) -> None:
        spec = build_run_spec(
            run_id="run123",
            input_root="/in",
            output_root="/out",
            scratch_root="/tmp/kp_export",
            pod_count=2,
            gpu_type="NVIDIA RTX 4090",
            network_volume_id="vol_1",
            container_image="repo/image:latest",
            extractor_args={"argv": ["--image-coords"]},
            shard_manifests=["a.jsonl", "b.jsonl"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = spec.write_json(Path(tmp) / "run_spec.json")
            restored = RunSpec.read_json(path)
            self.assertEqual(restored.run_id, "run123")
            self.assertEqual(restored.shard_manifests, ["a.jsonl", "b.jsonl"])
