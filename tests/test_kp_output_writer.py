from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import zarr

from kp_export.output.staging import write_staged_payload
from kp_export.output.writer import ExtractorOutputWriter


def _sample_payload(sample_id: str, *, with_pp: bool) -> dict:
    raw_arrays = {
        "ts_ms": np.asarray([0, 33], dtype=np.int64),
        "left_xyz": np.zeros((2, 21, 3), dtype=np.float32),
        "right_xyz": np.ones((2, 21, 3), dtype=np.float32),
        "left_score": np.asarray([0.9, 0.8], dtype=np.float32),
        "right_score": np.asarray([0.7, 0.6], dtype=np.float32),
        "left_valid": np.asarray([True, True], dtype=np.bool_),
        "right_valid": np.asarray([True, False], dtype=np.bool_),
        "pose_xyz": np.zeros((2, 5, 3), dtype=np.float32),
        "pose_vis": np.ones((2, 5), dtype=np.float32),
    }
    pp_arrays = None
    if with_pp:
        pp_arrays = {
            key: (value.copy() if isinstance(value, np.ndarray) else value)
            for key, value in raw_arrays.items()
        }
        pp_arrays["left_xyz"][1, 0, 0] = 42.0

    return {
        "sample_id": sample_id,
        "slug": sample_id,
        "source_video": f"/tmp/{sample_id}.mp4",
        "sample_attrs": {
            "sample_id": sample_id,
            "source_video": f"/tmp/{sample_id}.mp4",
            "slug": sample_id,
            "coords_mode": "image",
            "pose_indices": [0, 1, 2, 3, 4],
            "num_frames": 2,
            "fps": 30.0,
            "fps_est": 30.3,
            "postprocess_enabled": with_pp,
        },
        "video_row": {
            "sample_id": sample_id,
            "slug": sample_id,
            "source_video": f"/tmp/{sample_id}.mp4",
            "num_frames": 2,
            "pose_joint_count": 5,
            "coords_mode": "image",
            "fps": 30.0,
            "fps_est": 30.3,
            "quality_score": 0.75,
            "hands_coverage": 1.0,
            "swap_frames": 0,
            "missing_frames_1": 0,
            "missing_frames_2": 1,
            "outlier_frames_1": 0,
            "outlier_frames_2": 0,
            "sanity_reject_frames_1": 0,
            "sanity_reject_frames_2": 0,
            "occluded_frames_1": 0,
            "occluded_frames_2": 0,
            "pp_filled_left": 1 if with_pp else 0,
            "pp_filled_right": 0,
            "pp_smoothing_delta_left": 0.1 if with_pp else 0.0,
            "pp_smoothing_delta_right": 0.2 if with_pp else 0.0,
        },
        "frame_rows": [
            {
                "frame_idx": 0,
                "ts_ms": 0,
                "dt_ms": 0,
                "hand_1_present": True,
                "hand_2_present": True,
                "pose_present": True,
                "both_hands": True,
                "pose_interpolated": False,
            },
            {
                "frame_idx": 1,
                "ts_ms": 33,
                "dt_ms": 33,
                "hand_1_present": True,
                "hand_2_present": False,
                "pose_present": True,
                "both_hands": False,
                "pose_interpolated": False,
            },
        ],
        "raw_arrays": raw_arrays,
        "pp_arrays": pp_arrays,
    }


class ExtractorOutputWriterTests(unittest.TestCase):
    def test_writer_creates_raw_only_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            writer = ExtractorOutputWriter(
                out_dir=out_dir,
                run_id="run_raw",
                args_snapshot={"jobs": 1},
                versions={"python": "test"},
            )
            staged = write_staged_payload(writer.current_run_dir, "sample_raw", _sample_payload("sample_raw", with_pp=False))
            video_row = writer.commit_staged_sample(staged)
            paths = writer.finalize(
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

            self.assertEqual(video_row["sample_id"], "sample_raw")
            self.assertEqual(video_row["has_pp"], False)
            self.assertTrue(Path(paths["zarr_path"]).exists())
            self.assertTrue(Path(paths["videos_parquet_path"]).exists())
            self.assertTrue(Path(paths["frames_parquet_path"]).exists())
            self.assertTrue(Path(paths["runs_parquet_path"]).exists())

            root = zarr.open_group(str(out_dir / "landmarks.zarr"), mode="r")
            sample = root["samples"]["sample_raw"]
            self.assertIn("raw", sample)
            self.assertNotIn("pp", sample)
            self.assertEqual(sample.attrs["schema_version"], 1)
            self.assertEqual(sample["raw"]["left_xyz"].shape, (2, 21, 3))

            videos = pq.read_table(out_dir / "videos.parquet").to_pylist()
            frames = pq.read_table(out_dir / "frames.parquet").to_pylist()
            runs = pq.read_table(out_dir / "runs.parquet").to_pylist()
            self.assertEqual(len(videos), 1)
            self.assertEqual(len(frames), 2)
            self.assertEqual(len(runs), 1)
            self.assertEqual(videos[0]["sample_id"], "sample_raw")
            self.assertEqual(frames[0]["variant_pp_group"], "")
            self.assertEqual(runs[0]["sample_count_processed"], 1)

    def test_writer_creates_postprocessed_group_and_skip_existing_detects_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            writer = ExtractorOutputWriter(
                out_dir=out_dir,
                run_id="run_pp",
                args_snapshot={"jobs": 1},
                versions={"python": "test"},
            )
            staged = write_staged_payload(writer.current_run_dir, "sample_pp", _sample_payload("sample_pp", with_pp=True))
            writer.commit_staged_sample(staged)
            writer.finalize(
                scheduled_count=1,
                skipped_count=0,
                failed_count=0,
                processed_count=1,
                segments_mode=True,
                jobs=1,
                seed=0,
                mp_backend="tasks",
                aggregate={"quality_score": 0.75, "pp_filled_frac": 0.25},
            )

            root = zarr.open_group(str(out_dir / "landmarks.zarr"), mode="r")
            sample = root["samples"]["sample_pp"]
            self.assertIn("pp", sample)
            self.assertEqual(float(sample["pp"]["left_xyz"][1, 0, 0]), 42.0)

            writer2 = ExtractorOutputWriter(
                out_dir=out_dir,
                run_id="run_pp_2",
                args_snapshot={"jobs": 1},
                versions={"python": "test"},
            )
            self.assertTrue(writer2.is_sample_committed("sample_pp"))

    def test_finalize_writes_empty_parquet_tables_when_no_samples_committed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            writer = ExtractorOutputWriter(
                out_dir=out_dir,
                run_id="run_empty",
                args_snapshot={"jobs": 1},
                versions={"python": "test"},
            )
            paths = writer.finalize(
                scheduled_count=0,
                skipped_count=0,
                failed_count=1,
                processed_count=0,
                segments_mode=False,
                jobs=1,
                seed=0,
                mp_backend="tasks",
                aggregate={},
            )

            videos = pq.read_table(paths["videos_parquet_path"]).to_pylist()
            frames = pq.read_table(paths["frames_parquet_path"]).to_pylist()
            runs = pq.read_table(paths["runs_parquet_path"]).to_pylist()
            self.assertEqual(videos, [])
            self.assertEqual(frames, [])
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["sample_count_failed"], 1)

    def test_latest_run_row_reflects_merged_artifact_after_repair_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)

            writer1 = ExtractorOutputWriter(
                out_dir=out_dir,
                run_id="run_1",
                args_snapshot={"jobs": 1},
                versions={"python": "test"},
            )
            staged1 = write_staged_payload(writer1.current_run_dir, "sample_a", _sample_payload("sample_a", with_pp=False))
            writer1.commit_staged_sample(staged1)
            writer1.finalize(
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

            writer2 = ExtractorOutputWriter(
                out_dir=out_dir,
                run_id="run_2",
                args_snapshot={"jobs": 1},
                versions={"python": "test"},
            )
            staged2 = write_staged_payload(writer2.current_run_dir, "sample_b", _sample_payload("sample_b", with_pp=True))
            writer2.commit_staged_sample(staged2)
            writer2.finalize(
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

            runs = pq.read_table(out_dir / "runs.parquet").to_pylist()
            self.assertEqual(len(runs), 2)
            self.assertEqual(runs[-1]["sample_count_existing"], 1)
            self.assertEqual(runs[-1]["sample_count_scheduled"], 2)
            self.assertEqual(runs[-1]["sample_count_processed"], 2)
            self.assertIsNotNone(runs[-1]["quality_score"])


if __name__ == "__main__":
    unittest.main()
