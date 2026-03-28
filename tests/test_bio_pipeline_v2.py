from __future__ import annotations

import argparse
import csv
import io
import json
import tempfile
import unittest
import warnings
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from bio.pipeline import build_dataset as build_dataset_mod
from bio.pipeline.build_dataset import _promote_many_transactional, build_overlap_report
from bio.pipeline.prelabel import PrelabelConfig, VideoStore, build_mirror_idx, parse_csv, process_sample
from bio.pipeline import signer_split as signer_split_mod
from bio.pipeline import synth_build
from bio.pipeline import train as train_mod
from bio.pipeline import train_curriculum as train_curriculum_mod
from bio.pipeline.synth_build import evaluate_realism_gate
from bio.core.datasets.shard_dataset import ShardedBiosDataset, make_shard_aware_boundary_batch_sampler
from bio.pipeline.train import (
    _batch_prepare_fn,
    _checkpoint_schedule_state,
    _finalize_threshold_sweep_state,
    _loader_profile_event,
    _init_threshold_sweep_state,
    _is_better_boundary,
    _lr_for_epoch,
    _make_schedule_state,
    _make_runtime_summary,
    _raise_if_bad_synth_realism,
    _restore_history_from_checkpoint,
    _should_write_prediction_artifacts,
    _resolve_auto_workers,
    _set_optimizer_lr,
    _startup_prefix_mask_torch,
    _update_threshold_sweep_state,
    _threshold_sweep,
    _update_early_stop_counter,
    binary_calibration_metrics,
    eval_epoch,
    JsonlLogger,
    LoaderProfile,
    parse_bio_segments_strict,
    train_one_epoch,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["attachment_id", "text", "begin", "end", "split"])
        writer.writeheader()
        writer.writerows(rows)


def _hand_points(base: float) -> list[dict[str, float]]:
    return [
        {"x": float(base + 0.01 * idx), "y": float(base + 0.02 * idx), "z": float(-0.005 * idx)}
        for idx in range(21)
    ]


def _write_tiny_synth_dir(root: Path, *, bios: list[np.ndarray]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    shards_dir = root / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    pts = np.stack([np.zeros((seq.shape[0], 42, 3), dtype=np.float32) for seq in bios], axis=0)
    mask = np.stack([np.ones((seq.shape[0], 42, 1), dtype=np.float32) for seq in bios], axis=0)
    bio = np.stack([seq.astype(np.uint8, copy=False) for seq in bios], axis=0)
    has_b = (bio == 1).any(axis=1).astype(np.uint8)
    meta = np.asarray([json.dumps({"sample_index": int(i)}) for i in range(len(bios))], dtype="<U64")
    shard_path = shards_dir / "shard_000000.npz"
    np.savez_compressed(shard_path, pts=pts, mask=mask, bio=bio, has_b=has_b, meta=meta)
    (root / "index.json").write_text(
        json.dumps(
            [
                {
                    "path_to_npz": "shards/shard_000000.npz",
                    "seq_len": int(bio.shape[1]),
                    "V": 42,
                    "num_samples": int(bio.shape[0]),
                }
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    stats = {
        "dataset_signature": {
            "kind": "test_fixture",
            "samples": int(bio.shape[0]),
            "seq_len": int(bio.shape[1]),
        }
    }
    (root / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return root


def _write_multi_shard_synth_dir(root: Path, *, shard_bios: list[list[np.ndarray]]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    shards_dir = root / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    total_samples = 0
    seq_len = 0
    for shard_idx, bios in enumerate(shard_bios):
        pts = np.stack([np.zeros((seq.shape[0], 42, 3), dtype=np.float32) for seq in bios], axis=0)
        mask = np.stack([np.ones((seq.shape[0], 42, 1), dtype=np.float32) for seq in bios], axis=0)
        bio = np.stack([seq.astype(np.uint8, copy=False) for seq in bios], axis=0)
        has_b = (bio == 1).any(axis=1).astype(np.uint8)
        meta = np.asarray([json.dumps({"sample_index": int(i), "shard_index": int(shard_idx)}) for i in range(len(bios))], dtype="<U96")
        shard_path = shards_dir / f"shard_{shard_idx:06d}.npz"
        np.savez_compressed(shard_path, pts=pts, mask=mask, bio=bio, has_b=has_b, meta=meta)
        rows.append(
            {
                "path_to_npz": f"shards/{shard_path.name}",
                "seq_len": int(bio.shape[1]),
                "V": 42,
                "num_samples": int(bio.shape[0]),
            }
        )
        total_samples += int(bio.shape[0])
        seq_len = int(bio.shape[1])
    (root / "index.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "stats.json").write_text(
        json.dumps({"dataset_signature": {"kind": "multi_shard_fixture", "samples": total_samples, "seq_len": seq_len}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return root


class BioPipelineV2Tests(unittest.TestCase):
    def test_jsonl_logger_text_console_keeps_file_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "train_log.jsonl"
            buf = io.StringIO()
            with redirect_stdout(buf):
                logger = JsonlLogger(log_path, console_format="text")
                logger.log({"event": "start", "device": "cpu", "amp_dtype": "float32", "train_samples": 10, "val_samples": 2, "batch_size": 4, "num_params": 5, "train_loader_profile": {"workers": 0}, "val_loader_profile": {}, "prediction_artifacts_every": 5})
            stdout = buf.getvalue().strip()
            self.assertTrue(stdout.startswith("start device=cpu"))
            line = log_path.read_text(encoding="utf-8").strip().splitlines()[0]
            self.assertTrue(line.startswith("{"))

    def test_jsonl_logger_json_console_preserves_json_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "train_log.jsonl"
            buf = io.StringIO()
            with redirect_stdout(buf):
                logger = JsonlLogger(log_path, console_format="json")
                logger.log({"event": "done", "step": 7})
            stdout = buf.getvalue().strip()
            self.assertTrue(stdout.startswith("{"))
            self.assertIn('"event": "done"', stdout)

    def test_batch_prepare_fn_skips_device_move_for_prefetched_batches(self) -> None:
        class Prefetched:
            moves_to_device = True

        batch = {"pts": torch.zeros((1, 2, 42, 3), dtype=torch.float32)}
        with mock.patch("bio.pipeline.train._move_batch_to_device", side_effect=AssertionError("unexpected move")):
            prepare = _batch_prepare_fn(Prefetched(), torch.device("cpu"))
            self.assertIs(prepare(batch), batch)

        with mock.patch("bio.pipeline.train._move_batch_to_device", wraps=train_mod._move_batch_to_device) as mocked_move:
            prepare = _batch_prepare_fn(object(), torch.device("cpu"))
            moved = prepare(batch)
            self.assertIsInstance(moved["pts"], torch.Tensor)
            self.assertEqual(mocked_move.call_count, 1)

    def test_sharded_dataset_parse_meta_toggle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_tiny_synth_dir(Path(tmp) / "synth", bios=[np.asarray([0, 1, 2, 0], dtype=np.uint8)])
            train_ds = ShardedBiosDataset(root, shard_cache_items=1, parse_meta=False)
            val_ds = ShardedBiosDataset(root, shard_cache_items=1, parse_meta=True)
            self.assertNotIn("meta", train_ds[0])
            self.assertIn("meta", val_ds[0])
            self.assertEqual(val_ds[0]["meta"]["sample_index"], 0)

    def test_auto_workers_prefers_lower_workers_within_margin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_tiny_synth_dir(
                Path(tmp) / "train",
                bios=[np.asarray([0, 1, 2, 0], dtype=np.uint8), np.asarray([0, 1, 2, 2], dtype=np.uint8)],
            )
            dataset = ShardedBiosDataset(root, shard_cache_items=1, parse_meta=False)
            args = argparse.Namespace(
                num_workers=8,
                batch_size=2,
                prefetch=2,
                channels_last=False,
                use_prefetch_loader=False,
                auto_workers=True,
                auto_workers_rebench=True,
                auto_workers_max=4,
                auto_workers_warmup_batches=1,
                auto_workers_measure_batches=2,
            )

            def fake_measure(*_args, **kwargs):
                workers = int(kwargs["workers"])
                lookup = {0: 96.0, 1: 100.0, 2: 98.0, 4: 80.0}
                return {
                    "workers": workers,
                    "measured_batches": 2,
                    "samples_per_sec": lookup[workers],
                    "success": True,
                    "benchmark_kind": "forward_only_runtime",
                }

            with mock.patch("bio.pipeline.train._measure_loader_runtime_sps", side_effect=fake_measure):
                profile, info = _resolve_auto_workers(
                    args=args,
                    dataset=dataset,
                    dataset_dir=root,
                    device=torch.device("cpu"),
                    model=torch.nn.Linear(4, 4),
                    use_amp=False,
                    amp_dtype=torch.float32,
                    requested_workers=8,
                    dataset_role="train",
                    model_signature={"model": "dummy", "num_params": 20},
                )
            self.assertEqual(profile.workers, 0)

    def test_evaluate_realism_gate_prefers_semantic_metrics(self) -> None:
        generated = {
            "dataset_profile": "main_continuous",
            "semantic_seam_realism": {
                "targets": {
                    "boundary_internal_center_jump_ratio_max": 2.0,
                    "boundary_internal_scale_jump_ratio_max": 2.5,
                    "cross_source_boundary_frac_max": 0.01,
                },
                "boundary_internal_center_jump_ratio": 1.5,
                "boundary_internal_scale_jump_ratio": 2.0,
                "cross_source_boundary_frac": 0.0,
            },
            "expanded_seam_realism": {
                "targets": {
                    "boundary_internal_center_jump_ratio_max": 2.0,
                    "boundary_internal_scale_jump_ratio_max": 2.5,
                    "cross_source_boundary_frac_max": 0.01,
                },
                "boundary_internal_center_jump_ratio": 9.0,
                "boundary_internal_scale_jump_ratio": 9.0,
                "cross_source_boundary_frac": 0.9,
            },
            "first_B_frame_eq0_frac_over_total": 0.0,
            "samples_with_leading_o_prefix_frac": 0.9,
            "all_o_samples_frac": 0.15,
        }
        acceptance = evaluate_realism_gate(generated)
        self.assertEqual(acceptance["profile"], "main_continuous")
        self.assertTrue(bool(acceptance["passed"]))

    def test_evaluate_realism_gate_warmup_profile_only_blocks_cross_source(self) -> None:
        generated = {
            "dataset_profile": "warmup_single_sign",
            "semantic_seam_realism": {
                "targets": {
                    "boundary_internal_center_jump_ratio_max": 2.2,
                    "boundary_internal_scale_jump_ratio_max": 2.5,
                    "cross_source_boundary_frac_max": 0.0,
                },
                "boundary_internal_center_jump_ratio": 9.0,
                "boundary_internal_scale_jump_ratio": 9.0,
                "cross_source_boundary_frac": 0.0,
            },
            "first_B_frame_eq0_frac_over_total": 0.0,
            "samples_with_leading_o_prefix_frac": 0.3,
            "all_o_samples_frac": 0.1,
        }
        acceptance = evaluate_realism_gate(generated)
        self.assertEqual(acceptance["profile"], "warmup_single_sign")
        self.assertTrue(bool(acceptance["passed"]))

    def test_process_sample_trimmed_mode_keeps_full_clip_and_uses_csv_gold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = []
            for idx in range(8):
                row = {"ts": float(idx * 33.3)}
                if 2 <= idx <= 5:
                    row["hand 1"] = _hand_points(0.1 + 0.01 * idx)
                    row["hand 1_score"] = 0.99
                frames.append(row)
            (root / "clip_a.json").write_text(json.dumps({"frames": frames, "meta": {"coords": "image"}}, ensure_ascii=False), encoding="utf-8")
            out_dir = root / "out"
            out_dir.mkdir(parents=True, exist_ok=True)
            cfg = PrelabelConfig(trimmed_mode=True, include_pose=False, preprocessing_version="canonical_hands42_v3")
            store = VideoStore(root, cache_size=0, prefer_pp=True)
            result = process_sample(
                {
                    "vid": "clip_a",
                    "sample_id": "clip_a__2_6",
                    "label_str": "hello",
                    "begin": 2,
                    "end": 6,
                    "split": "train",
                    "dataset": "slovo",
                    "source_group": "clip_a",
                    "out_path": str(out_dir / "sample.npz"),
                    "debug": False,
                },
                cfg,
                store,
                build_mirror_idx(cfg.include_pose, cfg.pose_keep),
                out_dir,
            )
            self.assertEqual(int(result["index"]["T_total"]), 8)
            self.assertEqual(int(result["index"]["start_idx"]), 2)
            self.assertEqual(int(result["index"]["end_idx"]), 5)
            with np.load(out_dir / "sample.npz", allow_pickle=False) as z:
                bio = z["bio"].tolist()
                meta = json.loads(str(z["meta"].item()))
            self.assertEqual(bio, [0, 0, 1, 2, 2, 2, 0, 0])
            self.assertTrue(bool(meta.get("trimmed_mode", False)))
            self.assertEqual(str(meta.get("label_source", "")), "trimmed_csv")

    def test_build_overlap_report_is_slovo_only_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            slovo_csv = root / "annotations.csv"
            slovo_noev_csv = root / "annotations_no_event.csv"
            _write_csv(
                slovo_csv,
                [
                    {"attachment_id": "video_a.mp4", "text": "A", "begin": 0, "end": 10, "split": "train"},
                    {"attachment_id": "video_a.mp4", "text": "A", "begin": 10, "end": 20, "split": "val"},
                ],
            )
            _write_csv(
                slovo_noev_csv,
                [
                    {"attachment_id": "video_b.mp4", "text": "no_event", "begin": 0, "end": 10, "split": "train"},
                    {"attachment_id": "video_b.mp4", "text": "no_event", "begin": 10, "end": 20, "split": "val"},
                ],
            )
            report = build_overlap_report(slovo_csv, slovo_noev_csv)
            self.assertFalse(report["ok"])
            self.assertEqual(report["slovo_sign"]["overlap_count"], 1)
            self.assertEqual(report["slovo_no_event"]["overlap_count"], 1)
            self.assertNotIn("ipn_hand", report)

    def test_build_overlap_report_uses_user_id_not_clip_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            slovo_csv = root / "annotations.tsv"
            slovo_noev_csv = root / "annotations_no_event.tsv"
            slovo_csv.write_text(
                "attachment_id\ttext\tuser_id\tbegin\tend\tsplit\n"
                "video_a.mp4\tHELLO\tuser_1\t0\t10\ttrain\n"
                "video_b.mp4\tHELLO\tuser_1\t0\t10\tval\n",
                encoding="utf-8",
            )
            slovo_noev_csv.write_text(
                "attachment_id\ttext\tuser_id\tbegin\tend\tsplit\n"
                "video_c.mp4\tno_event\tuser_2\t0\t10\ttrain\n"
                "video_d.mp4\tno_event\tuser_3\t0\t10\tval\n",
                encoding="utf-8",
            )
            report = build_overlap_report(slovo_csv, slovo_noev_csv)
            self.assertFalse(report["ok"])
            self.assertEqual(report["slovo_sign"]["overlap_count"], 1)

    def test_train_curriculum_runs_warmup_then_finetune(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            warmup_dir = root / "warmup_ds"
            main_dir = root / "main_ds"
            for path in (warmup_dir, main_dir):
                path.mkdir(parents=True, exist_ok=True)
            calls: list[list[str]] = []

            def fake_train_main(argv: list[str]) -> None:
                calls.append(list(argv))
                out_dir = Path(argv[argv.index("--out_dir") + 1])
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "best_recall_safe.pt").write_text("stub", encoding="utf-8")

            with mock.patch("bio.pipeline.train_curriculum.train.main", side_effect=fake_train_main):
                train_curriculum_mod.main(
                    [
                        "--train_warmup_dir", str(warmup_dir),
                        "--val_warmup_dir", str(warmup_dir),
                        "--train_dir", str(main_dir),
                        "--val_dir", str(main_dir),
                        "--out_dir", str(root / "curriculum"),
                        "--config", "bio/configs/bio_default.json",
                    ]
                )

            self.assertEqual(len(calls), 2)
            self.assertNotIn("--resume", calls[0])
            self.assertIn("--resume", calls[1])
            self.assertIn("--resume_model_only", calls[1])

    def test_train_rejects_bad_synth_realism_stats(self) -> None:
        stats = {
            "generated": {
                "acceptance": {
                    "passed": False,
                    "failures": [
                        "cross_source_boundary_frac=0.6600 > 0.1000",
                    ],
                }
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                _raise_if_bad_synth_realism(Path(tmp), stats, allow_bad_synth_stats=False)
            _raise_if_bad_synth_realism(Path(tmp), stats, allow_bad_synth_stats=True)

    def test_shard_aware_batch_sampler_keeps_batch_locality(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_multi_shard_synth_dir(
                Path(tmp) / "train",
                shard_bios=[
                    [
                        np.asarray([0, 1, 2, 0], dtype=np.uint8),
                        np.asarray([0, 0, 0, 0], dtype=np.uint8),
                        np.asarray([0, 1, 2, 2], dtype=np.uint8),
                        np.asarray([0, 0, 0, 0], dtype=np.uint8),
                    ],
                    [
                        np.asarray([0, 1, 2, 0], dtype=np.uint8),
                        np.asarray([0, 0, 0, 0], dtype=np.uint8),
                        np.asarray([0, 1, 2, 2], dtype=np.uint8),
                        np.asarray([0, 0, 0, 0], dtype=np.uint8),
                    ],
                ],
            )
            dataset = ShardedBiosDataset(root, shard_cache_items=1, parse_meta=False)
            sampler = make_shard_aware_boundary_batch_sampler(dataset, batch_size=4, p_with_b=0.5, seed=123)
            batch = next(iter(sampler))
            shard_ids = {dataset._locate(idx)[0] for idx in batch}
            self.assertLessEqual(len(shard_ids), 2)
            has_b = [bool(dataset.has_b[idx]) for idx in batch]
            self.assertGreaterEqual(sum(has_b), 1)

    def test_dataset_scan_cache_is_written_and_reused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_tiny_synth_dir(Path(tmp) / "synth", bios=[np.asarray([0, 1, 2, 0], dtype=np.uint8)])
            ds1 = ShardedBiosDataset(root, shard_cache_items=1, parse_meta=False)
            cache_path = root / ".dataset_scan_cache_v1.npz"
            self.assertTrue(cache_path.exists())
            with mock.patch.object(ShardedBiosDataset, "_save_scan_cache", wraps=ds1._save_scan_cache) as save_mock:
                ds2 = ShardedBiosDataset(root, shard_cache_items=1, parse_meta=False)
            self.assertEqual(int(ds2.has_b.sum()), int(ds1.has_b.sum()))
            self.assertEqual(save_mock.call_count, 0)

    def test_slovo_split_test_is_treated_as_val(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "annotations.csv"
            _write_csv(
                csv_path,
                [
                    {"attachment_id": "video_a.mp4", "text": "A", "begin": 0, "end": 10, "split": "test"},
                    {"attachment_id": "video_b.mp4", "text": "B", "begin": 0, "end": 10, "split": "train"},
                ],
            )
            parsed = parse_csv(csv_path, "val")
            self.assertEqual(len(parsed.rows), 1)
            self.assertEqual(parsed.rows[0].vid, "video_a")

    def test_parse_csv_uses_user_id_as_source_group_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "annotations.tsv"
            csv_path.write_text(
                "attachment_id\ttext\tuser_id\tbegin\tend\tsplit\n"
                "video_a.mp4\tHELLO\tuser_42\t0\t10\ttrain\n",
                encoding="utf-8",
            )
            parsed = parse_csv(csv_path, "train")
            self.assertEqual(len(parsed.rows), 1)
            self.assertEqual(parsed.rows[0].vid, "video_a")
            self.assertEqual(parsed.rows[0].signer_id, "user_42")
            self.assertEqual(parsed.rows[0].source_group, "user_42")

    def test_signer_split_rewrites_csvs_to_signer_disjoint_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sign_csv = root / "annotations.tsv"
            noev_csv = root / "annotations_no_event.tsv"
            sign_csv.write_text(
                (
                    "attachment_id\ttext\tuser_id\tbegin\tend\tsplit\n"
                    "clip_a.mp4\tA\tuser_1\t0\t10\ttrain\n"
                    "clip_b.mp4\tB\tuser_1\t0\t10\tval\n"
                    "clip_c.mp4\tC\tuser_2\t0\t10\ttrain\n"
                    "clip_d.mp4\tD\tuser_3\t0\t10\tval\n"
                ),
                encoding="utf-8",
            )
            noev_csv.write_text(
                (
                    "attachment_id\ttext\tuser_id\tbegin\tend\tsplit\n"
                    "clip_e.mp4\tno_event\tuser_1\t0\t10\tval\n"
                    "clip_f.mp4\tno_event\tuser_2\t0\t10\ttrain\n"
                    "clip_g.mp4\tno_event\tuser_3\t0\t10\tval\n"
                ),
                encoding="utf-8",
            )
            out_dir = root / "signer_split"
            signer_split_mod.main(
                [
                    "--csv",
                    str(sign_csv),
                    "--csv",
                    str(noev_csv),
                    "--out_dir",
                    str(out_dir),
                ]
            )
            report = build_overlap_report(out_dir / sign_csv.name, out_dir / noev_csv.name)
            self.assertTrue(report["ok"])
            summary = json.loads((out_dir / "signer_split_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(int(summary["after_overlap_signer_count"]), 0)
            mapping = json.loads((out_dir / "signer_assignments.json").read_text(encoding="utf-8"))
            self.assertIn(mapping["user_1"], {"train", "val"})

    def test_signer_split_preserves_nontrivial_train_share(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sign_csv = root / "annotations.tsv"
            noev_csv = root / "annotations_no_event.tsv"
            sign_csv.write_text(
                (
                    "attachment_id\ttext\tuser_id\tbegin\tend\ttrain\n"
                    "a1.mp4\tA\tuser_1\t0\t10\tTrue\n"
                    "a2.mp4\tA\tuser_1\t0\t10\tTrue\n"
                    "a3.mp4\tA\tuser_1\t0\t10\tFalse\n"
                    "b1.mp4\tB\tuser_2\t0\t10\tTrue\n"
                    "b2.mp4\tB\tuser_2\t0\t10\tTrue\n"
                    "c1.mp4\tC\tuser_3\t0\t10\tTrue\n"
                    "d1.mp4\tD\tuser_4\t0\t10\tFalse\n"
                    "d2.mp4\tD\tuser_4\t0\t10\tFalse\n"
                ),
                encoding="utf-8",
            )
            noev_csv.write_text(
                (
                    "attachment_id\ttext\tuser_id\tbegin\tend\ttrain\n"
                    "n1.mp4\tno_event\tuser_1\t0\t10\tTrue\n"
                    "n2.mp4\tno_event\tuser_2\t0\t10\tTrue\n"
                    "n3.mp4\tno_event\tuser_3\t0\t10\tTrue\n"
                    "n4.mp4\tno_event\tuser_4\t0\t10\tFalse\n"
                ),
                encoding="utf-8",
            )
            out_dir = root / "signer_split"
            signer_split_mod.main(
                [
                    "--csv",
                    str(sign_csv),
                    "--csv",
                    str(noev_csv),
                    "--out_dir",
                    str(out_dir),
                ]
            )
            with (out_dir / "annotations.tsv").open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f, delimiter="\t"))
            split_counts = Counter((row.get("split") or "").strip().lower() for row in rows)
            self.assertGreater(int(split_counts.get("train", 0)), 0)
            self.assertGreater(int(split_counts.get("val", 0)), 0)
            self.assertGreater(int(split_counts.get("train", 0)), int(split_counts.get("val", 0)))

    def test_signer_split_keeps_pure_signers_on_their_original_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sign_csv = root / "annotations.tsv"
            noev_csv = root / "annotations_no_event.tsv"
            sign_csv.write_text(
                (
                    "attachment_id\ttext\tuser_id\tbegin\tend\tsplit\n"
                    "a1.mp4\tA\tuser_train_only\t0\t10\ttrain\n"
                    "a2.mp4\tA\tuser_overlap\t0\t10\ttrain\n"
                    "a3.mp4\tA\tuser_overlap\t0\t10\tval\n"
                    "v1.mp4\tV\tuser_val_only\t0\t10\tval\n"
                ),
                encoding="utf-8",
            )
            noev_csv.write_text(
                (
                    "attachment_id\ttext\tuser_id\tbegin\tend\tsplit\n"
                    "n1.mp4\tno_event\tuser_train_only\t0\t10\ttrain\n"
                    "n2.mp4\tno_event\tuser_val_only\t0\t10\tval\n"
                ),
                encoding="utf-8",
            )
            out_dir = root / "signer_split"
            signer_split_mod.main(
                [
                    "--csv",
                    str(sign_csv),
                    "--csv",
                    str(noev_csv),
                    "--out_dir",
                    str(out_dir),
                ]
            )
            with (out_dir / "annotations.tsv").open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f, delimiter="\t"))
            split_by_user = {}
            for row in rows:
                split_by_user.setdefault(row["user_id"], set()).add((row.get("split") or "").strip().lower())
            self.assertEqual(split_by_user["user_train_only"], {"train"})
            self.assertEqual(split_by_user["user_val_only"], {"val"})

    def test_build_overlap_report_fails_on_shared_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            slovo_csv = root / "annotations.csv"
            slovo_noev_csv = root / "annotations_no_event.csv"
            manifest = root / "ipn_manifest.jsonl"

            _write_csv(
                slovo_csv,
                [
                    {"attachment_id": "video_a.mp4", "text": "A", "begin": 0, "end": 10, "split": "train"},
                    {"attachment_id": "video_a.mp4", "text": "A", "begin": 10, "end": 20, "split": "val"},
                ],
            )
            _write_csv(
                slovo_noev_csv,
                [
                    {"attachment_id": "video_b.mp4", "text": "no_event", "begin": 0, "end": 10, "split": "train"},
                    {"attachment_id": "video_b.mp4", "text": "no_event", "begin": 10, "end": 20, "split": "val"},
                ],
            )
            manifest.write_text(
                "\n".join(
                    [
                        json.dumps({"video_id": "ipn_video_1", "split": "train"}),
                        json.dumps({"video_id": "ipn_video_1", "split": "val"}),
                    ]
                ),
                encoding="utf-8",
            )

            report = build_overlap_report(slovo_csv, slovo_noev_csv, manifest)
            self.assertFalse(report["ok"])
            self.assertEqual(report["slovo_sign"]["overlap_count"], 1)
            self.assertEqual(report["slovo_no_event"]["overlap_count"], 1)
            self.assertEqual(report["ipn_hand"]["overlap_count"], 1)

    def test_threshold_sweep_emits_ranked_metrics(self) -> None:
        y = np.asarray([[0, 1, 2, 2, 0, 0]], dtype=np.uint8)
        probs = np.asarray(
            [
                [
                    [0.90, 0.05, 0.05],
                    [0.05, 0.90, 0.05],
                    [0.10, 0.20, 0.70],
                    [0.10, 0.15, 0.75],
                    [0.80, 0.10, 0.10],
                    [0.85, 0.05, 0.10],
                ]
            ],
            dtype=np.float32,
        )
        mask = np.ones((1, 6, 42, 1), dtype=np.float32)
        sweep = _threshold_sweep(y, probs, [0.3, 0.5, 0.7], mask_seq=mask)
        self.assertEqual(len(sweep["thresholds"]), 3)
        self.assertIn("b_f1_tol", sweep["best_b_f1_tol"])
        self.assertIn("balanced_guardrails_passed", sweep["thresholds"][0])
        self.assertIn("bio_violation_rate_pred", sweep["thresholds"][0])
        self.assertIn("bio_violation_abs_err", sweep["thresholds"][0])
        self.assertIn("startup_false_start_rate", sweep["thresholds"][0])

    def test_train_one_epoch_logs_first_batch_even_with_large_log_every(self) -> None:
        class TinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, pts: torch.Tensor, mask: torch.Tensor):
                logits = torch.zeros((pts.shape[0], pts.shape[1], 3), dtype=pts.dtype, device=pts.device)
                logits[..., 0] = self.scale
                return logits, None

        batch = {
            "pts": torch.zeros((1, 4, 42, 3), dtype=torch.float32),
            "mask": torch.ones((1, 4, 42, 1), dtype=torch.float32),
            "bio": torch.tensor([[0, 1, 2, 0]], dtype=torch.long),
        }
        logger = mock.Mock()
        model = TinyModel()
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler(enabled=False)
        _step, metrics = train_one_epoch(
            model=model,
            loader=[batch],
            optim=optim,
            scaler=scaler,
            device=torch.device("cpu"),
            class_weights=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
            grad_clip=0.0,
            log_every=100,
            step0=0,
            use_amp=False,
            amp_dtype=torch.float32,
            epoch=1,
            logger=logger,
            tb_logger=None,
            tb_log_every=1,
            label_smoothing=0.0,
            ema=None,
            channels_last=False,
        )
        train_step_calls = [call.args[0] for call in logger.log.call_args_list if call.args and call.args[0].get("event") == "train_step"]
        self.assertEqual(len(train_step_calls), 1)
        self.assertEqual(train_step_calls[0]["step_in_epoch"], 1)
        self.assertEqual(train_step_calls[0]["steps_in_epoch"], 1)
        self.assertGreater(metrics["avg_batch_time_ms"], 0.0)

    def test_make_runtime_summary_exposes_startup_and_meta_flags(self) -> None:
        summary = _make_runtime_summary(
            args=argparse.Namespace(tf32=False, channels_last=False),
            device=torch.device("cpu"),
            use_amp=False,
            amp_dtype=torch.float32,
            compile_info={"enabled": False},
            train_profile=LoaderProfile(workers=0, persistent_workers=False, prefetch_factor=None, pin_memory=False, use_prefetch_loader=False),
            train_loader_info={"candidates": [{"workers": 0, "samples_per_sec": 10.0}], "selection_reason": "picked"},
            train_loader=[],
            val_profile=None,
            val_loader_info={},
            val_loader=None,
            train_dataset_signature={"kind": "fixture"},
            val_dataset_signature={},
            score_weights={"lambda_len": 0.1},
            schedule_state={"type": "warmup_cosine_epoch"},
            startup_phase_times={"dataset_scan": 1.2},
            startup_total_sec=1.5,
            meta_parsing_train=False,
            meta_parsing_val=True,
            train_shard_cache_items=8,
            val_shard_cache_items=2,
        )
        self.assertEqual(summary["startup_phase_times"]["dataset_scan"], 1.2)
        self.assertAlmostEqual(summary["startup_total_sec"], 1.5)
        self.assertFalse(summary["meta_parsing_train"])
        self.assertTrue(summary["meta_parsing_val"])
        self.assertEqual(summary["train_batching_mode"], "shard_aware_boundary_batch_sampler")

    def test_threshold_sweep_streaming_matches_buffered(self) -> None:
        y = np.asarray(
            [
                [0, 1, 2, 0, 0, 1],
                [0, 0, 1, 2, 2, 0],
            ],
            dtype=np.uint8,
        )
        probs = np.asarray(
            [
                [
                    [0.80, 0.10, 0.10],
                    [0.05, 0.90, 0.05],
                    [0.05, 0.10, 0.85],
                    [0.70, 0.20, 0.10],
                    [0.75, 0.10, 0.15],
                    [0.10, 0.75, 0.15],
                ],
                [
                    [0.85, 0.10, 0.05],
                    [0.70, 0.20, 0.10],
                    [0.05, 0.85, 0.10],
                    [0.05, 0.10, 0.85],
                    [0.05, 0.10, 0.85],
                    [0.80, 0.10, 0.10],
                ],
            ],
            dtype=np.float32,
        )
        thresholds = [0.3, 0.5, 0.7]
        mask = np.ones((2, 6, 42, 1), dtype=np.float32)
        buffered = _threshold_sweep(y, probs, thresholds, mask_seq=mask)
        state = _init_threshold_sweep_state(thresholds)
        _update_threshold_sweep_state(state, y[:1], probs[:1], mask[:1], None)
        _update_threshold_sweep_state(state, y[1:], probs[1:], mask[1:], None)
        streamed = _finalize_threshold_sweep_state(state)
        self.assertEqual(buffered["best_b_f1_tol"], streamed["best_b_f1_tol"])
        self.assertEqual(buffered["best_balanced"], streamed["best_balanced"])
        self.assertEqual(buffered["thresholds"], streamed["thresholds"])

    def test_startup_prefix_mask_finds_pre_hand_frames(self) -> None:
        mask = torch.zeros((1, 5, 42, 1), dtype=torch.float32)
        mask[0, 2:, 21:, 0] = 1.0
        startup_mask, first_visible, left, right, total = _startup_prefix_mask_torch(
            mask,
            min_valid_joints=8,
            min_visible_frames=1,
        )
        self.assertEqual(int(first_visible[0].item()), 2)
        self.assertEqual(startup_mask[0].tolist(), [True, True, False, False, False])
        self.assertEqual(int(left[0, 2].item()), 0)
        self.assertEqual(int(right[0, 2].item()), 21)
        self.assertEqual(int(total[0, 2].item()), 21)

    def test_boundary_comparator_uses_tie_break_chain(self) -> None:
        best = {
            "b_f1_tol": 0.8,
            "b_err_mean": 1.2,
            "pred_B_ratio": 1.1,
            "transition_rate_abs_err": 0.2,
            "bio_violation_abs_err": 0.1,
            "selection_threshold": 0.5,
        }
        better_berr = dict(best, b_err_mean=1.0)
        self.assertTrue(_is_better_boundary(better_berr, best))
        ratio_tie = dict(best, b_err_mean=1.2, pred_B_ratio=1.02)
        self.assertTrue(_is_better_boundary(ratio_tie, best))
        trans_tie = dict(best, b_err_mean=1.2, pred_B_ratio=1.1, transition_rate_abs_err=0.05)
        self.assertTrue(_is_better_boundary(trans_tie, best))
        vio_tie = dict(best, b_err_mean=1.2, pred_B_ratio=1.1, transition_rate_abs_err=0.2, bio_violation_abs_err=0.05)
        self.assertTrue(_is_better_boundary(vio_tie, best))

    def test_binary_calibration_metrics_uses_event_frequency(self) -> None:
        prob = np.asarray([0.1, 0.1, 0.1, 0.1], dtype=np.float32)
        target = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        metrics = binary_calibration_metrics(prob, target)
        self.assertAlmostEqual(metrics["ece"], 0.1, places=6)
        self.assertAlmostEqual(metrics["brier"], 0.01, places=6)

    def test_parse_bio_segments_strict_does_not_start_segments_from_orphan_i(self) -> None:
        stats = parse_bio_segments_strict(np.asarray([2, 2, 0, 1, 2, 0], dtype=np.uint8))
        self.assertEqual(stats["lengths"], [2])
        self.assertEqual(stats["violation_count"], 2)
        self.assertAlmostEqual(stats["violation_rate"], 2.0 / 6.0, places=6)

    def test_eval_epoch_threshold_sweep_does_not_emit_prediction_rows_when_disabled(self) -> None:
        class DummyModel(torch.nn.Module):
            def forward(self, pts: torch.Tensor, mask: torch.Tensor):
                logits = torch.zeros((pts.shape[0], pts.shape[1], 3), dtype=pts.dtype, device=pts.device)
                logits[..., 0] = 0.5
                logits[..., 1] = 0.25
                logits[..., 2] = 0.25
                return logits, None

        batch = {
            "pts": torch.zeros((2, 4, 42, 3), dtype=torch.float32),
            "mask": torch.ones((2, 4, 42, 1), dtype=torch.float32),
            "bio": torch.tensor([[0, 1, 2, 0], [0, 0, 1, 2]], dtype=torch.long),
            "meta": [{"id": "a"}, {"id": "b"}],
        }
        metrics, _cm, _examples, analysis = eval_epoch(
            DummyModel(),
            [batch],
            device=torch.device("cpu"),
            use_amp=False,
            amp_dtype=torch.float32,
            class_weights=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
            collect_predictions=False,
            threshold_sweep_points=[0.3, 0.5, 0.7],
            score_weights={},
        )
        self.assertIn("threshold_sweep", analysis)
        self.assertNotIn("prediction_rows", analysis)
        self.assertIn("bio_violation_rate_pred", metrics)

    def test_eval_epoch_prediction_rows_include_bio_violation_fields(self) -> None:
        class DummyModel(torch.nn.Module):
            def forward(self, pts: torch.Tensor, mask: torch.Tensor):
                logits = torch.zeros((pts.shape[0], pts.shape[1], 3), dtype=pts.dtype, device=pts.device)
                logits[..., 2] = 1.0
                return logits, None

        batch = {
            "pts": torch.zeros((1, 3, 42, 3), dtype=torch.float32),
            "mask": torch.ones((1, 3, 42, 1), dtype=torch.float32),
            "bio": torch.tensor([[0, 1, 2]], dtype=torch.long),
            "meta": [{"id": "sample"}],
        }
        _metrics, _cm, _examples, analysis = eval_epoch(
            DummyModel(),
            [batch],
            device=torch.device("cpu"),
            use_amp=False,
            amp_dtype=torch.float32,
            class_weights=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
            collect_predictions=True,
            threshold_sweep_points=[0.5],
            score_weights={},
        )
        row = analysis["prediction_rows"][0]
        self.assertIn("true_bio_violation_count", row)
        self.assertIn("pred_bio_violation_rate", row)
        self.assertIn("bio_violation_abs_err", row)

    def test_prediction_artifact_cadence_forces_best_and_final_epochs(self) -> None:
        self.assertFalse(
            _should_write_prediction_artifacts(
                enabled=False,
                epoch=1,
                every=5,
                total_epochs=10,
                boundary_improved=False,
                balanced_improved=False,
            )
        )
        self.assertFalse(
            _should_write_prediction_artifacts(
                enabled=True,
                epoch=1,
                every=5,
                total_epochs=10,
                boundary_improved=False,
                balanced_improved=False,
            )
        )
        self.assertTrue(
            _should_write_prediction_artifacts(
                enabled=True,
                epoch=5,
                every=5,
                total_epochs=10,
                boundary_improved=False,
                balanced_improved=False,
            )
        )
        self.assertTrue(
            _should_write_prediction_artifacts(
                enabled=True,
                epoch=3,
                every=5,
                total_epochs=10,
                boundary_improved=True,
                balanced_improved=False,
            )
        )
        self.assertTrue(
            _should_write_prediction_artifacts(
                enabled=True,
                epoch=10,
                every=5,
                total_epochs=10,
                boundary_improved=False,
                balanced_improved=False,
            )
        )

    def test_early_stop_counter_waits_for_first_valid_balanced_candidate(self) -> None:
        self.assertEqual(
            _update_early_stop_counter(
                0,
                enabled=True,
                epoch=3,
                armed_epoch=2,
                balanced_tracking_started=False,
                balanced_improved=False,
            ),
            0,
        )
        self.assertEqual(
            _update_early_stop_counter(
                2,
                enabled=True,
                epoch=3,
                armed_epoch=2,
                balanced_tracking_started=True,
                balanced_improved=True,
            ),
            0,
        )
        self.assertEqual(
            _update_early_stop_counter(
                2,
                enabled=True,
                epoch=3,
                armed_epoch=2,
                balanced_tracking_started=True,
                balanced_improved=False,
            ),
            3,
        )

    def test_save_every_alias_maps_to_save_every_epochs(self) -> None:
        args = train_mod.parse_args(["--train_dir", "outputs/bio_out_v2/synth_train", "--out_dir", "outputs/tmp", "--save_every", "3"])
        self.assertEqual(args.save_every_epochs, 3)
        self.assertTrue(args.save_every_deprecated_used)

    def test_parse_args_reads_prediction_artifacts_every(self) -> None:
        args = train_mod.parse_args(
            [
                "--train_dir", "outputs/bio_out_v2/synth_train",
                "--out_dir", "outputs/tmp",
                "--prediction_artifacts_every", "7",
            ]
        )
        self.assertEqual(args.prediction_artifacts_every, 7)

    def test_parse_args_can_disable_auto_workers(self) -> None:
        args = train_mod.parse_args(
            [
                "--train_dir", "outputs/bio_out_v2/synth_train",
                "--out_dir", "outputs/tmp",
                "--no_auto_workers",
            ]
        )
        self.assertFalse(args.auto_workers)

    def test_schedule_resume_sets_exact_next_epoch_lr(self) -> None:
        schedule_state = _make_schedule_state(base_lr=0.2, epochs=6, warmup_frac=0.5)
        uninterrupted = [_lr_for_epoch(schedule_state, epoch) for epoch in (1, 2)]
        ckpt_schedule = _checkpoint_schedule_state(schedule_state, completed_epochs=1)
        param = torch.nn.Parameter(torch.tensor(1.0))
        optim = torch.optim.Adam([param], lr=0.2)
        _set_optimizer_lr(optim, _lr_for_epoch(ckpt_schedule, int(ckpt_schedule["completed_epochs"]) + 1))
        self.assertAlmostEqual(optim.param_groups[0]["lr"], uninterrupted[1], places=12)

    def test_restore_history_from_compact_checkpoint_uses_neighbor_history_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt_path = root / "epoch_0002.pt"
            ckpt_path.write_bytes(b"placeholder")
            history = [
                {"epoch": 1, "train": {"loss": 1.0}},
                {"epoch": 2, "train": {"loss": 0.8}},
                {"epoch": 3, "train": {"loss": 0.7}},
            ]
            (root / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
            restored = _restore_history_from_checkpoint({"epoch": 2, "epoch_summary": {"epoch": 2}}, ckpt_path)
            self.assertEqual([row["epoch"] for row in restored], [1, 2])

    def test_deterministic_schedule_matches_pytorch_warmup_cosine(self) -> None:
        cases = [(2, 0.0), (5, 0.0), (5, 0.4), (5, 0.2), (3, 0.33)]
        for epochs, warmup_frac in cases:
            param = torch.nn.Parameter(torch.tensor(1.0))
            optim = torch.optim.AdamW([param], lr=0.002)
            warmup_epochs = int(round(warmup_frac * epochs))
            if warmup_epochs >= epochs:
                warmup_epochs = max(0, epochs - 1)
            if warmup_epochs > 0:
                main_epochs = max(1, epochs - warmup_epochs)
                sched = torch.optim.lr_scheduler.SequentialLR(
                    optim,
                    schedulers=[
                        torch.optim.lr_scheduler.LinearLR(optim, start_factor=0.1, total_iters=warmup_epochs),
                        torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=main_epochs),
                    ],
                    milestones=[warmup_epochs],
                )
            else:
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, epochs))
            expected = []
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for epoch in range(1, epochs + 1):
                    expected.append(float(optim.param_groups[0]["lr"]))
                    param.grad = torch.ones_like(param)
                    optim.step()
                    sched.step()
            schedule_state = _make_schedule_state(base_lr=0.002, epochs=epochs, warmup_frac=warmup_frac)
            actual = [_lr_for_epoch(schedule_state, epoch) for epoch in range(1, epochs + 1)]
            for exp_lr, act_lr in zip(expected, actual):
                self.assertAlmostEqual(exp_lr, act_lr, places=12)
            self.assertAlmostEqual(float(optim.param_groups[0]["lr"]), _lr_for_epoch(schedule_state, epochs + 1), places=12)

    def test_promote_many_transactional_rolls_back_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src_a = root / "src_a"
            src_b = root / "src_b"
            dst_a = root / "dst_a"
            dst_b = root / "dst_b"
            src_a.mkdir()
            src_b.mkdir()
            dst_a.mkdir()
            dst_b.mkdir()
            (src_a / "value.txt").write_text("new_a", encoding="utf-8")
            (src_b / "value.txt").write_text("new_b", encoding="utf-8")
            (dst_a / "value.txt").write_text("old_a", encoding="utf-8")
            (dst_b / "value.txt").write_text("old_b", encoding="utf-8")

            real_replace = __import__("os").replace

            def flaky_replace(src: str, dst: str) -> None:
                if Path(src).name == "src_b" and Path(dst).name == "dst_b":
                    raise OSError("boom")
                real_replace(src, dst)

            with mock.patch("bio.pipeline.build_dataset.os.replace", side_effect=flaky_replace):
                with self.assertRaises(OSError):
                    _promote_many_transactional([(src_a, dst_a), (src_b, dst_b)])

            self.assertEqual((dst_a / "value.txt").read_text(encoding="utf-8"), "old_a")
            self.assertEqual((dst_b / "value.txt").read_text(encoding="utf-8"), "old_b")
            self.assertTrue(src_b.exists())

    def test_main_fails_fast_when_val_has_no_b_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_dir = _write_tiny_synth_dir(
                root / "train",
                bios=[np.asarray([0, 1, 2, 0], dtype=np.uint8), np.asarray([0, 1, 2, 2], dtype=np.uint8)],
            )
            val_dir = _write_tiny_synth_dir(
                root / "val",
                bios=[np.asarray([0, 0, 2, 2], dtype=np.uint8)],
            )
            out_dir = root / "run"
            with self.assertRaisesRegex(RuntimeError, "Validation split contains no true B events"):
                train_mod.main(
                    [
                        "--train_dir", str(train_dir),
                        "--val_dir", str(val_dir),
                        "--out_dir", str(out_dir),
                        "--device", "cpu",
                        "--epochs", "1",
                        "--batch_size", "1",
                        "--num_workers", "0",
                        "--no_amp",
                        "--save_every_epochs", "1",
                        "--no_save_analysis_artifacts",
                        "--ema_decay", "0.0",
                    ]
                )
            self.assertFalse((out_dir / "best_boundary.pt").exists())
            self.assertFalse((out_dir / "best_balanced.pt").exists())

    def test_synth_build_cli_dirs_override_config_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "bio_cfg.json"
            config_path.write_text(
                json.dumps(
                    {
                        "synth_build": {
                            "prelabel_dir": "cfg/prelabels",
                            "out_dir": "cfg/out",
                            "preferred_noev_prelabel_dir": ["cfg/preferred"],
                            "extra_noev_prelabel_dir": ["cfg/extra"],
                        }
                    }
                ),
                encoding="utf-8",
            )
            args = synth_build.parse_args(
                [
                    "--config", str(config_path),
                    "--prelabel_dir", "cli/prelabels",
                    "--out_dir", "cli/out",
                    "--preferred_noev_prelabel_dir", "cli/preferred",
                    "--extra_noev_prelabel_dir", "cli/extra",
                ]
            )
            self.assertEqual(args.preferred_noev_prelabel_dir, ["cli/preferred"])
            self.assertEqual(args.extra_noev_prelabel_dir, ["cli/extra"])


if __name__ == "__main__":
    unittest.main()
