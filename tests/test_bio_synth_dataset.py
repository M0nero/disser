from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bio.core.datasets.synth_dataset import (
    PrelabelRow,
    SynthConfig,
    SyntheticContinuousDataset,
    _align_chunk_to_tail,
    _build_source_groups,
    _compute_seam_diagnostics,
    _pick_pool_segment,
    load_prelabel_index,
    split_pools,
)
from bio.pipeline.continuous_stats import build_continuous_stats
from bio.pipeline.synth_build import _resolve_synth_workers, build_offline


def _write_step1_npz(
    path: Path,
    *,
    pts_value: float,
    length: int,
    bio: np.ndarray,
    start_idx: int,
    end_idx: int,
    is_no_event: bool,
    frame_values: np.ndarray | None = None,
    raw_frame_values: np.ndarray | None = None,
) -> None:
    if frame_values is None:
        pts = np.full((length, 42, 3), pts_value, dtype=np.float32)
    else:
        vals = np.asarray(frame_values, dtype=np.float32).reshape(length, 1, 1)
        pts = np.broadcast_to(vals, (length, 42, 3)).copy()
    if raw_frame_values is None:
        pts_raw = pts.copy()
    else:
        raw_vals = np.asarray(raw_frame_values, dtype=np.float32).reshape(length, 1, 1)
        pts_raw = np.broadcast_to(raw_vals, (length, 42, 3)).copy()
    mask = np.ones((length, 42, 1), dtype=np.float32)
    ts = np.arange(length, dtype=np.float32)
    meta = {
        "dataset": "slovo",
        "split": "train",
        "source_group": path.stem.split("__", 1)[0],
        "preprocessing_version": "canonical_hands42_v3",
        "coords": "image",
        "raw_pts_key": "pts_raw",
    }
    np.savez(
        path,
        pts=pts,
        pts_raw=pts_raw,
        mask=mask,
        ts=ts,
        bio=bio.astype(np.uint8),
        label_str=np.asarray("no_event" if is_no_event else "hello"),
        is_no_event=np.asarray(is_no_event),
        start_idx=np.asarray(start_idx, dtype=np.int64),
        end_idx=np.asarray(end_idx, dtype=np.int64),
        meta=np.asarray(json.dumps(meta)),
    )


class SynthDatasetTests(unittest.TestCase):
    def test_compute_seam_diagnostics_reports_semantic_vs_expanded_boundaries(self) -> None:
        pts = np.zeros((6, 42, 3), dtype=np.float32)
        for idx, value in enumerate([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]):
            pts[idx, :, 0] = value
        mask = np.ones((6, 42, 1), dtype=np.float32)
        parts = [
            {"type": "sign", "source_group": "clip_a", "length": 2},
            {"type": "transition", "source_group": "clip_a", "length": 2},
            {"type": "no_event", "source_group": "clip_a", "length": 2},
        ]
        diag = _compute_seam_diagnostics(pts, mask, parts)
        expanded = dict(diag.get("expanded", {}) or {})
        semantic = dict(diag.get("semantic", {}) or {})
        self.assertIn("sign->transition", dict(expanded.get("boundary_type_counts", {}) or {}))
        self.assertIn("transition->no_event", dict(expanded.get("boundary_type_counts", {}) or {}))
        self.assertIn("sign->no_event", dict(semantic.get("boundary_type_counts", {}) or {}))
        self.assertNotIn("sign->transition", dict(semantic.get("boundary_type_counts", {}) or {}))
        self.assertNotEqual(
            dict(expanded.get("boundary_type_counts", {}) or {}),
            dict(semantic.get("boundary_type_counts", {}) or {}),
        )

    def test_compute_seam_diagnostics_ignores_pad_edges_in_semantic_scope(self) -> None:
        pts = np.zeros((6, 42, 3), dtype=np.float32)
        pts[:, :, 0] = np.asarray([0.0, 0.1, 0.2, 0.3, 0.0, 0.0], dtype=np.float32).reshape(6, 1)
        mask = np.ones((6, 42, 1), dtype=np.float32)
        parts = [
            {"type": "sign", "source_group": "user_a", "length": 4},
            {"type": "no_event", "role": "pad_post", "source_group": "", "length": 2},
        ]
        diag = _compute_seam_diagnostics(pts, mask, parts)
        expanded = dict(diag.get("expanded", {}) or {})
        semantic = dict(diag.get("semantic", {}) or {})
        self.assertEqual(int(expanded.get("boundary_count", 0)), 1)
        self.assertEqual(int(semantic.get("boundary_count", 0)), 0)

    def test_split_pools_extracts_tails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sign_path = root / "clip_a.npz"
            bio = np.zeros((10,), dtype=np.uint8)
            bio[3] = 1
            bio[4:7] = 2
            _write_step1_npz(sign_path, pts_value=1.0, length=10, bio=bio, start_idx=3, end_idx=6, is_no_event=False)
            rows = [
                PrelabelRow(
                    vid="clip_a",
                    label_str="hello",
                    path_to_npz=sign_path.name,
                    T_total=10,
                    start_idx=3,
                    end_idx=6,
                    is_no_event=False,
                    split="train",
                    dataset="slovo",
                    source_group="clip_a",
                )
            ]
            sign_pool, noev_pool = split_pools(root, rows, min_sign_len=1, include_sign_tails_as_noev=True, min_tail_len=2)
            self.assertEqual(len(sign_pool), 1)
            self.assertCountEqual([seg.source_type for seg in noev_pool], ["tail_pre", "tail_post"])
            self.assertEqual([seg.length for seg in noev_pool], [3, 3])

    def test_uniform_source_sampling_balances_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sign_path = root / "sign_seed.npz"
            bio = np.zeros((6,), dtype=np.uint8)
            bio[1] = 1
            bio[2:5] = 2
            _write_step1_npz(sign_path, pts_value=7.0, length=6, bio=bio, start_idx=1, end_idx=4, is_no_event=False)
            rows = []
            rows.append(
                PrelabelRow(
                    vid="sign_seed",
                    label_str="hello",
                    path_to_npz=sign_path.name,
                    T_total=6,
                    start_idx=1,
                    end_idx=4,
                    is_no_event=False,
                    split="train",
                    dataset="slovo",
                    source_group="sign_seed",
                )
            )
            for idx in range(8):
                path = root / f"clip_a_{idx}.npz"
                _write_step1_npz(path, pts_value=float(idx + 1), length=6, bio=np.zeros((6,), dtype=np.uint8), start_idx=-1, end_idx=-1, is_no_event=True)
                rows.append(
                    PrelabelRow(
                        vid=path.stem,
                        label_str="no_event",
                        path_to_npz=path.name,
                        T_total=6,
                        start_idx=-1,
                        end_idx=-1,
                        is_no_event=True,
                        split="train",
                        dataset="slovo",
                        source_group="group_a",
                    )
                )
            path_b = root / "clip_b.npz"
            _write_step1_npz(path_b, pts_value=99.0, length=6, bio=np.zeros((6,), dtype=np.uint8), start_idx=-1, end_idx=-1, is_no_event=True)
            rows.append(
                PrelabelRow(
                    vid="clip_b",
                    label_str="no_event",
                    path_to_npz=path_b.name,
                    T_total=6,
                    start_idx=-1,
                    end_idx=-1,
                    is_no_event=True,
                    split="train",
                    dataset="slovo",
                    source_group="group_b",
                )
            )
            _, noev_pool = split_pools(root, rows, min_sign_len=1, include_sign_tails_as_noev=False, min_tail_len=4)
            grouped = _build_source_groups(noev_pool)
            rng = np.random.default_rng(123)
            counts = {"group_a": 0, "group_b": 0}
            for _ in range(400):
                seg = _pick_pool_segment(rng, noev_pool, grouped, "uniform_source")
                counts[seg.source_group] += 1
            self.assertGreater(counts["group_a"], 120)
            self.assertGreater(counts["group_b"], 120)

    def test_uniform_label_source_sampling_balances_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for idx in range(8):
                path = root / f"hello_{idx}.npz"
                bio = np.zeros((6,), dtype=np.uint8)
                bio[1] = 1
                bio[2:5] = 2
                _write_step1_npz(path, pts_value=float(idx + 1), length=6, bio=bio, start_idx=1, end_idx=4, is_no_event=False)
                rows.append(
                    PrelabelRow(
                        vid=path.stem,
                        label_str="hello",
                        path_to_npz=path.name,
                        T_total=6,
                        start_idx=1,
                        end_idx=4,
                        is_no_event=False,
                        split="train",
                        dataset="slovo",
                        source_group=f"hello_group_{idx}",
                    )
                )
            rare_path = root / "rare.npz"
            bio = np.zeros((6,), dtype=np.uint8)
            bio[1] = 1
            bio[2:5] = 2
            _write_step1_npz(rare_path, pts_value=99.0, length=6, bio=bio, start_idx=1, end_idx=4, is_no_event=False)
            rows.append(
                PrelabelRow(
                    vid="rare",
                    label_str="rare",
                    path_to_npz=rare_path.name,
                    T_total=6,
                    start_idx=1,
                    end_idx=4,
                    is_no_event=False,
                    split="train",
                    dataset="slovo",
                    source_group="rare_group",
                )
            )
            noev_path = root / "noev.npz"
            _write_step1_npz(noev_path, pts_value=0.0, length=6, bio=np.zeros((6,), dtype=np.uint8), start_idx=-1, end_idx=-1, is_no_event=True)
            rows.append(
                PrelabelRow(
                    vid="noev",
                    label_str="no_event",
                    path_to_npz=noev_path.name,
                    T_total=6,
                    start_idx=-1,
                    end_idx=-1,
                    is_no_event=True,
                    split="train",
                    dataset="slovo",
                    source_group="noev_group",
                )
            )

            ds = SyntheticContinuousDataset(
                prelabel_dir=root,
                rows=rows,
                cfg=SynthConfig(seq_len=16, min_signs=1, max_signs=1, sign_sampling="uniform_label_source"),
                epoch_size=1,
                seed=7,
                min_sign_len=1,
            )
            rng = np.random.default_rng(123)
            counts = {"hello": 0, "rare": 0}
            for _ in range(400):
                _x, _m, _y, meta = ds._pick_sign_segment(rng)
                counts[str(meta["label_str"])] += 1
            self.assertGreater(counts["hello"], 120)
            self.assertGreater(counts["rare"], 120)

    def test_stitching_fills_exact_no_event_length(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sign_path = root / "clip_sign.npz"
            bio = np.zeros((6,), dtype=np.uint8)
            bio[1] = 1
            bio[2:5] = 2
            _write_step1_npz(sign_path, pts_value=5.0, length=6, bio=bio, start_idx=1, end_idx=4, is_no_event=False)

            noev_a = root / "clip_noev_a.npz"
            noev_b = root / "clip_noev_b.npz"
            _write_step1_npz(
                noev_a,
                pts_value=1.0,
                length=2,
                bio=np.zeros((2,), dtype=np.uint8),
                start_idx=-1,
                end_idx=-1,
                is_no_event=True,
                frame_values=np.asarray([1.0, 2.0], dtype=np.float32),
            )
            _write_step1_npz(
                noev_b,
                pts_value=9.0,
                length=2,
                bio=np.zeros((2,), dtype=np.uint8),
                start_idx=-1,
                end_idx=-1,
                is_no_event=True,
                frame_values=np.asarray([9.0, 10.0], dtype=np.float32),
            )

            index_rows = [
                {
                    "vid": "clip_sign",
                    "label_str": "hello",
                    "path_to_npz": sign_path.name,
                    "T_total": 6,
                    "start_idx": 1,
                    "end_idx": 4,
                    "is_no_event": False,
                    "split": "train",
                    "dataset": "slovo",
                    "source_group": "clip_sign",
                },
                {
                    "vid": "clip_noev_a",
                    "label_str": "no_event",
                    "path_to_npz": noev_a.name,
                    "T_total": 2,
                    "start_idx": -1,
                    "end_idx": -1,
                    "is_no_event": True,
                    "split": "train",
                    "dataset": "slovo",
                    "source_group": "group_a",
                },
                {
                    "vid": "clip_noev_b",
                    "label_str": "no_event",
                    "path_to_npz": noev_b.name,
                    "T_total": 2,
                    "start_idx": -1,
                    "end_idx": -1,
                    "is_no_event": True,
                    "split": "train",
                    "dataset": "slovo",
                    "source_group": "group_b",
                },
            ]
            (root / "index.json").write_text(json.dumps(index_rows), encoding="utf-8")

            ds = SyntheticContinuousDataset(
                prelabel_dir=root,
                rows=load_prelabel_index(root),
                cfg=SynthConfig(seq_len=16, min_signs=1, max_signs=1, stitch_noev_chunks=True, source_sampling="uniform_source"),
                epoch_size=1,
                seed=7,
                min_sign_len=1,
            )
            x, _m, y, parts = ds._pick_no_event_chunk(np.random.default_rng(0), 5, role="gap")
            self.assertEqual(x.shape[0], 5)
            self.assertEqual(y.shape[0], 5)
            self.assertGreaterEqual(len(parts), 2)
            self.assertFalse(np.allclose(x[:, 0, 0], np.asarray([9.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float32)))

    def test_noncanonical_step1_index_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "index.json").write_text(
                json.dumps(
                    [
                        {
                            "vid": "clip_a",
                            "label_str": "hello",
                            "path_to_npz": "clip_a.npz",
                            "T_total": 10,
                            "start_idx": 1,
                            "end_idx": 5,
                            "is_no_event": False,
                            "dataset": "slovo",
                            "source_group": "clip_a",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                load_prelabel_index(root)

    def test_no_event_side_pool_must_be_canonical_no_event_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sign_root = root / "sign"
            extra_root = root / "extra"
            sign_root.mkdir()
            extra_root.mkdir()

            sign_path = sign_root / "clip_sign.npz"
            sign_bio = np.zeros((6,), dtype=np.uint8)
            sign_bio[1] = 1
            sign_bio[2:5] = 2
            _write_step1_npz(sign_path, pts_value=5.0, length=6, bio=sign_bio, start_idx=1, end_idx=4, is_no_event=False)
            (sign_root / "index.json").write_text(
                json.dumps(
                    [
                        {
                            "vid": "clip_sign",
                            "label_str": "hello",
                            "path_to_npz": sign_path.name,
                            "T_total": 6,
                            "start_idx": 1,
                            "end_idx": 4,
                            "is_no_event": False,
                            "split": "train",
                            "dataset": "slovo",
                            "source_group": "clip_sign",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            extra_path = extra_root / "clip_bad.npz"
            _write_step1_npz(extra_path, pts_value=9.0, length=6, bio=sign_bio, start_idx=1, end_idx=4, is_no_event=False)
            (extra_root / "index.json").write_text(
                json.dumps(
                    [
                        {
                            "vid": "clip_bad",
                            "label_str": "hello",
                            "path_to_npz": extra_path.name,
                            "T_total": 6,
                            "start_idx": 1,
                            "end_idx": 4,
                            "is_no_event": False,
                            "split": "train",
                            "dataset": "slovo",
                            "source_group": "clip_bad",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaises(RuntimeError):
                SyntheticContinuousDataset(
                    prelabel_dir=sign_root,
                    rows=load_prelabel_index(sign_root),
                    cfg=SynthConfig(seq_len=16, min_signs=1, max_signs=1),
                    epoch_size=1,
                    seed=7,
                    min_sign_len=1,
                    extra_noev_prelabel_dirs=[extra_root],
                )

    def test_generate_one_supports_leading_o_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sign_path = root / "sign.npz"
            sign_bio = np.asarray([1, 2, 2, 2], dtype=np.uint8)
            _write_step1_npz(sign_path, pts_value=5.0, length=4, bio=sign_bio, start_idx=0, end_idx=3, is_no_event=False)
            noev_path = root / "noev.npz"
            _write_step1_npz(noev_path, pts_value=0.0, length=12, bio=np.zeros((12,), dtype=np.uint8), start_idx=-1, end_idx=-1, is_no_event=True)
            rows = [
                PrelabelRow("sign", "hello", sign_path.name, 4, 0, 3, False, "train", "slovo", "sign_group"),
                PrelabelRow("noev", "no_event", noev_path.name, 12, -1, -1, True, "train", "slovo", "noev_group"),
            ]
            ds = SyntheticContinuousDataset(
                prelabel_dir=root,
                rows=rows,
                cfg=SynthConfig(
                    seq_len=9,
                    min_signs=1,
                    max_signs=1,
                    gap_min=0,
                    gap_max=0,
                    leading_noev_prob=1.0,
                    leading_noev_min=5,
                    leading_noev_max=5,
                    all_noev_prob=0.0,
                    pad_mode="end_no_event",
                    transition_all_boundaries=False,
                    align_chunks=False,
                    continuous_mode_weight=1.0,
                    hard_negative_mode_weight=0.0,
                    stress_mode_weight=0.0,
                ),
                epoch_size=1,
                seed=11,
                min_sign_len=1,
            )
            sample = ds[0]
            self.assertTrue(sample["meta"]["has_leading_o_prefix"])
            self.assertGreater(int(sample["meta"]["first_B_frame"]), 0)
            self.assertEqual(int(sample["meta"]["leading_o_prefix_len"]), int(sample["meta"]["first_B_frame"]))
            first_b = int(sample["meta"]["first_B_frame"])
            self.assertTrue(np.all(sample["bio"][:first_b] == 0))
            self.assertEqual(int(sample["bio"][first_b]), 1)

    def test_generate_one_supports_all_o_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sign_path = root / "sign.npz"
            sign_bio = np.asarray([1, 2, 2, 2], dtype=np.uint8)
            _write_step1_npz(sign_path, pts_value=5.0, length=4, bio=sign_bio, start_idx=0, end_idx=3, is_no_event=False)
            noev_path = root / "noev.npz"
            _write_step1_npz(noev_path, pts_value=0.0, length=12, bio=np.zeros((12,), dtype=np.uint8), start_idx=-1, end_idx=-1, is_no_event=True)
            rows = [
                PrelabelRow("sign", "hello", sign_path.name, 4, 0, 3, False, "train", "slovo", "sign_group"),
                PrelabelRow("noev", "no_event", noev_path.name, 12, -1, -1, True, "train", "slovo", "noev_group"),
            ]
            ds = SyntheticContinuousDataset(
                prelabel_dir=root,
                rows=rows,
                cfg=SynthConfig(
                    seq_len=12,
                    min_signs=1,
                    max_signs=1,
                    gap_min=0,
                    gap_max=0,
                    all_noev_prob=1.0,
                    all_noev_min_len=12,
                    all_noev_max_len=12,
                ),
                epoch_size=1,
                seed=13,
                min_sign_len=1,
            )
            sample = ds[0]
            self.assertTrue(sample["meta"]["is_all_noev"])
            self.assertIsNone(sample["meta"]["first_B_frame"])
            self.assertTrue(np.all(sample["bio"] == 0))

    def test_build_offline_stats_report_startup_negative_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sign_path = root / "sign.npz"
            sign_bio = np.asarray([1, 2, 2, 2], dtype=np.uint8)
            _write_step1_npz(sign_path, pts_value=5.0, length=4, bio=sign_bio, start_idx=0, end_idx=3, is_no_event=False)
            noev_path = root / "noev.npz"
            _write_step1_npz(noev_path, pts_value=0.0, length=12, bio=np.zeros((12,), dtype=np.uint8), start_idx=-1, end_idx=-1, is_no_event=True)
            index_rows = [
                {
                    "vid": "sign",
                    "label_str": "hello",
                    "path_to_npz": sign_path.name,
                    "T_total": 4,
                    "start_idx": 0,
                    "end_idx": 3,
                    "is_no_event": False,
                    "split": "train",
                    "dataset": "slovo",
                    "source_group": "sign_group",
                },
                {
                    "vid": "noev",
                    "label_str": "no_event",
                    "path_to_npz": noev_path.name,
                    "T_total": 12,
                    "start_idx": -1,
                    "end_idx": -1,
                    "is_no_event": True,
                    "split": "train",
                    "dataset": "slovo",
                    "source_group": "noev_group",
                },
            ]
            (root / "index.json").write_text(json.dumps(index_rows, ensure_ascii=False, indent=2), encoding="utf-8")
            stats = build_offline(
                prelabel_dir=root,
                out_dir=root / "out",
                cfg=SynthConfig(
                    seq_len=9,
                    min_signs=1,
                    max_signs=1,
                    gap_min=0,
                    gap_max=0,
                    leading_noev_prob=1.0,
                    leading_noev_min=5,
                    leading_noev_max=5,
                    all_noev_prob=0.0,
                    pad_mode="end_no_event",
                    transition_all_boundaries=False,
                    align_chunks=False,
                    continuous_mode_weight=1.0,
                    hard_negative_mode_weight=0.0,
                    stress_mode_weight=0.0,
                ),
                num_samples=4,
                shard_size=2,
                seed=17,
                min_sign_len=1,
            )
            generated = stats.generated
            self.assertEqual(generated["samples_with_leading_o_prefix_frac"], 1.0)
            self.assertEqual(generated["all_o_samples_frac"], 0.0)
            self.assertEqual(generated["first_B_frame_eq0_frac_over_samples_with_B"], 0.0)
            self.assertTrue(all(int(k) > 0 for k in generated["first_B_frame_distribution"].keys()))
            self.assertEqual(generated["first_B_frame_eq0_frac_over_samples_with_B"], 0.0)

    def test_generate_one_reports_post_assembly_hand_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sign_path = root / "sign.npz"
            sign_bio = np.asarray([1, 2, 2, 2, 2, 2], dtype=np.uint8)
            _write_step1_npz(sign_path, pts_value=5.0, length=6, bio=sign_bio, start_idx=0, end_idx=5, is_no_event=False)
            noev_path = root / "noev.npz"
            _write_step1_npz(noev_path, pts_value=0.0, length=12, bio=np.zeros((12,), dtype=np.uint8), start_idx=-1, end_idx=-1, is_no_event=True)
            rows = [
                PrelabelRow("sign", "hello", sign_path.name, 6, 0, 5, False, "train", "slovo", "sign_group"),
                PrelabelRow("noev", "no_event", noev_path.name, 12, -1, -1, True, "train", "slovo", "noev_group"),
            ]
            ds = SyntheticContinuousDataset(
                prelabel_dir=root,
                rows=rows,
                cfg=SynthConfig(
                    seq_len=8,
                    min_signs=1,
                    max_signs=1,
                    gap_min=0,
                    gap_max=0,
                    single_hand_dropout_prob=0.0,
                    both_hands_dropout_prob=1.0,
                    both_hands_dropout_span_min=8,
                    both_hands_dropout_span_max=8,
                    mask_flicker_prob=0.0,
                    joint_jitter_prob=0.0,
                    continuous_mode_weight=1.0,
                    hard_negative_mode_weight=0.0,
                    stress_mode_weight=0.0,
                ),
                epoch_size=1,
                seed=23,
                min_sign_len=1,
            )
            sample = ds[0]
            self.assertTrue(sample["meta"]["corruption_applied"])
            self.assertIn("both_hands_dropout", sample["meta"]["corruption_counts"])
            self.assertEqual(int(sample["mask"].sum()), 0)
            self.assertEqual(int(sample["meta"]["longest_no_hand_span_after_corruption"]), 8)

    def test_chunk_alignment_reduces_boundary_center_jump(self) -> None:
        prev_pts = np.zeros((4, 42, 3), dtype=np.float32)
        prev_mask = np.ones((4, 42, 1), dtype=np.float32)
        next_pts = np.zeros((4, 42, 3), dtype=np.float32)
        next_mask = np.ones((4, 42, 1), dtype=np.float32)
        for j in range(21):
            prev_pts[:, 21 + j, 0] = 0.1 + 0.01 * j
            prev_pts[:, 21 + j, 1] = 0.2 + 0.005 * j
            next_pts[:, 21 + j, 0] = 1.5 + 0.02 * j
            next_pts[:, 21 + j, 1] = 1.8 + 0.01 * j
        raw_diag = _compute_seam_diagnostics(
            np.concatenate([prev_pts, next_pts], axis=0),
            np.concatenate([prev_mask, next_mask], axis=0),
            [
                {"type": "sign", "length": 4, "source_group": "a"},
                {"type": "sign", "length": 4, "source_group": "a"},
            ],
        )
        aligned_next, align_meta = _align_chunk_to_tail(prev_pts, prev_mask, next_pts, next_mask)
        self.assertTrue(align_meta["aligned"])
        aligned_diag = _compute_seam_diagnostics(
            np.concatenate([prev_pts, aligned_next], axis=0),
            np.concatenate([prev_mask, next_mask], axis=0),
            [
                {"type": "sign", "length": 4, "source_group": "a"},
                {"type": "sign", "length": 4, "source_group": "a"},
            ],
        )
        self.assertLess(
            float(np.mean(aligned_diag["boundary_center_deltas"])),
            float(np.mean(raw_diag["boundary_center_deltas"])),
        )

    def test_generate_one_prefers_same_source_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sign_a = root / "sign_a.npz"
            sign_b = root / "sign_b.npz"
            noev_a = root / "noev_a.npz"
            noev_b = root / "noev_b.npz"
            sign_bio = np.asarray([1, 2, 2, 2], dtype=np.uint8)
            _write_step1_npz(sign_a, pts_value=1.0, length=4, bio=sign_bio, start_idx=0, end_idx=3, is_no_event=False)
            _write_step1_npz(sign_b, pts_value=2.0, length=4, bio=sign_bio, start_idx=0, end_idx=3, is_no_event=False)
            _write_step1_npz(noev_a, pts_value=0.5, length=8, bio=np.zeros((8,), dtype=np.uint8), start_idx=-1, end_idx=-1, is_no_event=True)
            _write_step1_npz(noev_b, pts_value=0.75, length=8, bio=np.zeros((8,), dtype=np.uint8), start_idx=-1, end_idx=-1, is_no_event=True)
            rows = [
                PrelabelRow("sign_a", "hello", sign_a.name, 4, 0, 3, False, "train", "slovo", "group_a"),
                PrelabelRow("sign_b", "hello", sign_b.name, 4, 0, 3, False, "train", "slovo", "group_b"),
                PrelabelRow("noev_a", "no_event", noev_a.name, 8, -1, -1, True, "train", "slovo", "group_a"),
                PrelabelRow("noev_b", "no_event", noev_b.name, 8, -1, -1, True, "train", "slovo", "group_b"),
            ]
            ds = SyntheticContinuousDataset(
                prelabel_dir=root,
                rows=rows,
                cfg=SynthConfig(
                    seq_len=16,
                    min_signs=2,
                    max_signs=2,
                    gap_min=4,
                    gap_max=4,
                    same_source_sequence_prob=1.0,
                    cross_source_boundary_prob=0.0,
                    transition_all_boundaries=False,
                    align_chunks=False,
                ),
                epoch_size=1,
                seed=31,
                min_sign_len=1,
            )
            sample = ds[0]
            groups = {str(part.get("source_group", "")) for part in sample["meta"]["parts"] if str(part.get("source_group", ""))}
            self.assertEqual(len(groups), 1)

    def test_pick_sign_segment_uniform_label_source_respects_strict_same_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sign_a = root / "group_a_label_a.npz"
            sign_b = root / "group_b_label_b.npz"
            sign_c = root / "group_a_label_c.npz"
            noev = root / "group_a_noev.npz"
            sign_bio = np.asarray([1, 2, 2, 2], dtype=np.uint8)
            _write_step1_npz(sign_a, pts_value=1.0, length=4, bio=sign_bio, start_idx=0, end_idx=3, is_no_event=False)
            _write_step1_npz(sign_b, pts_value=2.0, length=4, bio=sign_bio, start_idx=0, end_idx=3, is_no_event=False)
            _write_step1_npz(sign_c, pts_value=3.0, length=4, bio=sign_bio, start_idx=0, end_idx=3, is_no_event=False)
            _write_step1_npz(noev, pts_value=0.0, length=8, bio=np.zeros((8,), dtype=np.uint8), start_idx=-1, end_idx=-1, is_no_event=True)
            rows = [
                PrelabelRow("clip_a", "label_a", sign_a.name, 4, 0, 3, False, "train", "slovo", "group_a"),
                PrelabelRow("clip_b", "label_b", sign_b.name, 4, 0, 3, False, "train", "slovo", "group_b"),
                PrelabelRow("clip_c", "label_c", sign_c.name, 4, 0, 3, False, "train", "slovo", "group_a"),
                PrelabelRow("noev", "no_event", noev.name, 8, -1, -1, True, "train", "slovo", "group_a"),
            ]
            ds = SyntheticContinuousDataset(
                prelabel_dir=root,
                rows=rows,
                cfg=SynthConfig(seq_len=16, min_signs=1, max_signs=1, sign_sampling="uniform_label_source"),
                epoch_size=1,
                seed=31,
                min_sign_len=1,
            )
            rng = np.random.default_rng(7)
            for _ in range(20):
                _x, _m, _y, meta = ds._pick_sign_segment(
                    rng,
                    preferred_source_group="group_a",
                    allow_cross_source=False,
                    strict_same_source=True,
                )
                self.assertEqual(str(meta.get("source_group", "")), "group_a")

    def test_build_offline_supports_parallel_workers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sign_path = root / "sign.npz"
            sign_bio = np.asarray([1, 2, 2, 2], dtype=np.uint8)
            _write_step1_npz(sign_path, pts_value=5.0, length=4, bio=sign_bio, start_idx=0, end_idx=3, is_no_event=False)
            noev_path = root / "noev.npz"
            _write_step1_npz(noev_path, pts_value=0.0, length=12, bio=np.zeros((12,), dtype=np.uint8), start_idx=-1, end_idx=-1, is_no_event=True)
            index_rows = [
                {
                    "vid": "sign",
                    "label_str": "hello",
                    "path_to_npz": sign_path.name,
                    "T_total": 4,
                    "start_idx": 0,
                    "end_idx": 3,
                    "is_no_event": False,
                    "split": "train",
                    "dataset": "slovo",
                    "source_group": "sign_group",
                },
                {
                    "vid": "noev",
                    "label_str": "no_event",
                    "path_to_npz": noev_path.name,
                    "T_total": 12,
                    "start_idx": -1,
                    "end_idx": -1,
                    "is_no_event": True,
                    "split": "train",
                    "dataset": "slovo",
                    "source_group": "noev_group",
                },
            ]
            (root / "index.json").write_text(json.dumps(index_rows, ensure_ascii=False, indent=2), encoding="utf-8")
            out_dir = root / "out_parallel"
            stats = build_offline(
                prelabel_dir=root,
                out_dir=out_dir,
                cfg=SynthConfig(
                    seq_len=9,
                    min_signs=1,
                    max_signs=1,
                    gap_min=0,
                    gap_max=0,
                    leading_noev_prob=1.0,
                    leading_noev_min=5,
                    leading_noev_max=5,
                    all_noev_prob=0.0,
                    pad_mode="end_no_event",
                    transition_all_boundaries=False,
                    align_chunks=False,
                    continuous_mode_weight=1.0,
                    hard_negative_mode_weight=0.0,
                    stress_mode_weight=0.0,
                ),
                num_samples=4,
                shard_size=2,
                seed=17,
                min_sign_len=1,
                workers=2,
            )
            self.assertEqual(int(stats.generated["shards"]), 2)
            self.assertEqual(stats.generated["samples_with_leading_o_prefix_frac"], 1.0)
            self.assertEqual(stats.generated["first_B_frame_eq0_frac_over_samples_with_B"], 0.0)
            self.assertTrue(all(int(k) > 0 for k in stats.generated["first_B_frame_distribution"].keys()))
            self.assertIn("seam_realism", stats.generated)
            self.assertIn("boundary_internal_center_jump_ratio", stats.generated["seam_realism"])
            self.assertIn("acceptance", stats.generated)
            self.assertIn("config", json.loads((out_dir / "stats.json").read_text(encoding="utf-8")))
            raw_index = json.loads((out_dir / "index.json").read_text(encoding="utf-8"))
            self.assertEqual([str(row.get("id", "")) for row in raw_index], ["shard_000000", "shard_000001"])

    def test_build_offline_resume_reuses_existing_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sign_path = root / "sign.npz"
            sign_bio = np.asarray([1, 2, 2, 2], dtype=np.uint8)
            _write_step1_npz(sign_path, pts_value=5.0, length=4, bio=sign_bio, start_idx=0, end_idx=3, is_no_event=False)
            noev_path = root / "noev.npz"
            _write_step1_npz(noev_path, pts_value=0.0, length=12, bio=np.zeros((12,), dtype=np.uint8), start_idx=-1, end_idx=-1, is_no_event=True)
            index_rows = [
                {
                    "vid": "sign",
                    "label_str": "hello",
                    "path_to_npz": sign_path.name,
                    "T_total": 4,
                    "start_idx": 0,
                    "end_idx": 3,
                    "is_no_event": False,
                    "split": "train",
                    "dataset": "slovo",
                    "source_group": "sign_group",
                },
                {
                    "vid": "noev",
                    "label_str": "no_event",
                    "path_to_npz": noev_path.name,
                    "T_total": 12,
                    "start_idx": -1,
                    "end_idx": -1,
                    "is_no_event": True,
                    "split": "train",
                    "dataset": "slovo",
                    "source_group": "noev_group",
                },
            ]
            (root / "index.json").write_text(json.dumps(index_rows, ensure_ascii=False, indent=2), encoding="utf-8")
            out_dir = root / "out_resume"
            cfg = SynthConfig(
                seq_len=9,
                min_signs=1,
                max_signs=1,
                gap_min=0,
                gap_max=0,
                leading_noev_prob=1.0,
                leading_noev_min=5,
                leading_noev_max=5,
                all_noev_prob=0.0,
                pad_mode="end_no_event",
                transition_all_boundaries=False,
                align_chunks=False,
                continuous_mode_weight=1.0,
                hard_negative_mode_weight=0.0,
                stress_mode_weight=0.0,
            )
            stats_first = build_offline(
                prelabel_dir=root,
                out_dir=out_dir,
                cfg=cfg,
                num_samples=4,
                shard_size=2,
                seed=17,
                min_sign_len=1,
            )
            self.assertTrue((out_dir / "shards" / "shard_000000.stats.json").is_file())
            self.assertTrue((out_dir / "shards" / "shard_000001.stats.json").is_file())
            (out_dir / "stats.json").unlink()
            (out_dir / "index.json").unlink()
            (out_dir / "index.csv").unlink()
            (out_dir / "shards" / "shard_000001.stats.json").unlink()
            stats_resumed = build_offline(
                prelabel_dir=root,
                out_dir=out_dir,
                cfg=cfg,
                num_samples=4,
                shard_size=2,
                seed=17,
                min_sign_len=1,
                resume=True,
            )
            self.assertEqual(stats_first.generated["first_B_frame_distribution"], stats_resumed.generated["first_B_frame_distribution"])
            self.assertEqual(stats_resumed.generated["samples_with_leading_o_prefix_frac"], 1.0)
            self.assertTrue((out_dir / "shards" / "shard_000001.stats.json").is_file())

    def test_generation_mode_weights_drive_selected_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sign_path = root / "sign.npz"
            noev_path = root / "noev.npz"
            sign_bio = np.asarray([1, 2, 2, 2], dtype=np.uint8)
            _write_step1_npz(sign_path, pts_value=5.0, length=4, bio=sign_bio, start_idx=0, end_idx=3, is_no_event=False)
            _write_step1_npz(noev_path, pts_value=0.0, length=12, bio=np.zeros((12,), dtype=np.uint8), start_idx=-1, end_idx=-1, is_no_event=True)
            rows = [
                PrelabelRow("sign", "hello", sign_path.name, 4, 0, 3, False, "train", "slovo", "sign_group"),
                PrelabelRow("noev", "no_event", noev_path.name, 12, -1, -1, True, "train", "slovo", "sign_group"),
            ]
            ds = SyntheticContinuousDataset(
                prelabel_dir=root,
                rows=rows,
                cfg=SynthConfig(
                    seq_len=12,
                    min_signs=1,
                    max_signs=1,
                    continuous_mode_weight=0.0,
                    hard_negative_mode_weight=1.0,
                    stress_mode_weight=0.0,
                ),
                epoch_size=1,
                seed=13,
                min_sign_len=1,
            )
            sample = ds[0]
            self.assertEqual(sample["meta"]["generation_mode"], "hard_negative")

    def test_warmup_single_sign_profile_generates_exactly_one_sign(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sign_a = root / "user_a_clip_1.npz"
            sign_b = root / "user_a_clip_2.npz"
            noev = root / "user_a_noev.npz"
            sign_bio = np.asarray([0, 1, 2, 2, 0], dtype=np.uint8)
            _write_step1_npz(sign_a, pts_value=1.0, length=5, bio=sign_bio, start_idx=1, end_idx=3, is_no_event=False)
            _write_step1_npz(sign_b, pts_value=2.0, length=5, bio=sign_bio, start_idx=1, end_idx=3, is_no_event=False)
            _write_step1_npz(noev, pts_value=0.0, length=8, bio=np.zeros((8,), dtype=np.uint8), start_idx=-1, end_idx=-1, is_no_event=True)
            rows = [
                PrelabelRow("user_a_clip_1", "hello", sign_a.name, 5, 1, 3, False, "train", "slovo", "user_a"),
                PrelabelRow("user_a_clip_2", "bye", sign_b.name, 5, 1, 3, False, "train", "slovo", "user_a"),
                PrelabelRow("user_a_noev", "no_event", noev.name, 8, -1, -1, True, "train", "slovo", "user_a"),
            ]
            ds = SyntheticContinuousDataset(
                prelabel_dir=root,
                rows=rows,
                cfg=SynthConfig(
                    seq_len=16,
                    min_signs=1,
                    max_signs=3,
                    dataset_profile="warmup_single_sign",
                    transition_all_boundaries=True,
                    align_chunks=False,
                ),
                epoch_size=1,
                seed=31,
                min_sign_len=1,
            )
            sample = ds[0]
            self.assertEqual(sample["meta"]["generation_mode"], "warmup_single_sign")
            self.assertEqual(int(sample["meta"]["num_signs"]), 1)

    def test_main_profile_caps_sparse_signer_multi_sign_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            noev_path = root / "user_sparse_noev.npz"
            _write_step1_npz(noev_path, pts_value=0.0, length=8, bio=np.zeros((8,), dtype=np.uint8), start_idx=-1, end_idx=-1, is_no_event=True)
            rows = [
                PrelabelRow("user_sparse_noev", "no_event", noev_path.name, 8, -1, -1, True, "train", "slovo", "user_sparse"),
            ]
            sign_bio = np.asarray([0, 1, 2, 2, 0], dtype=np.uint8)
            for idx in range(3):
                path = root / f"user_sparse_clip_{idx}.npz"
                _write_step1_npz(path, pts_value=float(idx + 1), length=5, bio=sign_bio, start_idx=1, end_idx=3, is_no_event=False)
                rows.append(
                    PrelabelRow(path.stem, f"label_{idx}", path.name, 5, 1, 3, False, "train", "slovo", "user_sparse")
                )
            ds = SyntheticContinuousDataset(
                prelabel_dir=root,
                rows=rows,
                cfg=SynthConfig(
                    seq_len=24,
                    min_signs=2,
                    max_signs=3,
                    dataset_profile="main_continuous",
                    dense_signer_min_clips=8,
                    align_chunks=False,
                    transition_all_boundaries=False,
                ),
                epoch_size=1,
                seed=37,
                min_sign_len=1,
            )
            sample = ds[0]
            self.assertFalse(bool(sample["meta"]["sequence_is_dense_signer"]))
            self.assertLessEqual(int(sample["meta"]["num_signs"]), 2)

    def test_main_profile_avoids_external_cross_source_no_event_fillers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sign_path = root / "clip_a_sign.npz"
            sign_bio = np.asarray([1, 2, 2, 2], dtype=np.uint8)
            _write_step1_npz(sign_path, pts_value=5.0, length=4, bio=sign_bio, start_idx=0, end_idx=3, is_no_event=False)
            rows = [
                PrelabelRow("clip_a_sign", "hello", sign_path.name, 4, 0, 3, False, "train", "slovo", "clip_a"),
            ]
            (root / "index.json").write_text(
                json.dumps(
                    [
                        {
                            "vid": "clip_a_sign",
                            "label_str": "hello",
                            "path_to_npz": sign_path.name,
                            "T_total": 4,
                            "start_idx": 0,
                            "end_idx": 3,
                            "is_no_event": False,
                            "split": "train",
                            "dataset": "slovo",
                            "source_group": "clip_a",
                        }
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            extra_root = root / "extra"
            extra_root.mkdir(parents=True, exist_ok=True)
            extra_path = extra_root / "clip_b_noev.npz"
            _write_step1_npz(
                extra_path,
                pts_value=0.0,
                length=12,
                bio=np.zeros((12,), dtype=np.uint8),
                start_idx=-1,
                end_idx=-1,
                is_no_event=True,
            )
            (extra_root / "index.json").write_text(
                json.dumps(
                    [
                        {
                            "vid": "clip_b_noev",
                            "label_str": "no_event",
                            "path_to_npz": extra_path.name,
                            "T_total": 12,
                            "start_idx": -1,
                            "end_idx": -1,
                            "is_no_event": True,
                            "split": "train",
                            "dataset": "ipn_hand",
                            "source_group": "clip_b",
                        }
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            ds = SyntheticContinuousDataset(
                prelabel_dir=root,
                rows=rows,
                cfg=SynthConfig(
                    seq_len=12,
                    min_signs=1,
                    max_signs=1,
                    leading_noev_prob=1.0,
                    leading_noev_min=4,
                    leading_noev_max=4,
                    all_noev_prob=0.0,
                    gap_min=0,
                    gap_max=0,
                    dataset_profile="main_continuous",
                    continuous_mode_weight=1.0,
                    hard_negative_mode_weight=0.0,
                    stress_mode_weight=0.0,
                    transition_all_boundaries=False,
                    align_chunks=False,
                ),
                epoch_size=1,
                seed=19,
                min_sign_len=1,
                preferred_noev_prelabel_dirs=[],
                extra_noev_prelabel_dirs=[extra_root],
            )
            sample = ds[0]
            semantic = dict(sample["meta"]["seam_diagnostics"].get("semantic", {}) or {})
            self.assertEqual(int(semantic.get("cross_source_boundary_count", 0)), 0)
            noev_parts = [dict(part) for part in sample["meta"].get("parts", []) if str(part.get("type", "")) == "no_event"]
            self.assertTrue(noev_parts)
            self.assertTrue(all(str(part.get("source_type", "")) != "ipn_no_event" for part in noev_parts))

    def test_continuous_stats_extractor_reads_step1_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sign_path = root / "sign.npz"
            noev_path = root / "noev.npz"
            sign_bio = np.asarray([0, 0, 1, 2, 2, 2], dtype=np.uint8)
            _write_step1_npz(
                sign_path,
                pts_value=5.0,
                length=6,
                bio=sign_bio,
                start_idx=2,
                end_idx=5,
                is_no_event=False,
                raw_frame_values=np.asarray([0.0, 0.0, 1.0, 1.0, 1.1, 1.1], dtype=np.float32),
            )
            _write_step1_npz(noev_path, pts_value=0.0, length=8, bio=np.zeros((8,), dtype=np.uint8), start_idx=-1, end_idx=-1, is_no_event=True)
            with np.load(sign_path, allow_pickle=False) as z:
                pts_raw = z["pts_raw"].copy()
                mask = z["mask"].copy()
            mask[:2, :, :] = 0.0
            np.savez(
                sign_path,
                pts=np.broadcast_to(np.asarray([0, 0, 1, 1, 1.1, 1.1], dtype=np.float32).reshape(6, 1, 1), (6, 42, 3)).copy(),
                pts_raw=pts_raw,
                mask=mask,
                ts=np.arange(6, dtype=np.float32),
                bio=sign_bio.astype(np.uint8),
                label_str=np.asarray("hello"),
                is_no_event=np.asarray(False),
                start_idx=np.asarray(2, dtype=np.int64),
                end_idx=np.asarray(5, dtype=np.int64),
                meta=np.asarray(json.dumps({"dataset": "slovo", "split": "train", "source_group": "sign_group", "preprocessing_version": "canonical_hands42_v3", "coords": "image", "raw_pts_key": "pts_raw"})),
            )
            index_rows = [
                {"vid": "sign", "label_str": "hello", "path_to_npz": sign_path.name, "T_total": 6, "start_idx": 2, "end_idx": 5, "is_no_event": False, "split": "train", "dataset": "slovo", "source_group": "sign_group"},
                {"vid": "noev", "label_str": "no_event", "path_to_npz": noev_path.name, "T_total": 8, "start_idx": -1, "end_idx": -1, "is_no_event": True, "split": "train", "dataset": "slovo", "source_group": "noev_group"},
            ]
            (root / "index.json").write_text(json.dumps(index_rows, ensure_ascii=False, indent=2), encoding="utf-8")
            payload = build_continuous_stats(session_dirs=[], prelabel_dirs=[root], motion_epsilon=0.05)
            self.assertGreaterEqual(payload["source"]["num_sequences"], 2)
            self.assertIn(2, payload["leading_noev_lengths"])
            self.assertTrue(payload["gap_lengths"])

    def test_resolve_synth_workers_auto_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sign_path = root / "sign.npz"
            sign_bio = np.asarray([1, 2, 2, 2], dtype=np.uint8)
            _write_step1_npz(sign_path, pts_value=5.0, length=4, bio=sign_bio, start_idx=0, end_idx=3, is_no_event=False)
            noev_path = root / "noev.npz"
            _write_step1_npz(noev_path, pts_value=0.0, length=12, bio=np.zeros((12,), dtype=np.uint8), start_idx=-1, end_idx=-1, is_no_event=True)
            index_rows = [
                {
                    "vid": "sign",
                    "label_str": "hello",
                    "path_to_npz": sign_path.name,
                    "T_total": 4,
                    "start_idx": 0,
                    "end_idx": 3,
                    "is_no_event": False,
                    "split": "train",
                    "dataset": "slovo",
                    "source_group": "sign_group",
                },
                {
                    "vid": "noev",
                    "label_str": "no_event",
                    "path_to_npz": noev_path.name,
                    "T_total": 12,
                    "start_idx": -1,
                    "end_idx": -1,
                    "is_no_event": True,
                    "split": "train",
                    "dataset": "slovo",
                    "source_group": "noev_group",
                },
            ]
            (root / "index.json").write_text(json.dumps(index_rows, ensure_ascii=False, indent=2), encoding="utf-8")
            cfg = SynthConfig(
                seq_len=9,
                min_signs=1,
                max_signs=1,
                gap_min=0,
                gap_max=0,
                leading_noev_prob=1.0,
                leading_noev_min=5,
                leading_noev_max=5,
                all_noev_prob=0.0,
                pad_mode="end_no_event",
            )
            workers, info = _resolve_synth_workers(
                prelabel_dir=root,
                cfg=cfg,
                preferred_dirs=[],
                extra_dirs=[],
                requested_workers=1,
                auto_workers=True,
                auto_workers_max=2,
                auto_workers_rebench=True,
                auto_workers_probe_samples=4,
                shard_size=2,
                seed=17,
                min_sign_len=1,
            )
            self.assertGreaterEqual(int(workers), 1)
            self.assertTrue(bool(info.get("enabled", False)))
            self.assertIn("candidates", info)
            self.assertTrue(bool(info.get("candidates", [])))


if __name__ == "__main__":
    unittest.main()
