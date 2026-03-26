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
    _build_source_groups,
    _pick_pool_segment,
    load_prelabel_index,
    split_pools,
)


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
) -> None:
    if frame_values is None:
        pts = np.full((length, 42, 3), pts_value, dtype=np.float32)
    else:
        vals = np.asarray(frame_values, dtype=np.float32).reshape(length, 1, 1)
        pts = np.broadcast_to(vals, (length, 42, 3)).copy()
    mask = np.ones((length, 42, 1), dtype=np.float32)
    ts = np.arange(length, dtype=np.float32)
    meta = {
        "dataset": "slovo",
        "split": "train",
        "source_group": path.stem.split("__", 1)[0],
    }
    np.savez(
        path,
        pts=pts,
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


if __name__ == "__main__":
    unittest.main()
