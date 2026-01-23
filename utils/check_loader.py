#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_loader.py — комплексная проверка MultiStreamGestureDataset

Запуск (PowerShell / VS Code):
.\venv\Scripts\python.exe .\check_loader.py `
  --json .\datasets\skeletons_balanced `
  --csv .\datasets\data\annotations.csv `
  --batch 32 `
  --max-frames 96 `
  --streams joints,bones,velocity `
  --include-pose `
  --connect-cross-edges `
  --center `
  --normalize `
  --thr-tune-steps 6 `
  --thr-tune-step 0.05
"""
import argparse
import os
import sys
import time
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader

# гарантируем доступ к локальному модулю
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (HERE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from msagcn.dataset_multistream import DSConfig, MultiStreamGestureDataset  # noqa: E402


def pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def check_dataset(ds: MultiStreamGestureDataset, name: str, batch_size: int, num_samples: int = 8):
    print(f"\n=== DATASET '{name}' ===")
    print(repr(ds))
    assert len(ds) > 0, "Empty dataset"

    # базовые атрибуты
    V = int(ds.V)
    P = int(getattr(ds, "P", 0))
    n_classes = len(ds.label2idx)
    print(f"V={V}, P={P}, classes={n_classes}, samples={len(ds)}")

    # no_event фильтр
    assert "no_event" not in {k.lower() for k in ds.label2idx.keys()}, "no_event попал в label2idx (должен быть выкинут на этапе CSV)"

    # _mirror_idx
    mir = getattr(ds, "_mirror_idx", None)
    assert mir is not None and len(mir) == V and len(torch.unique(mir)) == V, "_mirror_idx должен быть перестановкой [0..V-1]"
    assert torch.equal(mir[mir], torch.arange(V)), "mirror(mirror(i)) != i"

    # adjacency (sym)
    A = ds.build_adjacency(normalize="sym")
    assert A.shape == (V, V), "Adjacency имеет неверную форму"
    sym_err = torch.abs(A - A.T).max().item()
    print(f"A(sym): max|A-A^T|={sym_err:.3e}, row_sums≈{A.sum(1)[:3].tolist()}")

    # какие потоки ожидаем
    wanted = set(k for k in ("joints", "bones", "velocity") if k in ds.cfg.use_streams)

    # одиночные проверки
    idxs = random.sample(range(len(ds)), k=min(num_samples, len(ds)))
    for i in idxs:
        sample = ds[i]
        meta = sample["meta"]
        assert meta["V"] == V
        present = set(sample.keys()) & {"joints", "bones", "velocity"}
        assert wanted.issubset(present), f"Не все выбранные потоки в sample: {wanted} vs {present}"
        # формы
        T = ds.cfg.max_frames
        C = 3
        if "joints" in present:
            assert sample["joints"].shape == (C, V, T), "joints: ожидаем (3,V,T)"
        if "bones" in present:
            assert sample["bones"].shape == (C, V, T), "bones: ожидаем (3,V,T)"
        if "velocity" in present:
            assert sample["velocity"].shape == (C, V, T), "velocity: ожидаем (3,V,T)"
        assert sample["mask"].shape == (1, V, T), "mask: ожидаем (1,V,T)"
        assert sample["label"].dtype == torch.long

        # типы/NaN
        for k in present | {"mask"}:
            t = sample[k]
            if k != "label":
                assert t.dtype == torch.float32, f"{k} должен быть float32"
                assert torch.isfinite(t).all(), f"{k} содержит NaN/Inf"

        # бинарность mask + ковередж
        maskv = sample["mask"].unique()
        assert set(maskv.cpu().tolist()).issubset({0.0, 1.0}), f"mask не бинарный: {maskv}"
        cov = float(sample["mask"].float().mean())
        tgt = float(meta.get("target_ratio", meta.get("ratio_used", 0.0)))
        ach = float(meta.get("achieved_ratio", cov))
        print(f"sample[{i}] | coverage={pct(cov)} | label={int(sample['label'])} | thr={meta['thr_used']:.2f} target={tgt:.2f} achieved={ach:.2f}")
        if ach < 0.5:
            print(f"  WARN: low achieved coverage {ach:.2f} (<0.50)")

        # velocity: нулевой первый кадр
        if "velocity" in present:
            assert torch.allclose(sample["velocity"][..., 0], torch.zeros_like(sample["velocity"][..., 0])), "velocity на 0‑м кадре должен быть нулём"

        # bones согласованность (только там где есть родитель и валидные точки)
        if "bones" in present and "joints" in present:
            child_idx = getattr(ds, "_child_idx")
            par_idx = getattr(ds, "_par_idx")
            m = sample["mask"].squeeze(0)  # (V,T)
            m_child = m[child_idx]  # (Nc,T)
            m_par = m[par_idx]      # (Nc,T)
            valid = (m_child == 1) & (m_par == 1)
            if valid.any():
                j = sample["joints"]
                diff = j[:, child_idx, :] - j[:, par_idx, :]
                b = sample["bones"][:, child_idx, :]
                err = torch.where(valid.unsqueeze(0), torch.abs(b - diff), torch.zeros_like(b))
                mae = err.sum() / valid.sum()
                assert mae.item() < 1e-4, f"bones != joints(child)-joints(parent), MAE={mae.item():.2e}"

    # краткая сводка покрытия
    achs = []
    for _ in range(min(64, len(ds))):
        s = ds[random.randrange(len(ds))]
        achs.append(float(s["mask"].float().mean()))
    if achs:
        import numpy as np
        print(f"{name} coverage: mean={np.mean(achs):.3f} | p10={np.percentile(achs,10):.3f} | p50={np.percentile(achs,50):.3f} | p90={np.percentile(achs,90):.3f}")

    # DataLoader + collate_fn
    def _collate(batch):
        return MultiStreamGestureDataset.collate_fn(batch)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=_collate, drop_last=False)
    xb, yb, metas = next(iter(loader))
    # ключи
    for k in wanted | {"mask"}:
        assert k in xb, f"collate: отсутствует ключ {k}"
        assert xb[k].shape[0] == min(batch_size, len(ds)), f"collate: первый размер != batch"
    assert yb.shape[0] == min(batch_size, len(ds))
    assert isinstance(metas, list) and len(metas) == min(batch_size, len(ds))

    # простая скорость
    t0 = time.time()
    for _ in range(min(2, len(loader))):
        _ = next(iter(loader))
    t1 = time.time()
    print(f"loader ok | ~{(t1-t0):.3f}s на два итерации (num_workers=0)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="путь к combined.json или директории с *.json")
    ap.add_argument("--csv", required=True, help="annotations.csv")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-frames", type=int, default=64)
    ap.add_argument("--streams", default="joints,bones,velocity")
    ap.add_argument("--include-pose", action="store_true")
    ap.add_argument("--connect-cross-edges", action="store_true")
    ap.add_argument("--center", action="store_true")
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--thr-tune-steps", type=int, default=6)
    ap.add_argument("--thr-tune-step", type=float, default=0.05)
    args = ap.parse_args()

    use_streams = tuple(s.strip() for s in args.streams.split(",") if s.strip())

    cfg = DSConfig(
        max_frames=args.max_frames,
        use_streams=use_streams,
        include_pose=args.include_pose,
        connect_cross_edges=args.connect_cross_edges,
        center=args.center,
        normalize=args.normalize,
        augment=args.augment,
    )

    # тюнинг порога
    if hasattr(cfg, "thr_tune_steps"):
        cfg.thr_tune_steps = args.thr_tune_steps
    if hasattr(cfg, "thr_tune_step"):
        cfg.thr_tune_step = args.thr_tune_step

    # сначала train, затем val с тем же label2idx
    print("Инициализация train…")
    ds_train = MultiStreamGestureDataset(args.json, args.csv, split="train", cfg=cfg, label2idx=None)
    check_dataset(ds_train, "train", args.batch)

    try:
        print("\nИнициализация val…")
        ds_val = MultiStreamGestureDataset(args.json, args.csv, split="val", cfg=cfg, label2idx=ds_train.label2idx)
        check_dataset(ds_val, "val", args.batch)
    except Exception as e:
        print(f"val пропущен: {e}")

    # детерминизм при augment=False
    if not cfg.augment:
        i = random.randrange(len(ds_train))
        a = ds_train[i]
        b = ds_train[i]
        same = True
        for k in set(a.keys()) & {"joints", "bones", "velocity", "mask"}:
            same = same and torch.equal(a[k], b[k])
        print(f"\nDeterminism (augment=False) on sample[{i}]: {'OK' if same else 'MISMATCH'}")

    print("\nГотово.")


if __name__ == "__main__":
    main()
