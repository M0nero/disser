"""train.py — ST‑GCN trainer (v6, GPU‑prefetch)
════════════════════════════════════════════
Сохранил всё из исходного скрипта *и* добавил асинхронную подгрузку
батчей прямо в память GPU, чтобы график util не «пилой».

Новые фичи
──────────
1. **PrefetchLoader**
   – отдельный CUDA‑stream копирует следующий батч на GPU, пока текущий
     считается → пропуски в загрузке ≈ 0 %.
2. **Dynamic num_workers**
   – по умолчанию `min(8, os.cpu_count())`, CLI `--workers` остаётся.
3. **CLI**
   – `--no_prefetch` (если нужно отключить prefetch на старых драйверах).

Остальное без изменений: label‑smoothing, WeightedSampler, AMP, GradAccum,
ранний стоп, логгер.

Запуск
──────
```bash
python train.py \
  --json skeletons.json --csv ann.csv \
  --center --normalize --augment \
  --batch 64 --epochs 80 --workers 8
```
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import Tuple, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import GestureDataset
from model import STGCN

# -----------------------------------------------------------------------------
# Prefetch wrapper
# -----------------------------------------------------------------------------

class PrefetchLoader:
    """Wrap a DataLoader to pre‑copy batches to GPU asynchronously."""

    def __init__(self, loader: DataLoader, device: torch.device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream() if device.type == "cuda" else None

    def __iter__(self) -> "PrefetchLoader":
        self.iter: Iterator = iter(self.loader)
        if self.stream is not None:
            self._preload()
        return self

    def __next__(self):
        if self.stream is None:  # CPU fallback
            return next(self.iter)

        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        if batch is None:
            raise StopIteration
        X, y = batch
        X.record_stream(torch.cuda.current_stream())
        y.record_stream(torch.cuda.current_stream())
        self._preload()
        return X, y

    def _preload(self):
        try:
            X, y = next(self.iter)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            X = X.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            self.next_batch = (X, y)

    def __len__(self):
        return len(self.loader)

# -----------------------------------------------------------------------------
# Logger helper
# -----------------------------------------------------------------------------

def setup_logger(logdir: str) -> logging.Logger:
    os.makedirs(logdir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(logdir) / f"train_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    return logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Training & evaluation loops
# -----------------------------------------------------------------------------

def train_epoch(model: nn.Module, loader: PrefetchLoader | DataLoader, criterion, optimizer, scaler, device, accum: int):
    device_type = device.type
    model.train()

    total_loss = total_correct = total_samples = 0.0

    optimizer.zero_grad(set_to_none=True)
    for step, (X, y) in enumerate(loader, start=1):
        with autocast(device_type):
            logits = model(X)
            loss = criterion(logits, y) / accum  # scale loss for grad‑accum
        scaler.scale(loss).backward()

        if step % accum == 0 or step == len(loader):  # update weights
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * accum * y.size(0)  # de‑scaled
        total_correct += (preds == y).sum().item()
        total_samples += y.size(0)

    return total_loss / total_samples, total_correct / total_samples


def eval_epoch(model: nn.Module, loader: PrefetchLoader | DataLoader, criterion, device):
    model.eval()
    total_loss = total_correct = total_samples = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, y in loader:
            logits = model(X.to(device, non_blocking=True)) if isinstance(loader, DataLoader) else model(X)
            y = y.to(device, non_blocking=True) if isinstance(loader, DataLoader) else y
            loss = criterion(logits, y)

            preds = logits.argmax(dim=1)
            total_loss += loss.item() * y.size(0)
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    f1 = f1_score(all_labels, all_preds, average="macro") if total_samples else 0.0
    return total_loss / total_samples, total_correct / total_samples, f1

# -----------------------------------------------------------------------------
# Arg‑parser
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train ST‑GCN gesture recognizer (GPU‑prefetch)")
    p.add_argument("--json", required=True, help="Path to skeletons JSON")
    p.add_argument("--csv", required=True, help="Path to annotations CSV")

    p.add_argument("--batch", type=int, default=64, help="Batch size per update step")
    p.add_argument("--accum", type=int, default=1, help="Gradient accumulation steps")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=5e-4)

    default_workers = min(8, os.cpu_count() or 8)
    p.add_argument("--workers", type=int, default=default_workers, help="DataLoader workers")
    p.add_argument("--prefetch", type=int, default=4, help="prefetch_factor for DataLoader")
    p.add_argument("--no_prefetch", action="store_true", help="Disable GPU prefetch wrapper")

    p.add_argument("--smooth", type=float, default=0.05, help="Label smoothing value")
    p.add_argument("--center", action="store_true", help="Center skeletons by wrist")
    p.add_argument("--normalize", action="store_true", help="Normalize skeleton scale")
    p.add_argument("--augment", action="store_true", help="Enable on‑the‑fly spatial augs (dataset)")

    p.add_argument("--logdir", default="logs")
    p.add_argument("--checkpoint", default="best_model.pth", help="Path to save best model")
    p.add_argument("--patience", type=int, default=15, help="Early‑stop patience (epochs)")
    return p.parse_args()

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def build_sampler_and_class_weights(ds: GestureDataset) -> Tuple[WeightedRandomSampler, torch.Tensor]:
    labels = [lbl for _, lbl, *_ in ds.samples]
    counts = np.bincount(labels)
    n_classes = len(counts)

    # sample weights = 1 / freq(label)
    sample_weights = [1.0 / counts[lbl] for lbl in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # class weights для loss: normalized inverse frequency
    class_weights = torch.tensor(1.0 / counts, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * n_classes
    return sampler, class_weights

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    log = setup_logger(args.logdir)
    log.info("Starting training (GPU‑prefetch)")

    # Device & torch.backends tweaks
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    log.info(f"Using device: {device}")

    if device_type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Datasets
    train_ds = GestureDataset(
        args.json, args.csv, split="train",
        center=args.center, normalize=args.normalize, augment=args.augment,
    )
    val_ds = GestureDataset(
        args.json, args.csv, split="val",
        center=args.center, normalize=args.normalize, augment=False,
    )
    log.info(f"Dataset sizes → train: {len(train_ds)} | val: {len(val_ds)}")

    # Sampler + class weights
    sampler, class_weights = build_sampler_and_class_weights(train_ds)

    # DataLoader kwargs
    dl_kwargs = dict(
        batch_size=args.batch,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=(device_type == "cuda"),
        collate_fn=train_ds.collate_fn,
        prefetch_factor=args.prefetch,
    )

    train_loader = DataLoader(train_ds, sampler=sampler, **dl_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **dl_kwargs)

    # GPU prefetch wrapper
    if device_type == "cuda" and not args.no_prefetch:
        train_loader = PrefetchLoader(train_loader, device)
        val_loader = PrefetchLoader(val_loader, device)

    # Model
    num_classes = len(train_ds.label2idx)
    model = STGCN(num_classes=num_classes).to(device)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=args.smooth,
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()

    # Training loop
    best_f1 = 0.0
    no_improve = 0
    for epoch in range(1, args.epochs + 1):
        start_t = time.time()

        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device, args.accum)
        val_loss, val_acc, val_f1 = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        log.info(
            f"Ep {epoch:02d}/{args.epochs} | "
            f"TrL={tr_loss:.4f} TrA={tr_acc:.3f} | "
            f"ValL={val_loss:.4f} ValA={val_acc:.3f} ValF1={val_f1:.3f} | "
            f"{time.time() - start_t:.1f}s"
        )

        # Early‑stopping & checkpoint
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), args.checkpoint)
            log.info(f"🔖 New best model (F1={best_f1:.3f}) saved → {args.checkpoint}")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                log.info("⏹️ Early stop: no improvement for %d epochs", args.patience)
                break

    log.info(f"Training complete. Best macro‑F1 = {best_f1:.3f}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Interrupted by user — exiting…")
