
"""Train ST-GCN gesture recognizer with optional tiny subset for debug/testing."""
import argparse
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score
from dataset import GestureDataset
from model import STGCN

# ----------------------------------------
# Logger & utils
# ----------------------------------------
def setup_logger(logdir: str) -> logging.Logger:
    os.makedirs(logdir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = Path(logdir) / f"train_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding='utf-8'),
        ]
    )
    return logging.getLogger(__name__)

def mem_report() -> str:
    import psutil
    ram = psutil.Process().memory_info().rss / 1024 ** 2
    gpu = torch.cuda.memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0
    return f"RAM={ram:.1f}MB GPU={gpu:.1f}MB"

# ----------------------------------------
# Training & evaluation loops
# ----------------------------------------
def train_epoch(model, loader, criterion, optimizer, scaler, device):
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type):
            logits = model(X)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * y.size(0)
        total_correct += (preds == y).sum().item()
        total_samples += y.size(0)
    return total_loss / total_samples, total_correct / total_samples

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(X)
            loss = criterion(logits, y)
            preds = logits.argmax(dim=1)

            total_loss += loss.item() * y.size(0)
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    f1 = f1_score(all_labels, all_preds, average='macro') if total_samples > 0 else 0.0
    return total_loss / total_samples, total_correct / total_samples, f1

# ----------------------------------------
# Argument parsing
# ----------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--json',     required=True, help='Path to skeletons JSON')
    p.add_argument('--csv',      required=True, help='Path to annotations CSV')
    p.add_argument('--batch',    type=int, default=8)
    p.add_argument('--epochs',   type=int, default=20)
    p.add_argument('--lr',       type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-4, help='Weight decay for optimizer')
    p.add_argument('--workers',  type=int, default=0)
    p.add_argument('--logdir',   default='logs')
    p.add_argument('--checkpoint', default='best_model.pth')
    p.add_argument('--compile',  action='store_true', help='Use torch.compile if available')
    p.add_argument('--debug',    action='store_true', help='Use tiny subset for quick debugging')
    p.add_argument('--subset-size', type=int, default=5, help='Number of samples per split when --debug')
    p.add_argument('--center', action='store_true', help='Center skeletons by wrist')
    p.add_argument('--normalize', action='store_true', help='Normalize skeleton scale')
    return p.parse_args()

# ----------------------------------------
# Main
# ----------------------------------------
def main():
    args = parse_args()
    log = setup_logger(args.logdir)
    log.info("Starting training")
    log.info(mem_report())

    # Device
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_type)
    log.info(f"Using device: {device}")

    # Datasets
    train_ds = GestureDataset(args.json, args.csv, split='train', center=args.center, normalize=args.normalize)
    val_ds   = GestureDataset(args.json, args.csv, split='val', center=args.center, normalize=args.normalize)
    log.info(f"Full dataset: train={len(train_ds)} val={len(val_ds)} samples")

    # Optionally wrap in tiny Subset for debug
    if args.debug:
        random.seed(42)
        # берём N случайных индексов из полного train_ds
        full_idxs = list(range(len(train_ds)))
        subset_idxs = random.sample(full_idxs, k=args.subset_size)

        # и используем их и для обучения, и для валидации
        train_ds = Subset(train_ds, subset_idxs)
        val_ds   = train_ds

        log.info(f"DEBUG: using same {args.subset_size} samples for train & val")

    # DataLoaders
    dl_kwargs = dict(
        batch_size=args.batch,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=train_ds.dataset.collate_fn if isinstance(train_ds, Subset) else train_ds.collate_fn
    )
    train_loader = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **dl_kwargs)

    # Model, loss, optimizer, scaler
    num_classes = len(train_ds.dataset.label2idx if isinstance(train_ds, Subset) else train_ds.label2idx)
    model = STGCN(num_classes=num_classes, temp_k=9, drop=0.5, use_data_bn=True).to(device)
    if args.compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            log.info("Model compiled with torch.compile")
        except Exception as e:
            log.warning(f"torch.compile failed: {e}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler    = GradScaler(device=device_type)

    # Training loop
    best_f1 = 0.0
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        v_loss, v_acc, v_f1 = eval_epoch(model, val_loader, criterion, device)

        scheduler.step()

        log.info(
            f"Ep {epoch:02d}/{args.epochs} | "
            f"TrL={tr_loss:.4f} TrA={tr_acc:.3f} | "
            f"ValL={v_loss:.4f} ValA={v_acc:.3f} ValF1={v_f1:.3f} | "
            f"{mem_report()} | {time.time()-t0:.1f}s"
        )

        if v_f1 > best_f1:
            best_f1 = v_f1
            torch.save(model.state_dict(), args.checkpoint)
            log.info(f"🔖 Saved new best model (F1={best_f1:.3f}) to {args.checkpoint}")
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= 10:
            log.info("⏹️ Early stopping: no improvement in 10 epochs")
            break

    log.info("Training complete")
    log.info(mem_report())

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Interrupted by user — exiting")
