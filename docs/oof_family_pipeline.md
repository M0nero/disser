# OOF Family Pipeline

This pipeline builds train-only class families for auxiliary supervision without using validation or test data.

## Stage 1: Build Strict OOF Cache

Run `tools/build_oof_cache.py` on the original training split only.

- The script derives `K` stratified folds from the original `train` subset.
- For each fold it fine-tunes from the provided checkpoint on `K-1` folds.
- It does **not** use holdout-fold labels for early stopping or model selection.
- It then runs inference on the held-out fold and saves:
  - `fold_assignments.csv`
  - `oof_predictions.csv`
  - `oof_features.npy`
  - `oof_kinematics.npy`
  - optional `oof_logits.npy`
  - `oof_meta.json`

Example:

```bash
python tools/build_oof_cache.py \
  --json datasets/skeletons \
  --csv datasets/data/annotations.csv \
  --ckpt outputs/runs/agcn_best/best.ckpt \
  --out outputs/families/oof_cache \
  --folds 5 \
  --fold_epochs 6
```

## Stage 2: Build Family Map

Run `tools/build_family_map.py` on the OOF cache.

- It computes class-level embedding prototypes.
- It computes symmetric OOF confusion similarity.
- It computes class-level kinematic prototypes.
- It combines them with deterministic agglomerative clustering.

Outputs:

- `family_map.json`
- `family_stats.json`
- `class_neighbors.csv`
- `family_diagnostics.md`

Example:

```bash
python tools/build_family_map.py \
  --oof_dir outputs/families/oof_cache \
  --out outputs/families/family_map.json \
  --num_families 128
```

Or let the builder choose `num_families` automatically from the train-only OOF cache:

```bash
python tools/build_family_map.py \
  --oof_dir outputs/families/oof_cache \
  --out outputs/families/family_map.json \
  --auto_num_families
```

## Stage 3: Fine-Tune With Auxiliary Family Supervision

Use the family map in the normal training pipeline:

```bash
python -m msagcn.train \
  --json datasets/skeletons \
  --csv datasets/data/annotations.csv \
  --out outputs/runs/agcn_family \
  --resume outputs/runs/agcn_best/best.ckpt \
  --resume_model_only \
  --use_family_head \
  --family_map outputs/families/family_map.json \
  --family_loss_weight 0.25
```

## Optional Stage 4: Head-Only Rebalance

Add a short classifier-only stage after the main run:

```bash
python -m msagcn.train ... \
  --use_family_head \
  --family_map outputs/families/family_map.json \
  --head_only_rebalance_epochs 5 \
  --auto_head_only_rebalance_stop \
  --head_only_rebalance_lr 1e-4 \
  --head_only_rebalance_use_logit_adjustment
```

This stage freezes the backbone and trains only the class head. It saves separate artifacts:

- `best_rebalance.ckpt`
- `last_rebalance.ckpt`
- `history_rebalance.json`
- `analysis/head_only_rebalance_summary.json`
