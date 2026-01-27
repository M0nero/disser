# CoreML Exporter

CLI to export an ST-GCN checkpoint to CoreML (.mlmodel).

## Requirements

- `coremltools` installed (already in `requirements.txt`)
- A trained checkpoint (e.g. `outputs/runs/.../best.ckpt` or `.pth`)

## Run (module)

```
python -m coreml.exporter --ckpt outputs/runs/best_model.pth --out outputs/coreml/STGCN.mlmodel
```

## Run (script wrapper)

```
python coreml/export_coreml.py --ckpt outputs/runs/best_model.pth --out outputs/coreml/STGCN.mlmodel
```

## Common options

```
python -m coreml.exporter \
  --ckpt outputs/runs/best_model.pth \
  --out outputs/coreml/STGCN.mlmodel \
  --labels outputs/artifacts/labels.txt \
  --fixed-t 64 \
  --prefer-script
```

## Flexible T (iOS16+)

```
python -m coreml.exporter \
  --ckpt outputs/runs/best_model.pth \
  --out outputs/coreml/STGCN.mlmodel \
  --min-t 32 --max-t 128
```

## ONNX path (fallback)

```
python -m coreml.exporter \
  --ckpt outputs/runs/best_model.pth \
  --out outputs/coreml/STGCN.mlmodel \
  --via-onnx --opset 13
```

## Notes

- If your training uses `label2idx.json`, export labels in the same order for correct class mapping.
- For smaller models, `--prefer-script` is usually faster to export.
