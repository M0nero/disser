from dataset import GestureDataset
from pathlib import Path

# подставь пути
JSON = "data/skeletons.json"
CSV  = "data/annotations.csv"

ds = GestureDataset(JSON, CSV, split="train")  # тот же split, что в обучении
# idx -> label в правильном порядке
idx2label = {i: lbl for lbl, i in ds.label2idx.items()}
labels = [idx2label[i] for i in range(len(idx2label))]
Path("labels.txt").write_text("\n".join(labels), encoding="utf-8")
print("Saved labels.txt with", len(labels), "labels")
