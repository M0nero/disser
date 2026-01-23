from msagcn.dataset import GestureDataset
from pathlib import Path

# подставь пути
JSON = "datasets/data/skeletons.json"
CSV  = "datasets/data/annotations.csv"

ds = GestureDataset(JSON, CSV, split="train")  # тот же split, что в обучении
# idx -> label в правильном порядке
idx2label = {i: lbl for lbl, i in ds.label2idx.items()}
labels = [idx2label[i] for i in range(len(idx2label))]
out_path = Path("outputs/artifacts/labels.txt")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text("\n".join(labels), encoding="utf-8")
print("Saved labels.txt with", len(labels), "labels")
