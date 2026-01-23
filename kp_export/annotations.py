
from __future__ import annotations
from pathlib import Path
from typing import List, Set

def _detect_delim(header_line: str) -> str:
    cands = ['\t', ';', ',', '|']
    counts = {d: header_line.count(d) for d in cands}
    return max(counts, key=counts.get) if any(counts.values()) else ','

def load_skip_ids(csv_path: str, skip_labels: List[str]) -> Set[str]:
    skip: Set[str] = set()
    if not csv_path:
        return skip
    try:
        import csv
        with open(csv_path, "r", encoding="utf-8") as f:
            sample = f.readline()
            delim = _detect_delim(sample)
            f.seek(0)
            reader = csv.DictReader(f, delimiter=delim)
            cols = [c.lower() for c in (reader.fieldnames or [])]
            id_candidates = ["attachment_id", "id", "video", "vid", "filename", "file", "name"]
            label_candidates = ["label", "text", "class", "tag"]
            id_col = next((c for c in id_candidates if c in cols), None)
            lab_col = next((c for c in label_candidates if c in cols), None)
            if not id_col or not lab_col:
                return skip
            for row in reader:
                try:
                    lbl = str(row[lab_col]).strip().lower()
                    if lbl in skip_labels:
                        vid_raw = str(row[id_col]).strip()
                        vid = Path(vid_raw).stem
                        if vid:
                            skip.add(vid)
                except Exception:
                    continue
    except FileNotFoundError:
        pass
    return skip
