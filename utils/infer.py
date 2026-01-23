import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from msagcn.model import STGCN

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
log=logging.getLogger()

def load_label_map(path: Path):
    if not path or not path.exists():
        return None
    data = json.load(path.open('r', encoding='utf-8'))
    return {int(k): v for k, v in data.items()}

def prepare_input(frames, T: int, device):
    # enforce at least 1 time step
    T_eff = max(T, 1)
    x = torch.zeros((1,3,42,T_eff), dtype=torch.float, device=device)
    for t, frame in enumerate(frames[:T_eff]):
        h1 = frame.get('hand 1') or []
        h2 = frame.get('hand 2') or []
        if len(h1)!=21: h1=[{'x':0,'y':0,'z':0}]*21
        if len(h2)!=21: h2=[{'x':0,'y':0,'z':0}]*21
        for j, pt in enumerate(h1+h2):
            x[0,0,j,t] = float(pt.get('x',0.0))
            x[0,1,j,t] = float(pt.get('y',0.0))
            x[0,2,j,t] = float(pt.get('z',0.0))
    return x

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model-path',   type=Path, required=True)
    p.add_argument('--seq-json',     type=Path, required=True)
    p.add_argument('--num-classes',  type=int,  required=True)
    p.add_argument('--label-map',    type=Path,  default=None)
    p.add_argument('--use-cuda',     action='store_true')
    args = p.parse_args()

    device = torch.device('cuda') if args.use_cuda and torch.cuda.is_available() else torch.device('cpu')
    log.info(f"Using device: {device}")

    model = STGCN(args.num_classes).to(device)
    state = torch.load(str(args.model_path), map_location=device)
    model.load_state_dict(state)
    model.eval()

    label_map = load_label_map(args.label_map)
    seq = json.load(args.seq_json.open('r', encoding='utf-8'))
    if not isinstance(seq, list):
        log.error("Sequence JSON must be a list of frames")
        sys.exit(1)
    # pad/min length
    T = max(len(seq), 1)
    inp = prepare_input(seq, T, device)

    logits = model(inp)
    idx = logits.argmax(dim=1).item()
    conf = logits.softmax(dim=1)[0, idx].item()
    label = label_map.get(idx, idx) if label_map else idx
    log.info(f"Predicted: {label} (idx={idx}) conf={conf:.3f}")
