"""Module 6 - Evaluation: within-domain, cross-dataset, and compression robustness.

    python evaluate.py --ckpt runs/attn/best.pt --data_root /data/ffpp   --split_json runs/attn/split.json
    python evaluate.py --ckpt runs/attn/best.pt --data_root /data/celebdf                # cross-dataset (all videos)
    python evaluate.py --ckpt runs/attn/best.pt --data_root /data/ffpp --split_json runs/attn/split.json --crf 28 35
"""
import argparse, json, os, subprocess, tempfile, shutil
import numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from models import DeepfakeDetector
from dataset import scan, VideoDataset


@torch.no_grad()
def run_eval(model, loader, dev):
    model.eval(); probs, ys = [], []
    for x, y in loader:
        p = torch.softmax(model(x.to(dev)), -1)[:, 1].cpu()
        probs += p.tolist(); ys += y.tolist()
    probs, ys = np.array(probs), np.array(ys)
    auc = roc_auc_score(ys, probs) if len(set(ys)) > 1 else float("nan")
    return {"acc": accuracy_score(ys, probs > 0.5), "auc": auc, "n": len(ys)}


def recompress(items, crf, tmp):
    """Re-encode videos with H.264 at the given CRF (higher = heavier compression)."""
    out = []
    for i, (p, l, ident) in enumerate(items):
        q = os.path.join(tmp, f"{i}.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", p, "-c:v", "libx264", "-crf", str(crf), q], check=True)
        out.append((q, l, ident))
    return out


def load_model(ckpt, dev):
    ck = torch.load(ckpt, map_location=dev)
    m = DeepfakeDetector(not ck["baseline"], pretrained=False).to(dev)
    m.load_state_dict(ck["model"]); return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--data_root", required=True)
    ap.add_argument("--split_json", help="use the 'test' list from a training run (within-domain)")
    ap.add_argument("--crf", type=int, nargs="*", default=[]); ap.add_argument("--frames", type=int, default=16)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(a.ckpt, dev)
    items = scan(a.data_root)
    if a.split_json:
        keep = set(json.load(open(a.split_json))["test"]); items = [i for i in items if i[0] in keep]

    def go(its, tag):
        dl = DataLoader(VideoDataset(its, a.frames), 4, num_workers=2)
        m = run_eval(model, dl, dev); print(f"{tag:<22} acc {m['acc']:.3f}  auc {m['auc']:.3f}  (n={m['n']})")

    go(items, "clean")
    for crf in a.crf:
        tmp = tempfile.mkdtemp()
        try: go(recompress(items, crf, tmp), f"H.264 crf={crf}")
        finally: shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
