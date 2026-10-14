"""Train the attention model or the matched baseline.

    python train.py --data_root /data/ffpp --out runs/attn            # attention model
    python train.py --data_root /data/ffpp --out runs/base --baseline # baseline (no attention)
    python train.py --synthetic --epochs 1 --out runs/smoke           # no-data smoke test
"""
import argparse, os, json, time
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from models import DeepfakeDetector
from dataset import scan, identity_split, VideoDataset, SyntheticDataset
from evaluate import run_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root"); ap.add_argument("--out", default="runs/exp")
    ap.add_argument("--baseline", action="store_true", help="disable attention (matched baseline)")
    ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--frames", type=int, default=16); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--freeze_epochs", type=int, default=1, help="train only the new head first")
    ap.add_argument("--no_pretrained", action="store_true"); ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    if a.synthetic:
        tr, va = SyntheticDataset(8, 4), SyntheticDataset(4, 4)
    else:
        sp = identity_split(scan(a.data_root))
        json.dump({k: [x[0] for x in v] for k, v in sp.items()}, open(f"{a.out}/split.json", "w"), indent=1)
        tr = VideoDataset(sp["train"], a.frames, train=True); va = VideoDataset(sp["val"], a.frames)
    tl = DataLoader(tr, a.batch_size, shuffle=True, num_workers=0 if a.synthetic else a.workers, drop_last=True)
    vl = DataLoader(va, a.batch_size, num_workers=0 if a.synthetic else a.workers)

    model = DeepfakeDetector(not a.baseline, pretrained=not (a.no_pretrained or a.synthetic)).to(dev)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=(dev == "cuda"))
    best = -1
    for ep in range(a.epochs):
        model.freeze_backbone(ep < a.freeze_epochs)
        model.train(); t0 = time.time(); tot = 0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(dev, enabled=(dev == "cuda")):
                loss = crit(model(x), y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item()
        sched.step()
        m = run_eval(model, vl, dev)
        print(f"epoch {ep+1}/{a.epochs} loss {tot/max(1,len(tl)):.4f} val_acc {m['acc']:.3f} "
              f"val_auc {m['auc']:.3f} ({time.time()-t0:.0f}s)")
        if m["auc"] >= best:
            best = m["auc"]; torch.save({"model": model.state_dict(), "baseline": a.baseline}, f"{a.out}/best.pt")
    print("best val AUC:", best)


if __name__ == "__main__":
    main()
