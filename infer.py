"""Inference (Algorithm step 8): video -> label, P(fake), attention overlays. No backward pass.

    python infer.py --ckpt runs/attn/best.pt --video clip.mp4 --out result.png
"""
import argparse
import numpy as np, torch, cv2
from preprocess import load_clip
from interpret import overlay


@torch.no_grad()
def predict(model, video_path, frames=16, dev="cpu"):
    model.eval()
    clip, faces = load_clip(video_path, frames, device=dev)
    logits, ex = model(clip.unsqueeze(0).to(dev), return_attention=True)
    p_fake = torch.softmax(logits, -1)[0, 1].item()
    res = {"label": "FAKE" if p_fake > 0.5 else "REAL", "p_fake": p_fake, "faces": faces}
    if "spatial" in ex:
        res["spatial"] = ex["spatial"][0].cpu().numpy()                    # T x h x w
        res["temporal"] = ex["temporal"][0].cpu().numpy()                  # T
        res["overlays"] = [overlay(f, m) for f, m in zip(faces, res["spatial"])]
    return res


def montage(res, n=6):
    """Contact sheet of the n most-attended frames with their attention overlays."""
    faces = res["faces"]
    order = np.argsort(-res["temporal"])[:n] if "temporal" in res else range(min(n, len(faces)))
    rows = []
    for i in sorted(order):
        top = res["overlays"][i] if "overlays" in res else faces[i]
        tile = np.concatenate([faces[i], top], axis=0).copy()
        txt = f"t={i}" + (f" w={res['temporal'][i]:.2f}" if "temporal" in res else "")
        cv2.putText(tile, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        rows.append(tile)
    return np.concatenate(rows, axis=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="result.png"); ap.add_argument("--frames", type=int, default=16)
    a = ap.parse_args()
    from evaluate import load_model
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    r = predict(load_model(a.ckpt, dev), a.video, a.frames, dev)
    print(f"{r['label']}  P(fake)={r['p_fake']:.3f}")
    cv2.imwrite(a.out, cv2.cvtColor(montage(r), cv2.COLOR_RGB2BGR)); print("saved", a.out)
