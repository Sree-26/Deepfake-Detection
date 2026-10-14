"""Moderator demo UI.  pip install gradio ; python app.py --ckpt runs/attn/best.pt"""
import argparse, torch
import gradio as gr
from evaluate import load_model
from infer import predict, montage

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True)
ap.add_argument("--threshold", type=float, default=0.5)
args = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(args.ckpt, dev)


def moderate(video):
    r = predict(model, video, 16, dev)
    action = ("FLAG for human review" if r["p_fake"] >= args.threshold else "PASS")
    summary = f"**{r['label']}**  |  P(fake) = {r['p_fake']:.1%}  |  Action: {action}"
    return summary, montage(r)


gr.Interface(moderate, gr.Video(label="Upload video"),
             [gr.Markdown(label="Verdict"), gr.Image(label="Most-attended frames: face (top) / attention overlay (bottom)")],
             title="Deepfake Moderation Pre-Filter",
             description="Attention-based CNN (ResNeXt-50 + spatial & temporal attention). "
                         "Automated pre-filter ahead of human review.").launch()
