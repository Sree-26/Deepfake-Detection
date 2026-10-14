import os, tempfile
import pandas as pd, torch, streamlit as st
from evaluate import load_model
from infer import predict, montage

st.set_page_config(page_title="Deepfake Moderation Pre-Filter", page_icon="🛡️", layout="wide")
dev = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource(show_spinner="Loading model…")
def get_model(path):
    return load_model(path, dev)


with st.sidebar:
    st.header("Settings")
    ckpt = st.text_input("Checkpoint path", os.environ.get("DEEPFAKE_CKPT", "runs/attn/best.pt"))
    threshold = st.slider("Flag threshold  P(fake) ≥", 0.05, 0.95, 0.50, 0.05)
    frames = st.select_slider("Frames sampled per video", [8, 16, 24, 32], value=16)
    n_show = st.slider("Frames shown in evidence strip", 3, 10, 6)
    st.caption(f"Device: **{dev}**")

st.title("🛡️ Deepfake Video Moderation Pre-Filter")
st.caption("Attention-based CNN (ResNeXt-50 + spatial & temporal attention). Automated pre-filter ahead of human review.")

if not os.path.isfile(ckpt):
    st.error(f"Checkpoint not found: `{ckpt}`. Train one with `python train.py ...`.")
    st.stop()
model = get_model(ckpt)
is_baseline = not model.use_attention
if is_baseline:
    st.warning("Baseline checkpoint loaded: no attention maps available.")

up = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
if up is None:
    st.info("Upload a short video clip to analyse.")
    st.stop()

left, right = st.columns([1, 2])
with left:
    st.video(up)

with tempfile.NamedTemporaryFile(suffix=os.path.splitext(up.name)[1] or ".mp4", delete=False) as f:
    f.write(up.getbuffer()); path = f.name
try:
    with st.spinner("Analysing…"):
        r = predict(model, path, frames, dev)
except Exception as e:
    st.error(f"Could not process video: {e}")
    st.stop()
finally:
    os.unlink(path)

with right:
    flagged = r["p_fake"] >= threshold
    c1, c2, c3 = st.columns(3)
    c1.metric("Model verdict", r["label"])
    c2.metric("P(fake)", f"{r['p_fake']:.1%}")
    c3.metric("Moderation action", "FLAG" if flagged else "PASS")
    st.progress(float(r["p_fake"]))
    if flagged:
        st.error("Route to human review. Check the highlighted regions below.")
    else:
        st.success("Below threshold: no review required.")

if not is_baseline:
    st.subheader("Evidence: most-attended frames")
    st.image(montage(r, n_show), caption="Top: face crop. Bottom: spatial attention overlay. w = temporal weight.",
             width="stretch")
    st.subheader("Temporal attention across the clip")
    st.bar_chart(pd.DataFrame({"attention weight": r["temporal"]},
                              index=[f"t{i}" for i in range(len(r["temporal"]))]))
    with st.expander("All sampled frames"):
        cols = st.columns(4)
        for i, ov in enumerate(r["overlays"]):
            cols[i % 4].image(ov, caption=f"t={i}  w={r['temporal'][i]:.2f}", width="stretch")
