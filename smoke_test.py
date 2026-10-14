"""End-to-end check with NO data and NO downloads: shapes, gradients, training step, IoU, and a synthetic video."""
import numpy as np, torch, cv2, tempfile, os
from models import DeepfakeDetector
from interpret import attention_iou, overlay
from preprocess import load_clip

torch.manual_seed(0)
x = torch.randn(2, 4, 3, 224, 224); y = torch.tensor([0, 1])

for use_attn in (False, True):
    m = DeepfakeDetector(use_attention=use_attn, pretrained=False)
    logits, ex = m(x, return_attention=True)
    assert logits.shape == (2, 2)
    loss = torch.nn.functional.cross_entropy(logits, y); loss.backward()
    n_params = sum(p.numel() for p in m.parameters())
    print(f"{'attention' if use_attn else 'baseline ':<9} logits {tuple(logits.shape)}  params {n_params/1e6:.1f}M  loss {loss.item():.3f}", end="")
    if use_attn:
        assert ex["spatial"].shape == (2, 4, 7, 7) and ex["temporal"].shape == (2, 4)
        assert torch.allclose(ex["temporal"].sum(1), torch.ones(2), atol=1e-5)
        assert m.spatial.gamma.grad is not None
        print(f"  spatial map {tuple(ex['spatial'].shape)}  temporal weights sum to 1", end="")
    print()

# IoU sanity: attention concentrated exactly on the GT region -> high IoU; elsewhere -> ~0
gt = np.zeros((224, 224), np.uint8); gt[60:160, 60:160] = 1
good = np.zeros((7, 7)); good[2:5, 2:5] = 1; bad = np.zeros((7, 7)); bad[0, 0] = 1
print(f"IoU aligned {attention_iou(good, gt):.2f}   IoU misaligned {attention_iou(bad, gt):.2f}")
assert attention_iou(good, gt) > 0.6 > attention_iou(bad, gt)

# Preprocessing on a synthetic video (falls back to centre-crop: no face present)
p = os.path.join(tempfile.mkdtemp(), "t.mp4")
w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
for i in range(30): w.write(np.full((240, 320, 3), i * 8 % 255, np.uint8))
w.release()
clip, faces = load_clip(p, 8)
assert clip.shape == (8, 3, 224, 224)
print("load_clip ->", tuple(clip.shape), "OK")
print("ALL CHECKS PASSED")
