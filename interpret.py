"""Attention-map visualisation and IoU interpretability score (Algorithm step 9).

IoU = |A_bin ∩ G| / |A_bin ∪ G|, with A_bin = attention map thresholded and G = annotated forgery mask.
"""
import cv2, numpy as np


def upsample_map(attn_map: np.ndarray, size: int) -> np.ndarray:
    m = cv2.resize(attn_map.astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
    m = m - m.min(); return m / (m.max() + 1e-8)


def overlay(face_rgb: np.ndarray, attn_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = cv2.applyColorMap((upsample_map(attn_map, face_rgb.shape[0]) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(face_rgb, 1 - alpha, heat, alpha, 0)


def attention_iou(attn_map: np.ndarray, gt_mask: np.ndarray, thresh: float = 0.5) -> float:
    """gt_mask: HxW binary forgery region. Returns IoU in [0, 1]."""
    a = upsample_map(attn_map, gt_mask.shape[0]) >= thresh
    g = gt_mask.astype(bool)
    union = (a | g).sum()
    return float((a & g).sum() / union) if union else 0.0
