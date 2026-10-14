"""Module 1 - Preprocessing: uniform frame sampling, face detection/alignment, normalization."""
import cv2
import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_mtcnn = None


def sample_frames(video_path: str, num_frames: int = 16):
    """Uniformly sample `num_frames` RGB frames from a video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise IOError(f"Cannot read video: {video_path}")
    idxs = np.linspace(0, total - 1, num_frames).astype(int)
    frames, last = [], None
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            last = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if last is not None:            # reuse previous frame if a seek fails
            frames.append(last)
    cap.release()
    if not frames:
        raise IOError(f"No frames decoded: {video_path}")
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return frames


def _get_mtcnn(device):
    global _mtcnn
    if _mtcnn is None:
        try:
            from facenet_pytorch import MTCNN
            _mtcnn = MTCNN(keep_all=False, device=device, post_process=False)
        except ImportError:
            _mtcnn = False              # fall back to OpenCV Haar cascade
    return _mtcnn


def crop_face(frame: np.ndarray, size: int = 224, margin: float = 0.3, device="cpu"):
    """Detect the largest face and return a size x size crop (centre-crop if none found)."""
    h, w = frame.shape[:2]
    box = None
    mt = _get_mtcnn(device)
    if mt:
        boxes, _ = mt.detect(frame)
        if boxes is not None and len(boxes):
            box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    else:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
        if len(faces):
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            box = (x, y, x + fw, y + fh)
    if box is None:                     # no face: centre square crop
        s = min(h, w)
        x0, y0 = (w - s) // 2, (h - s) // 2
        box = (x0, y0, x0 + s, y0 + s)
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1) * (1 + margin)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    x1, x2 = int(max(0, cx - side / 2)), int(min(w, cx + side / 2))
    y1, y2 = int(max(0, cy - side / 2)), int(min(h, cy + side / 2))
    return cv2.resize(frame[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_AREA)


def to_tensor(faces):
    """List of HxWx3 uint8 faces -> T x 3 x H x W normalized tensor."""
    arr = (np.stack(faces).astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


def load_clip(video_path: str, num_frames: int = 16, size: int = 224, device="cpu"):
    """Full Module-1 pipeline. Returns (tensor T x 3 x H x W, list of uint8 face crops)."""
    frames = sample_frames(video_path, num_frames)
    faces = [crop_face(f, size, device=device) for f in frames]
    return to_tensor(faces), faces
