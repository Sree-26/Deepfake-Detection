"""Identity-based dataset splitting + video dataset.

Expected layout (works for FaceForensics++ and Celeb-DF after you arrange folders):

    data_root/
        real/<identity_id>/<video>.mp4      # e.g. original_sequences/youtube/c23/...
        fake/<identity_id>/<video>.mp4      # any manipulation; identity = source/target person id

Splitting is done per *identity*, so no person appears in more than one split
(prevents the leakage / inflated accuracy noted in the deck).
"""
import os, random, glob
import torch
from torch.utils.data import Dataset
from preprocess import load_clip

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv")


def scan(root):
    items = []                                   # (path, label, identity)
    for label, name in ((0, "real"), (1, "fake")):
        for p in glob.glob(os.path.join(root, name, "**", "*"), recursive=True):
            if p.lower().endswith(VIDEO_EXT):
                rel = os.path.relpath(p, os.path.join(root, name))
                identity = rel.split(os.sep)[0] if os.sep in rel else os.path.splitext(rel)[0].split("_")[0]
                items.append((p, label, identity))
    return items


def identity_split(items, val=0.15, test=0.15, seed=42):
    ids = sorted({i[2] for i in items})
    random.Random(seed).shuffle(ids)
    n_val, n_test = int(len(ids) * val), int(len(ids) * test)
    test_ids = set(ids[:n_test]); val_ids = set(ids[n_test:n_test + n_val])
    split = {"train": [], "val": [], "test": []}
    for it in items:
        key = "test" if it[2] in test_ids else "val" if it[2] in val_ids else "train"
        split[key].append(it)
    # sanity: no identity overlap
    sets = {k: {i[2] for i in v} for k, v in split.items()}
    assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"])
    return split


class VideoDataset(Dataset):
    def __init__(self, items, num_frames=16, size=224, train=False):
        self.items, self.num_frames, self.size, self.train = items, num_frames, size, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label, _ = self.items[i]
        clip, _ = load_clip(path, self.num_frames, self.size)
        if self.train and random.random() < 0.5:          # horizontal flip augmentation
            clip = clip.flip(-1)
        return clip, label


class SyntheticDataset(Dataset):
    """Random clips for smoke-testing the code path without any data."""
    def __init__(self, n=8, num_frames=4, size=224):
        self.n, self.t, self.s = n, num_frames, size
    def __len__(self): return self.n
    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        return torch.randn(self.t, 3, self.s, self.s, generator=g), i % 2
