"""
Attention-based CNN for deepfake video detection.

Pipeline (matches the review deck, Modules 2-5):
    frames -> ResNeXt-50 features F_t (C x H x W)
           -> spatial self-attention   F'_t = F_t + gamma * (V . A^T)
           -> global average pool      f_t   (C)
           -> temporal attention pool  v = sum_t alpha_t f_t
           -> classifier               y_hat = softmax(Wc v + bc)

`use_attention=False` gives the matched baseline: same backbone, plain
frame averaging, no attention blocks.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights


class SpatialSelfAttention(nn.Module):
    """Q = Wq F, K = Wk F, V = Wv F ; A = softmax(Q^T K / sqrt(d)) ; F' = F + gamma * (V A^T).

    Q and K are projected to a lower dimension `d` to keep the HW x HW
    attention matrix cheap; V keeps the full channel count so the residual
    add is shape-compatible. gamma starts at 0, so training begins from the
    baseline behaviour and learns how much attention to mix in.
    """

    def __init__(self, in_channels: int, d: int = 256):
        super().__init__()
        self.d = d
        self.wq = nn.Conv2d(in_channels, d, 1, bias=False)
        self.wk = nn.Conv2d(in_channels, d, 1, bias=False)
        self.wv = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, feat: torch.Tensor):
        b, c, h, w = feat.shape
        q = self.wq(feat).flatten(2)                 # B x d x N
        k = self.wk(feat).flatten(2)                 # B x d x N
        v = self.wv(feat).flatten(2)                 # B x C x N
        attn = torch.softmax(q.transpose(1, 2) @ k / math.sqrt(self.d), dim=-1)  # B x N x N
        out = v @ attn.transpose(1, 2)               # B x C x N
        out = feat + self.gamma * out.view(b, c, h, w)
        # Inspectable per-frame map: how much total attention each location receives.
        attn_map = attn.mean(dim=1).view(b, h, w)    # B x H x W
        attn_map = attn_map / (attn_map.amax(dim=(1, 2), keepdim=True) + 1e-8)
        return out, attn_map


class TemporalAttentionPooling(nn.Module):
    """e_t = w^T tanh(W f_t + b) ; alpha = softmax(e) ; v = sum_t alpha_t f_t."""

    def __init__(self, in_channels: int, hidden: int = 256):
        super().__init__()
        self.proj = nn.Linear(in_channels, hidden)
        self.score = nn.Linear(hidden, 1, bias=False)

    def forward(self, x: torch.Tensor):              # x: B x T x C
        e = self.score(torch.tanh(self.proj(x))).squeeze(-1)   # B x T
        alpha = torch.softmax(e, dim=1)
        v = (alpha.unsqueeze(-1) * x).sum(dim=1)     # B x C
        return v, alpha


class DeepfakeDetector(nn.Module):
    def __init__(self, use_attention: bool = True, pretrained: bool = True,
                 num_classes: int = 2, dropout: float = 0.4):
        super().__init__()
        self.use_attention = use_attention
        weights = ResNeXt50_32X4D_Weights.IMAGENET1K_V2 if pretrained else None
        net = resnext50_32x4d(weights=weights)
        self.backbone = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                      net.layer1, net.layer2, net.layer3, net.layer4)
        c = 2048
        if use_attention:
            self.spatial = SpatialSelfAttention(c)
            self.temporal = TemporalAttentionPooling(c)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(c, num_classes)

    def forward(self, clips: torch.Tensor, return_attention: bool = False):
        """clips: B x T x 3 x H x W  ->  logits (B x 2) [, dict of attention maps]"""
        b, t = clips.shape[:2]
        feat = self.backbone(clips.flatten(0, 1))    # (B*T) x C x h x w
        extras = {}
        if self.use_attention:
            feat, spatial_map = self.spatial(feat)
            extras["spatial"] = spatial_map.view(b, t, *spatial_map.shape[-2:])
        f = feat.mean(dim=(2, 3)).view(b, t, -1)     # B x T x C
        if self.use_attention:
            v, alpha = self.temporal(f)
            extras["temporal"] = alpha
        else:
            v = f.mean(dim=1)                        # naive frame averaging (baseline)
        logits = self.fc(self.drop(v))
        return (logits, extras) if return_attention else logits

    def freeze_backbone(self, freeze: bool = True):
        for p in self.backbone.parameters():
            p.requires_grad = not freeze
