"""Projected ("vision-aided") discriminator on frozen DINOv2 features.

Projected GANs (Sauer et al. 2021) and vision-aided GANs (Kumari et al. 2022) show that
discriminating in a pretrained feature space speeds convergence by an order of magnitude.
We use the *self-supervised* DINOv2-S (not an ImageNet classifier), so the discriminator
cannot simply learn Inception/ImageNet class cues that would game FID/KID (Kynkaanniemi
et al. 2023).  One small trainable head per tapped block; R1/R2 are applied to the
(frozen) features, so no second-order gradient flows through the backbone.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from cocogoat.data.augment import diff_augment
from cocogoat.nn import ConditionEmbedding, DownStage, InvertedBottleneck, LayerNorm2d, MinibatchStd, kaiming_, lrelu
from cocogoat.train.adversarial import Critic


class ProjectedHead(nn.Module):
    def __init__(self, c_in: int, width: int = 256, grid: int = 16, blocks: int = 2, num_classes: int = 0,
                 cond_vector_dim: int = 0):
        super().__init__()
        self.norm = LayerNorm2d(c_in, affine=False)
        self.proj = kaiming_(nn.Conv2d(c_in, width, 1))
        stages = []
        g = grid
        while g > 4:
            stages.append(nn.ModuleList([InvertedBottleneck(width, 2.0, 3) for _ in range(blocks)]))
            stages.append(DownStage(width, width))
            g //= 2
        self.stages = nn.ModuleList(stages)
        self.mbstd = MinibatchStd(4)
        self.final = kaiming_(nn.Conv2d(width + 1, width, 3, padding=1))
        self.fc = kaiming_(nn.Linear(width * g * g, width))
        self.out = nn.Linear(width, 1)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.cmap = ConditionEmbedding(width, num_classes, cond_vector_dim)
        self.width = width

    def forward(self, f: torch.Tensor, c: torch.Tensor | None) -> torch.Tensor:
        h = self.proj(self.norm(f))
        for st in self.stages:
            if isinstance(st, nn.ModuleList):
                for b in st:
                    h = b(h)
            else:
                h = st(h)
        h = lrelu(self.final(self.mbstd(h)))
        phi = lrelu(self.fc(h.flatten(1)))
        logit = self.out(phi).squeeze(1)
        if self.cmap.enabled and c is not None:
            logit = logit + (phi * self.cmap(c)).sum(1) / math.sqrt(self.width)
        return logit


class ProjectedCritic(Critic):
    """encode = DiffAugment -> frozen backbone features; head = sum of per-layer heads."""

    keys = ("img",)

    def __init__(self, backbone: nn.Module, feat_dim: int, n_layers: int, grid: int, width: int = 256,
                 num_classes: int = 0, cond_vector_dim: int = 0, policy: str = "color,translation,cutout",
                 p: float = 0.5):
        super().__init__()
        self.backbone = backbone.requires_grad_(False)
        self.heads = nn.ModuleList([ProjectedHead(feat_dim, width, grid, 2, num_classes, cond_vector_dim)
                                    for _ in range(n_layers)])
        self.policy = policy
        self.p = p

    def trainable_parameters(self):
        return list(self.heads.parameters())

    def encode(self, sample):
        x = diff_augment(sample["img"], self.policy, self.p)
        feats = self.backbone(x)
        return {f"f{i}": f.float() for i, f in enumerate(feats)}

    def head(self, enc, cond):
        logits = [h(enc[f"f{i}"], cond) for i, h in enumerate(self.heads)]
        return torch.stack(logits, 0).mean(0)


class TinyBackbone(nn.Module):
    """Stand-in for DINOv2 in offline CPU tests: random frozen conv features at 16x16."""

    def __init__(self, dim: int = 32, layers: int = 2, grid: int = 16):
        super().__init__()
        self.grid = grid
        self.convs = nn.ModuleList([nn.Conv2d(3 if i == 0 else dim, dim, 3, padding=1) for i in range(layers)])
        self.requires_grad_(False)

    def forward(self, x):
        h = torch.nn.functional.interpolate(x, size=(self.grid, self.grid), mode="bilinear", antialias=True)
        feats = []
        for c in self.convs:
            h = torch.tanh(c(h))
            feats.append(h)
        return feats
