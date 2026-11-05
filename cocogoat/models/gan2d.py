"""Pixel-space GAN: modernised, normalization-free generator and discriminator.

Generator (64..256 px):
    z (+ class / vector condition) -> mapping MLP -> w
    w -> 4x4 tensor -> [UpStage -> InvertedBottleneck(FiLM w) x n -> SelfAttention(FiLM w)] ... -> 1x1 conv -> RGB
Discriminator:
    RGB -> 1x1 conv -> [InvertedBottleneck x n -> SelfAttention -> DownStage(PixelUnshuffle)] ... ->
    4x4 -> minibatch-std -> 3x3 conv -> FC -> features phi
    logit = psi(phi) + <embed(c), phi> / sqrt(dim)   (projection discriminator, Miyato & Koyama 2018)

Both are trained with RpGAN + R1 + R2 (``cocogoat.losses``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from cocogoat.nn import (
    MLP,
    ConditionEmbedding,
    DownStage,
    InvertedBottleneck,
    MinibatchStd,
    SelfAttention2d,
    UpStage,
    kaiming_,
    lrelu,
    normalize_2nd_moment,
    resolution_stages,
)


def _expansion(e, r: int) -> float:
    """Expansion ratio for resolution r: a float, or {res: ratio} (nearest key in log2, ties -> larger)."""
    if isinstance(e, dict):
        e = {int(k): v for k, v in e.items()}
        if r in e:
            return e[r]
        key = min(e, key=lambda k: (abs(math.log2(k) - math.log2(r)), -k))
        return e[key]
    return e


def _default_channels(resolution: int, top: int = 512, bottom: int = 128) -> dict[int, int]:
    """512 channels at <=8 px, halving towards the output, floored at ``bottom``."""
    chans = {}
    for r in resolution_stages(resolution):
        chans[r] = int(max(bottom, min(top, top * 8 // r * 1 if r > 8 else top)))
    # smooth decay: 4,8 -> top; 16 -> 3/4 top; 32 -> top/2; 64 -> top/4 ...
    for r in chans:
        if r >= 16:
            chans[r] = int(max(bottom, top * (0.75 if r == 16 else 8.0 / r)))
    return chans


@dataclass
class GeneratorConfig:
    resolution: int = 64
    z_dim: int = 128
    w_dim: int = 512
    cond_dim: int = 256
    num_classes: int = 0
    cond_vector_dim: int = 0
    channels: dict[int, int] = field(default_factory=dict)
    blocks: int = 2
    attn_res: tuple[int, ...] = (16,)
    heads: int = 4
    expansion: float | dict = 3.0       # or {resolution: expansion}
    kernel: int = 3
    mapping_layers: int = 2
    out_channels: int = 3

    def resolved_channels(self) -> dict[int, int]:
        return {int(k): v for k, v in self.channels.items()} if self.channels else _default_channels(self.resolution)


@dataclass
class DiscriminatorConfig:
    resolution: int = 64
    num_classes: int = 0
    cond_vector_dim: int = 0
    channels: dict[int, int] = field(default_factory=dict)
    blocks: int = 2
    attn_res: tuple[int, ...] = (16, 8)
    heads: int = 4
    expansion: float | dict = 3.0       # or {resolution: expansion}
    kernel: int = 3
    mbstd_group: int = 4
    in_channels: int = 3

    def resolved_channels(self) -> dict[int, int]:
        return {int(k): v for k, v in self.channels.items()} if self.channels else _default_channels(self.resolution)


class Generator(nn.Module):
    def __init__(self, cfg: GeneratorConfig):
        super().__init__()
        self.cfg = cfg
        chans = cfg.resolved_channels()
        self.stages_res = resolution_stages(cfg.resolution)
        self.cond = ConditionEmbedding(cfg.cond_dim, cfg.num_classes, cfg.cond_vector_dim)
        in_dim = cfg.z_dim + (cfg.cond_dim if self.cond.enabled else 0)
        self.mapping = MLP([in_dim] + [cfg.w_dim] * cfg.mapping_layers)
        c4 = chans[4]
        self.input = kaiming_(nn.Linear(cfg.w_dim, c4 * 16))
        self.ups = nn.ModuleDict()
        self.blocks = nn.ModuleDict()
        self.attn = nn.ModuleDict()
        prev = c4
        for r in self.stages_res:
            c = chans[r]
            if r > 4:
                self.ups[str(r)] = UpStage(prev, c)
            self.blocks[str(r)] = nn.ModuleList(
                [InvertedBottleneck(c, _expansion(cfg.expansion, r), cfg.kernel, cond_dim=cfg.w_dim)
                 for _ in range(cfg.blocks)])
            if r in cfg.attn_res:
                self.attn[str(r)] = SelfAttention2d(c, cfg.heads, cond_dim=cfg.w_dim)
            prev = c
        self.to_rgb = nn.Conv2d(prev, cfg.out_channels, 1)
        nn.init.xavier_uniform_(self.to_rgb.weight, gain=0.5)
        nn.init.zeros_(self.to_rgb.bias)

    def mapping_w(self, z: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        parts = [normalize_2nd_moment(z)]
        if self.cond.enabled:
            assert c is not None, "conditional generator needs c"
            parts.append(normalize_2nd_moment(self.cond(c)))
        return self.mapping(torch.cat(parts, dim=1))

    def synthesis(self, w: torch.Tensor) -> torch.Tensor:
        x = lrelu(self.input(w)).view(w.shape[0], -1, 4, 4)
        for r in self.stages_res:
            key = str(r)
            if key in self.ups:
                x = self.ups[key](x)
            for blk in self.blocks[key]:
                x = blk(x, w)
            if key in self.attn:
                x = self.attn[key](x, w)
        return self.to_rgb(lrelu(x))

    def forward(self, z: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        return self.synthesis(self.mapping_w(z, c))


class Discriminator(nn.Module):
    def __init__(self, cfg: DiscriminatorConfig):
        super().__init__()
        self.cfg = cfg
        chans = cfg.resolved_channels()
        res_desc = list(reversed(resolution_stages(cfg.resolution)))
        self.res_desc = res_desc
        self.from_rgb = kaiming_(nn.Conv2d(cfg.in_channels, chans[cfg.resolution], 1))
        self.blocks = nn.ModuleDict()
        self.attn = nn.ModuleDict()
        self.downs = nn.ModuleDict()
        for r in res_desc:
            c = chans[r]
            self.blocks[str(r)] = nn.ModuleList(
                [InvertedBottleneck(c, _expansion(cfg.expansion, r), cfg.kernel) for _ in range(cfg.blocks)])
            if r in cfg.attn_res:
                self.attn[str(r)] = SelfAttention2d(c, cfg.heads, double_backward_safe=True)
            if r > 4:
                self.downs[str(r)] = DownStage(c, chans[r // 2])
        c4 = chans[4]
        self.mbstd = MinibatchStd(cfg.mbstd_group) if cfg.mbstd_group > 1 else None
        self.final_conv = kaiming_(nn.Conv2d(c4 + (1 if self.mbstd else 0), c4, 3, padding=1))
        self.fc = kaiming_(nn.Linear(c4 * 16, c4))
        self.out = nn.Linear(c4, 1)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.cmap = ConditionEmbedding(c4, cfg.num_classes, cfg.cond_vector_dim)
        self.feat_dim = c4

    def features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.from_rgb(x)
        for r in self.res_desc:
            key = str(r)
            for blk in self.blocks[key]:
                h = blk(h)
            if key in self.attn:
                h = self.attn[key](h)
            if key in self.downs:
                h = self.downs[key](h)
        if self.mbstd is not None:
            h = self.mbstd(h)
        h = lrelu(self.final_conv(h))
        return lrelu(self.fc(h.flatten(1)))

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        phi = self.features(x)
        logit = self.out(phi).squeeze(1)
        if self.cmap.enabled and c is not None:
            logit = logit + (phi * self.cmap(c)).sum(dim=1) / math.sqrt(self.feat_dim)
        return logit
