"""Latent-space generator for the latent GAN (Task 3) - data flow z -> G -> z' -> Decoder -> image.

The generator lives entirely in the (normalised) VAE latent space: its spatial input
noise has the latent's shape, it outputs a latent of the same shape, and with
``residual=True`` it learns a *displacement* z' = z + G(z, c).  Combined with the
optional optimal-transport penalty ||z' - z||^2 (Adversarial Flow Models, Lin et al.)
this biases G towards the unique OT map from N(0, I) to the latent data distribution.

Global randomness (style) enters through a mapping network that also embeds the class;
it modulates every block through FiLM (zero-initialised).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from cocogoat.nn import MLP, ConditionEmbedding, InvertedBottleneck, SelfAttention2d, lrelu, normalize_2nd_moment, zero_


@dataclass
class LatentGeneratorConfig:
    latent_channels: int = 16
    latent_size: int = 8
    z_global_dim: int = 128
    w_dim: int = 512
    cond_dim: int = 256
    num_classes: int = 0
    cond_vector_dim: int = 0
    width: int = 512
    depth: int = 8
    heads: int = 8
    expansion: float = 3.0
    kernel: int = 3
    attn_every: int = 1
    residual: bool = True


class LatentGenerator(nn.Module):
    def __init__(self, cfg: LatentGeneratorConfig):
        super().__init__()
        self.cfg = cfg
        self.cond = ConditionEmbedding(cfg.cond_dim, cfg.num_classes, cfg.cond_vector_dim)
        in_dim = cfg.z_global_dim + (cfg.cond_dim if self.cond.enabled else 0)
        self.mapping = MLP([in_dim, cfg.w_dim, cfg.w_dim])
        self.stem = nn.Conv2d(cfg.latent_channels, cfg.width, 1)
        nn.init.xavier_uniform_(self.stem.weight)
        nn.init.zeros_(self.stem.bias)
        self.pos = nn.Parameter(torch.zeros(1, cfg.width, cfg.latent_size, cfg.latent_size))
        blocks = []
        for i in range(cfg.depth):
            blocks.append(InvertedBottleneck(cfg.width, cfg.expansion, cfg.kernel, cond_dim=cfg.w_dim))
            if cfg.attn_every and (i % cfg.attn_every == 0):
                blocks.append(SelfAttention2d(cfg.width, cfg.heads, cond_dim=cfg.w_dim))
        self.blocks = nn.ModuleList(blocks)
        self.out = nn.Conv2d(cfg.width, cfg.latent_channels, 1)
        if cfg.residual:
            zero_(self.out)

    def forward(self, z: torch.Tensor, z_global: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        parts = [normalize_2nd_moment(z_global)]
        if self.cond.enabled:
            assert c is not None
            parts.append(normalize_2nd_moment(self.cond(c)))
        w = self.mapping(torch.cat(parts, dim=1))
        h = self.stem(z) + self.pos
        for blk in self.blocks:
            h = blk(h, w)
        delta = self.out(lrelu(h))
        return z + delta if self.cfg.residual else delta
