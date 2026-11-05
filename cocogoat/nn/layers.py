"""Building blocks shared by every model in the project.

Design notes (see docs/superpowers/specs/2026-11-05-cocogoat-revival-design.md):

* Residual branches are *normalization-free* and use a Fix-up / SkipInit style
  initialization (last layer of every residual branch starts at zero), as in R3GAN
  (Huang et al., NeurIPS 2024).  The network therefore starts as an identity map plus
  the stem, and gradient penalties (R1/R2) are not disturbed by batch statistics.
* Convolutions are depthwise-separable inverted bottlenecks (1x1 expand -> depthwise
  kxk -> 1x1 project), the MobileNetV2/ConvNeXt recipe.
* Attention is multi-head self-attention with pre-RMSNorm and QK-normalization
  (stabilises the logit scale).  ``double_backward_safe=True`` uses an explicit
  softmax(QK^T)V implementation, because fused SDPA kernels do not support the
  second-order gradients that R1/R2 penalties need in the discriminator.
* Conditioning enters through FiLM (per-channel scale/shift), zero-initialised so
  that every conditional block starts as its unconditional counterpart.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LRELU_SLOPE = 0.2


def lrelu(x: torch.Tensor) -> torch.Tensor:
    return F.leaky_relu(x, LRELU_SLOPE)


def kaiming_(module: nn.Module) -> nn.Module:
    """He-normal init for LeakyReLU(0.2) layers; zero bias."""
    nn.init.kaiming_normal_(module.weight, a=LRELU_SLOPE, mode="fan_in", nonlinearity="leaky_relu")
    if getattr(module, "bias", None) is not None:
        nn.init.zeros_(module.bias)
    return module


def zero_(module: nn.Module) -> nn.Module:
    nn.init.zeros_(module.weight)
    if getattr(module, "bias", None) is not None:
        nn.init.zeros_(module.bias)
    return module


def normalize_2nd_moment(x: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


class RMSNorm(nn.Module):
    """RMS normalisation over the last dimension (per token, per sample)."""

    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * (x.square().mean(dim=-1, keepdim=True) + self.eps).rsqrt()
        x = x.to(dtype)
        return x * self.weight.to(dtype) if self.weight is not None else x


class FiLM(nn.Module):
    """x * (1 + scale(w)) + shift(w), zero-initialised (identity at init)."""

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.proj = zero_(nn.Linear(cond_dim, 2 * channels))

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(w).chunk(2, dim=-1)
        if x.ndim == 4:  # B,C,H,W
            return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        if x.ndim == 3:  # B,N,C
            return x * (1 + scale[:, None, :]) + shift[:, None, :]
        return x * (1 + scale) + shift


class InvertedBottleneck(nn.Module):
    """Normalization-free depthwise-separable residual block.

    out = x + PW2( act( DW( act( FiLM( PW1(x) ) ) ) ) )
    """

    def __init__(self, channels: int, expansion: float = 2.0, kernel: int = 3, cond_dim: int | None = None,
                 zero_init: bool = True):
        super().__init__()
        hidden = int(round(channels * expansion))
        self.pw1 = kaiming_(nn.Conv2d(channels, hidden, 1))
        self.film = FiLM(cond_dim, hidden) if cond_dim else None
        self.dw = kaiming_(nn.Conv2d(hidden, hidden, kernel, padding=kernel // 2, groups=hidden))
        self.pw2 = nn.Conv2d(hidden, channels, 1)
        if zero_init:
            zero_(self.pw2)
        else:
            kaiming_(self.pw2)

    def forward(self, x: torch.Tensor, w: torch.Tensor | None = None) -> torch.Tensor:
        h = self.pw1(x)
        if self.film is not None and w is not None:
            h = self.film(h, w)
        h = self.dw(lrelu(h))
        h = self.pw2(lrelu(h))
        return x + h


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, double_backward_safe: bool) -> torch.Tensor:
    """Multi-head attention on (B, heads, N, d) tensors."""
    if double_backward_safe:
        scale = q.shape[-1] ** -0.5
        logits = torch.matmul(q, k.transpose(-1, -2)) * scale
        return torch.matmul(logits.softmax(dim=-1), v)
    return F.scaled_dot_product_attention(q, k, v)


class SelfAttention2d(nn.Module):
    """Pre-RMSNorm multi-head self-attention over the H*W tokens of a feature map."""

    def __init__(self, channels: int, heads: int = 4, cond_dim: int | None = None, double_backward_safe: bool = False):
        super().__init__()
        assert channels % heads == 0, (channels, heads)
        self.heads = heads
        self.head_dim = channels // heads
        self.norm = RMSNorm(channels, affine=cond_dim is None)
        self.film = FiLM(cond_dim, channels) if cond_dim else None
        self.qkv = nn.Linear(channels, 3 * channels)
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.zeros_(self.qkv.bias)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = zero_(nn.Linear(channels, channels))
        self.double_backward_safe = double_backward_safe

    def forward(self, x: torch.Tensor, w: torch.Tensor | None = None) -> torch.Tensor:
        b, c, h, wd = x.shape
        t = x.flatten(2).transpose(1, 2)  # B,N,C
        y = self.norm(t)
        if self.film is not None and w is not None:
            y = self.film(y, w)
        q, k, v = self.qkv(y).view(b, h * wd, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q), self.k_norm(k)
        o = attention(q, k, v, self.double_backward_safe)
        o = o.transpose(1, 2).reshape(b, h * wd, c)
        o = self.proj(o)
        return x + o.transpose(1, 2).reshape(b, c, h, wd)


class MinibatchStd(nn.Module):
    """StyleGAN minibatch standard-deviation feature (lets D see sample diversity)."""

    def __init__(self, group_size: int = 4, num_features: int = 1):
        super().__init__()
        self.group_size = group_size
        self.num_features = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        g = min(self.group_size, n)
        while n % g:
            g -= 1
        f = self.num_features
        y = x.float().reshape(g, -1, f, c // f, h, w)
        y = y - y.mean(dim=0)
        y = y.square().mean(dim=0)
        y = (y + 1e-8).sqrt()
        y = y.mean(dim=[2, 3, 4])
        y = y.reshape(-1, f, 1, 1).to(x.dtype)
        y = y.repeat(g, 1, h, w)
        return torch.cat([x, y], dim=1)


class UpStage(nn.Module):
    """Bilinear x2 upsampling followed by a 1x1 channel projection (alias-friendly)."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.proj = nn.Conv2d(cin, cout, 1)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.proj(x)


class DownStage(nn.Module):
    """PixelUnshuffle (lossless space-to-channel) followed by a 1x1 channel mix."""

    def __init__(self, cin: int, cout: int, factor: int = 2):
        super().__init__()
        self.factor = factor
        self.proj = nn.Conv2d(cin * factor * factor, cout, 1)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.pixel_unshuffle(x, self.factor))


class MLP(nn.Module):
    def __init__(self, dims: list[int], final_act: bool = True):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(kaiming_(nn.Linear(dims[i], dims[i + 1])))
        self.layers = nn.ModuleList(layers)
        self.final_act = final_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1 or self.final_act:
                x = lrelu(x)
        return x


class ConditionEmbedding(nn.Module):
    """Embeds integer class ids (nn.Embedding) or continuous vectors (Linear), or both.

    Integer tensors go through the table, floating tensors through the projection.
    """

    def __init__(self, dim: int, num_classes: int = 0, vector_dim: int = 0):
        super().__init__()
        self.num_classes = num_classes
        self.vector_dim = vector_dim
        self.table = nn.Embedding(num_classes, dim) if num_classes else None
        self.lin = nn.Linear(vector_dim, dim) if vector_dim else None
        if self.table is not None:
            nn.init.normal_(self.table.weight, std=1.0)
        if self.lin is not None:
            nn.init.xavier_uniform_(self.lin.weight)
            nn.init.zeros_(self.lin.bias)

    @property
    def enabled(self) -> bool:
        return bool(self.num_classes or self.vector_dim)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        if c.dtype in (torch.int64, torch.int32, torch.int16, torch.uint8):
            assert self.table is not None, "integer condition given but num_classes == 0"
            return self.table(c.long())
        assert self.lin is not None, "vector condition given but vector_dim == 0"
        return self.lin(c.to(self.lin.weight.dtype))


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def resolution_stages(res: int, base: int = 4) -> list[int]:
    """[4, 8, ..., res] for power-of-two resolutions."""
    n = int(math.log2(res // base))
    assert base * 2 ** n == res, f"resolution {res} must be {base} * 2^k"
    return [base * 2 ** i for i in range(n + 1)]
