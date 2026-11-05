"""Differentiable augmentation applied to *both* real and fake images before D.

DiffAugment (Zhao et al., NeurIPS 2020): colour (brightness/saturation/contrast),
translation and cutout, all differentiable (also twice-differentiable, so R1/R2 can be
computed through them).  Augmentation is applied per sample with probability ``p``.
Inputs are NCHW tensors in [-1, 1].
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _brightness(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)


def _saturation(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=1, keepdim=True)
    return (x - mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + mean


def _contrast(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    return (x - mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + mean


def _translation(x: torch.Tensor, ratio: float = 0.125) -> torch.Tensor:
    b, c, h, w = x.shape
    sh, sw = int(h * ratio + 0.5), int(w * ratio + 0.5)
    tx = torch.randint(-sh, sh + 1, size=[b, 1, 1], device=x.device)
    ty = torch.randint(-sw, sw + 1, size=[b, 1, 1], device=x.device)
    gb, gx, gy = torch.meshgrid(
        torch.arange(b, dtype=torch.long, device=x.device),
        torch.arange(h, dtype=torch.long, device=x.device),
        torch.arange(w, dtype=torch.long, device=x.device),
        indexing="ij",
    )
    gx = torch.clamp(gx + tx + 1, 0, h + 1)
    gy = torch.clamp(gy + ty + 1, 0, w + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    return x_pad.permute(0, 2, 3, 1).contiguous()[gb, gx, gy].permute(0, 3, 1, 2)


def _cutout(x: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
    b, c, h, w = x.shape
    ch, cw = int(h * ratio + 0.5), int(w * ratio + 0.5)
    ox = torch.randint(0, h + (1 - ch % 2), size=[b, 1, 1], device=x.device)
    oy = torch.randint(0, w + (1 - cw % 2), size=[b, 1, 1], device=x.device)
    gb, gx, gy = torch.meshgrid(
        torch.arange(b, dtype=torch.long, device=x.device),
        torch.arange(ch, dtype=torch.long, device=x.device),
        torch.arange(cw, dtype=torch.long, device=x.device),
        indexing="ij",
    )
    gx = torch.clamp(gx + ox - ch // 2, min=0, max=h - 1)
    gy = torch.clamp(gy + oy - cw // 2, min=0, max=w - 1)
    mask = torch.ones(b, h, w, dtype=x.dtype, device=x.device)
    mask[gb, gx, gy] = 0
    return x * mask.unsqueeze(1)


POLICIES = {
    "color": [_brightness, _saturation, _contrast],
    "translation": [_translation],
    "cutout": [_cutout],
}


def diff_augment(x: torch.Tensor, policy: str = "color,translation,cutout", p: float = 1.0) -> torch.Tensor:
    """Apply DiffAugment to each sample independently with probability ``p``.

    Works for any channel count (RGB images or VAE latents); the colour ops are only
    applied when the input has 3 channels.
    """
    if not policy or p <= 0:
        return x
    aug = x
    for name in policy.split(","):
        name = name.strip()
        if not name:
            continue
        if name == "color" and x.size(1) != 3:
            continue
        for fn in POLICIES[name]:
            aug = fn(aug)
    if p >= 1:
        return aug
    keep = (torch.rand(x.size(0), 1, 1, 1, device=x.device) >= p).to(x.dtype)
    return keep * x + (1 - keep) * aug
