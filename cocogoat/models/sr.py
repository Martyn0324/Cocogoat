"""x4 super-resolution GAN generator, trained from scratch (no pretrained SR weights).

LR body: normalization-free depthwise inverted bottlenecks (large 7x7 depthwise kernels
for a wide receptive field at low cost) with a long skip; two x2 PixelShuffle stages with
small HR refinement stacks; the output is a residual added to the bicubic upsample, so the
network starts as bicubic interpolation and only learns the missing high frequencies.
Trained with L1 + LPIPS + RpGAN(+R1/R2) on (degraded LR crop, real HR crop) pairs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from cocogoat.nn import InvertedBottleneck, kaiming_, lrelu, zero_


@dataclass
class SRConfig:
    scale: int = 4
    channels: int = 96
    blocks: int = 12
    expansion: float = 2.0
    kernel: int = 7
    hr_channels: int = 64
    hr_blocks: int = 2


class SRNet(nn.Module):
    def __init__(self, cfg: SRConfig):
        super().__init__()
        assert cfg.scale in (2, 4)
        self.cfg = cfg
        c, hc = cfg.channels, cfg.hr_channels
        self.head = kaiming_(nn.Conv2d(3, c, 3, padding=1))
        self.body = nn.ModuleList([InvertedBottleneck(c, cfg.expansion, cfg.kernel) for _ in range(cfg.blocks)])
        self.body_out = nn.Conv2d(c, c, 3, padding=1)
        nn.init.xavier_uniform_(self.body_out.weight, gain=0.2)
        nn.init.zeros_(self.body_out.bias)
        ups, refine = [], []
        cin = c
        for _ in range({2: 1, 4: 2}[cfg.scale]):
            conv = nn.Conv2d(cin, hc * 4, 3, padding=1)
            nn.init.xavier_uniform_(conv.weight)
            nn.init.zeros_(conv.bias)
            ups.append(conv)
            refine.append(nn.ModuleList([InvertedBottleneck(hc, cfg.expansion, 3) for _ in range(cfg.hr_blocks)]))
            cin = hc
        self.ups = nn.ModuleList(ups)
        self.refine = nn.ModuleList(refine)
        self.out = zero_(nn.Conv2d(hc, 3, 3, padding=1))

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        base = F.interpolate(lr, scale_factor=self.cfg.scale, mode="bicubic", align_corners=False)
        h = self.head(lr)
        b = h
        for blk in self.body:
            b = blk(b)
        h = h + self.body_out(b)
        for up, ref in zip(self.ups, self.refine):
            h = lrelu(F.pixel_shuffle(up(h), 2))
            for blk in ref:
                h = blk(h)
        return base + self.out(lrelu(h))


def degrade(hr: torch.Tensor, scale: int = 4, blur_prob: float = 0.5, noise_prob: float = 0.3,
            max_noise: float = 0.02) -> torch.Tensor:
    """Mild, randomised degradation (blur -> downsample -> noise) for robust SR training."""
    b = hr.shape[0]
    x = hr
    if blur_prob > 0:
        sig = torch.empty(b, device=hr.device).uniform_(0.2, 1.2)
        k = 7
        coords = torch.arange(k, device=hr.device, dtype=torch.float32) - k // 2
        g = torch.exp(-coords[None].square() / (2 * sig[:, None].square()))
        g = g / g.sum(1, keepdim=True)
        kern = (g[:, :, None] * g[:, None, :])[:, None]  # (b,1,k,k)
        use = (torch.rand(b, device=hr.device) < blur_prob).float()[:, None, None, None]
        ident = torch.zeros_like(kern)
        ident[:, :, k // 2, k // 2] = 1
        kern = use * kern + (1 - use) * ident
        xp = F.pad(x, (k // 2,) * 4, mode="reflect")
        x = F.conv2d(xp.reshape(1, b * 3, *xp.shape[2:]), kern.repeat_interleave(3, 0), groups=b * 3)
        x = x.reshape(b, 3, *hr.shape[2:])
    mode = "bicubic" if torch.rand(()) < 0.5 else "bilinear"
    x = F.interpolate(x, scale_factor=1 / scale, mode=mode, antialias=True, align_corners=False)
    if noise_prob > 0:
        use = (torch.rand(b, 1, 1, 1, device=hr.device) < noise_prob).float()
        x = x + use * torch.randn_like(x) * torch.rand(b, 1, 1, 1, device=hr.device) * max_noise * 2
    return x.clamp(-1, 1)
