"""Perceptual loss (LPIPS-VGG). VGG is an ImageNet classifier - allowed by the project rules."""

from __future__ import annotations

import torch
import torch.nn as nn


class PerceptualLoss(nn.Module):
    def __init__(self, device="cuda", random_weights: bool = False):
        super().__init__()
        import lpips

        if random_weights:  # offline CPU tests only
            net = lpips.LPIPS(net="vgg", pretrained=False, pnet_rand=True, verbose=False)
        else:
            net = lpips.LPIPS(net="vgg", verbose=False)
        self.net = net.to(device).eval().requires_grad_(False)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Mean LPIPS between batches in [-1, 1] (differentiable w.r.t. both)."""
        return self.net(x.float(), y.float()).mean()
