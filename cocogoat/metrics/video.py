"""Frame-level reconstruction / prediction metrics (inputs NCHW in [-1, 1])."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def psnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-image PSNR in dB for images in [-1, 1] (computed on the [0, 1] scale)."""
    mse = ((x.float() - y.float()) / 2).square().flatten(1).mean(1).clamp_min(1e-10)
    return 10 * torch.log10(1.0 / mse)


def _gaussian_window(size: int = 11, sigma: float = 1.5, device=None) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-coords.square() / (2 * sigma * sigma))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None]


def ssim(x: torch.Tensor, y: torch.Tensor, window: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Per-image SSIM (Wang et al. 2004) on the [0, 1] scale, averaged over channels."""
    x = (x.float() + 1) / 2
    y = (y.float() + 1) / 2
    c = x.shape[1]
    w = _gaussian_window(window, sigma, x.device).expand(c, 1, window, window)
    pad = window // 2
    mu_x = F.conv2d(x, w, padding=pad, groups=c)
    mu_y = F.conv2d(y, w, padding=pad, groups=c)
    sxx = F.conv2d(x * x, w, padding=pad, groups=c) - mu_x.square()
    syy = F.conv2d(y * y, w, padding=pad, groups=c) - mu_y.square()
    sxy = F.conv2d(x * y, w, padding=pad, groups=c) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x.square() + mu_y.square() + c1) * (sxx + syy + c2))
    return s.flatten(1).mean(1)


class LPIPSMetric(torch.nn.Module):
    """LPIPS (VGG) distance; ``random_weights=True`` only for offline CPU tests."""

    def __init__(self, device="cuda", random_weights: bool = False):
        super().__init__()
        import lpips

        if random_weights:
            self.net = lpips.LPIPS(net="vgg", pretrained=False, pnet_rand=True, verbose=False)
        else:
            self.net = lpips.LPIPS(net="vgg", verbose=False)
        self.net = self.net.to(device).eval().requires_grad_(False)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(x.float(), y.float()).flatten()
