"""Per-frame (2D) beta-VAE used by the world model (Task 2), the latent GAN (Task 3) and Task 5.

Encoder:  conv3x3 stem -> [ConvNeXt blocks -> ResidualDown(PixelUnshuffle)] x L -> bottleneck
          (ConvNeXt + multi-head self-attention) -> LN -> 1x1 -> (mean, logvar)
Decoder:  mirror, with ResidualUp(PixelShuffle).  The down/up blocks carry DC-AE style
          parameter-free space<->channel shortcuts, so the high-compression path is easy to optimise.

The VAE is deliberately 2D / per-frame (no temporal convolutions): all dynamics live in
the world model, as in the ViVAE design notes.

Latent normalisation: after training, per-channel mean/std of the posterior means are
stored in buffers; downstream models (world model, latent GAN) work on
``normalize(z) = (z - mean) / std`` and decode ``denormalize(z)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from cocogoat.nn import ConvNeXtBlock, LayerNorm2d, ResidualDown, ResidualUp, SelfAttention2d


# Bulk no-grad encode/decode passes (latent statistics, evaluation, encoding whole clips) are split so that one
# pass holds at most this many pixels: 512 frames at 64x64 (unchanged), 32 frames at 256x256. A fixed frame
# count tuned for 64x64 needs 16x the memory at 256x256 (Task 5: a 32 GiB activation per layer, OOM).
BULK_PIXELS = 512 * 64 * 64


def frames_per_pass(height: int, width: int) -> int:
    """Frames per bulk VAE pass at this (output) resolution."""
    return max(1, BULK_PIXELS // (height * width))


@dataclass
class VAEConfig:
    in_channels: int = 3
    latent_channels: int = 16
    channels: tuple = (64, 128, 256, 256)  # one entry per level; len-1 downsamplings => f = 2**(len-1)
    blocks: tuple = (1, 2, 2, 2)
    mid_blocks: int = 2
    attn: bool = True
    heads: int = 4
    expansion: float = 4.0
    kernel: int = 7
    logvar_min: float = -20.0
    logvar_max: float = 4.0

    @property
    def downsample(self) -> int:
        return 2 ** (len(self.channels) - 1)


class _Mid(nn.Module):
    def __init__(self, c: int, cfg: VAEConfig):
        super().__init__()
        n = max(cfg.mid_blocks, 1)
        self.pre = nn.ModuleList([ConvNeXtBlock(c, cfg.expansion, cfg.kernel) for _ in range((n + 1) // 2)])
        self.attn = SelfAttention2d(c, cfg.heads) if cfg.attn else None
        self.post = nn.ModuleList([ConvNeXtBlock(c, cfg.expansion, cfg.kernel) for _ in range(n // 2)])

    def forward(self, x):
        for b in self.pre:
            x = b(x)
        if self.attn is not None:
            x = self.attn(x)
        for b in self.post:
            x = b(x)
        return x


class Encoder(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        ch, nb = cfg.channels, cfg.blocks
        self.stem = nn.Conv2d(cfg.in_channels, ch[0], 3, padding=1)
        self.levels = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i, c in enumerate(ch):
            self.levels.append(nn.ModuleList([ConvNeXtBlock(c, cfg.expansion, cfg.kernel) for _ in range(nb[i])]))
            if i < len(ch) - 1:
                self.downs.append(ResidualDown(c, ch[i + 1]))
        self.mid = _Mid(ch[-1], cfg)
        self.norm = LayerNorm2d(ch[-1])
        self.out = nn.Conv2d(ch[-1], 2 * cfg.latent_channels, 1)

    def forward(self, x):
        h = self.stem(x)
        for i, blocks in enumerate(self.levels):
            for b in blocks:
                h = b(h)
            if i < len(self.downs):
                h = self.downs[i](h)
        h = self.mid(h)
        return self.out(F.gelu(self.norm(h)))


class Decoder(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        ch, nb = cfg.channels, cfg.blocks
        self.inp = nn.Conv2d(cfg.latent_channels, ch[-1], 3, padding=1)
        self.mid = _Mid(ch[-1], cfg)
        self.levels = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i in reversed(range(len(ch))):
            self.levels.append(nn.ModuleList([ConvNeXtBlock(ch[i], cfg.expansion, cfg.kernel)
                                              for _ in range(nb[i] + 1)]))
            if i > 0:
                self.ups.append(ResidualUp(ch[i], ch[i - 1]))
        self.norm = LayerNorm2d(ch[0])
        self.out = nn.Conv2d(ch[0], cfg.in_channels, 3, padding=1)

    def forward(self, z):
        h = self.mid(self.inp(z))
        for i, blocks in enumerate(self.levels):
            for b in blocks:
                h = b(h)
            if i < len(self.ups):
                h = self.ups[i](h)
        return self.out(F.gelu(self.norm(h)))


class VAE(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.register_buffer("latent_mean", torch.zeros(cfg.latent_channels))
        self.register_buffer("latent_std", torch.ones(cfg.latent_channels))

    # -- core --------------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, logvar = self.encoder(x).float().chunk(2, dim=1)
        return mean, logvar.clamp(self.cfg.logvar_min, self.cfg.logvar_max)

    @staticmethod
    def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mean + torch.randn_like(mean) * (0.5 * logvar).exp()

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor, sample: bool = True):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar) if sample else mean
        return self.decode(z), mean, logvar, z

    # -- normalised latent interface ---------------------------------------------------
    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.latent_mean.view(1, -1, 1, 1)) / self.latent_std.view(1, -1, 1, 1)

    def denormalize(self, zn: torch.Tensor) -> torch.Tensor:
        return zn * self.latent_std.view(1, -1, 1, 1) + self.latent_mean.view(1, -1, 1, 1)

    @torch.no_grad()
    def encode_normalized(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        out = []
        for xc in x.split(frames_per_pass(*x.shape[-2:])):
            mean, logvar = self.encode(xc)
            out.append(self.normalize(self.reparameterize(mean, logvar) if sample else mean))
        return torch.cat(out)

    def decode_normalized(self, zn: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():  # training: one pass (the activations are kept for backward anyway)
            return self.decode(self.denormalize(zn))
        f = self.cfg.downsample
        k = frames_per_pass(zn.shape[-2] * f, zn.shape[-1] * f)
        return torch.cat([self.decode(self.denormalize(zc)) for zc in zn.split(k)])

    @torch.no_grad()
    def fit_latent_stats(self, batches) -> None:
        s, s2, n = 0.0, 0.0, 0
        for x in batches:
            for xc in x.split(frames_per_pass(*x.shape[-2:])):
                mean, _ = self.encode(xc)
                s = s + mean.double().sum(dim=(0, 2, 3))
                s2 = s2 + mean.double().square().sum(dim=(0, 2, 3))
                n += mean.shape[0] * mean.shape[2] * mean.shape[3]
        mu = s / n
        std = (s2 / n - mu.square()).clamp_min(1e-12).sqrt()
        self.latent_mean.copy_(mu.float())
        self.latent_std.copy_(std.float())


def kl_divergence(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Per-element mean KL(q(z|x) || N(0, I))."""
    return 0.5 * (mean.square() + logvar.exp() - 1.0 - logvar).mean()


def build_vae(cfg: VAEConfig | None = None, **kw) -> VAE:
    cfg = cfg or VAEConfig(**kw)
    return VAE(cfg)


def load_vae(path: str, device: str | torch.device = "cpu") -> VAE:
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = VAEConfig(**state["vae_config"])
    vae = VAE(cfg)
    vae.load_state_dict(state["vae"])
    return vae.to(device).eval()

