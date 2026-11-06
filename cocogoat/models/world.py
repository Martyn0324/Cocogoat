"""Latent world model: predicts the next frame's latent from past latents.

Factorised space-time transformer (Genie-style ST blocks) over latent tokens:
    spatial attention  - full attention among the tokens of one frame
    temporal attention - causal attention across frames, per spatial position
    MLP
with pre-RMSNorm, QK-norm and (optional) adaLN-Zero conditioning.

Memory (ViVAE design): every frame token is augmented with multi-scale exponential
moving averages of the past latents, m_t^(a) = a m_{t-1}^(a) + (1 - a) z_t, for
decays a in ``ema_decays``.  This gives a cheap, causal, RetNet/RWKV-like decaying
summary of the history *beyond* the attention window.

Prediction is residual: z_{t+1} = z_t + head(h_t).  With a frozen, KL-regularised
encoder, plain MSE on normalised latents is a sufficient objective (no collapse:
the targets are fixed).

``noise_dim > 0`` adds per-frame noise tokens and ``cond_dim > 0`` adds adaLN
conditioning, which Task 5 uses to turn this network into the generator of an
adversarial (stochastic) world model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from cocogoat.nn import RMSNorm, zero_


@dataclass
class WorldModelConfig:
    latent_channels: int = 16
    latent_size: int = 8
    patch: int = 1
    dim: int = 384
    depth: int = 8
    heads: int = 6
    mlp_ratio: float = 4.0
    max_frames: int = 32
    ema_decays: tuple = (0.5, 0.9, 0.98)
    cond_dim: int = 0
    noise_dim: int = 0
    residual: bool = True
    # Task 2.1 options (defaults reproduce the Task 2 model exactly)
    time_window: int = 0        # >0: each frame attends only to the last ``time_window`` frames (sliding window)
    rope_time: bool = False     # rotary (relative) time positions in temporal attention instead of learned pos_t
    ltm_every: int = 0          # >0: cross-attention to the long-term memory in every ``ltm_every``-th block
    ltm_age_buckets: int = 16   # log-spaced age buckets of memory slots


class TokenAttention(nn.Module):
    def __init__(self, dim: int, heads: int, double_backward_safe: bool = False):
        super().__init__()
        self.heads = heads
        self.hd = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.q_norm = RMSNorm(self.hd)
        self.k_norm = RMSNorm(self.hd)
        self.proj = nn.Linear(dim, dim)
        self.safe = double_backward_safe

    def forward(self, x: torch.Tensor, causal: bool = False, window: int = 0,
                rope: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        if window and window < n:
            i = torch.arange(n, device=x.device)
            allowed = (i[None, :] <= i[:, None]) & (i[:, None] - i[None, :] < window)
            if self.safe:
                logits = (q @ k.transpose(-1, -2)) * self.hd ** -0.5
                o = logits.masked_fill(~allowed, float("-inf")).softmax(-1) @ v
            else:
                o = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
            return self.proj(o.transpose(1, 2).reshape(b, n, d))
        if self.safe:
            logits = (q @ k.transpose(-1, -2)) * self.hd ** -0.5
            if causal:
                mask = torch.ones(n, n, dtype=torch.bool, device=x.device).triu(1)
                logits = logits.masked_fill(mask, float("-inf"))
            o = logits.softmax(-1) @ v
        else:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return self.proj(o.transpose(1, 2).reshape(b, n, d))


def rope_tables(n: int, dim: int, device, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables (n, dim) for rotary embeddings over positions 0..n-1 (dim even)."""
    inv = base ** (-torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    ang = torch.arange(n, device=device, dtype=torch.float32)[:, None] * inv[None]
    ang = torch.cat([ang, ang], dim=-1)
    return ang.cos(), ang.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (..., n, dim). Rotate-half formulation; relative positions only matter in q.k."""
    x1, x2 = x.chunk(2, dim=-1)
    rot = torch.cat([-x2, x1], dim=-1)
    return (x.float() * cos + rot.float() * sin).to(x.dtype)


class MemoryCrossAttention(nn.Module):
    """Frame tokens (queries) read the long-term memory tokens (keys/values). Zero-init output => no-op at start."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads, self.hd = heads, dim // heads
        self.norm = RMSNorm(dim, affine=False)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.q_norm, self.k_norm = RMSNorm(self.hd), RMSNorm(self.hd)
        self.proj = zero_(nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor, mem: torch.Tensor, mem_mask: torch.Tensor | None) -> torch.Tensor:
        """x: (B,T,N,D) tokens; mem: (B,M,D); mem_mask: (B,M) bool, True = valid."""
        b, t, n, d = x.shape
        q = self.q(self.norm(x)).view(b, t * n, self.heads, self.hd).transpose(1, 2)
        k, v = self.kv(mem).view(b, mem.shape[1], 2, self.heads, self.hd).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q), self.k_norm(k)
        mask = None if mem_mask is None else mem_mask[:, None, None, :]
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return x + self.proj(o.transpose(1, 2).reshape(b, t, n, d))


class STBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, cond_dim: int = 0, double_backward_safe: bool = False):
        super().__init__()
        self.n1, self.n2, self.n3 = RMSNorm(dim, affine=False), RMSNorm(dim, affine=False), RMSNorm(dim, affine=False)
        self.sattn = TokenAttention(dim, heads, double_backward_safe)
        self.tattn = TokenAttention(dim, heads, double_backward_safe)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, dim))
        # adaLN-Zero: (shift, scale, gate) x 3 sub-layers; without conditioning use learned constants
        self.cond_dim = cond_dim
        if cond_dim:
            self.ada = zero_(nn.Linear(cond_dim, 9 * dim))
        else:
            self.ada_const = nn.Parameter(torch.zeros(9 * dim))
        self.dim = dim

    def _mod(self, c: torch.Tensor | None, batch: int):
        if self.cond_dim and c is not None:
            p = self.ada(c)
        else:
            p = self.ada_const.expand(batch, -1)
        return p.view(batch, 9, self.dim).unbind(1)

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None, causal: bool = True, window: int = 0,
                rope: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        b, t, n, d = x.shape
        s1, sc1, g1, s2, sc2, g2, s3, sc3, g3 = [m[:, None, None, :] for m in self._mod(c, b)]
        h = self.n1(x) * (1 + sc1) + s1
        h = self.sattn(h.reshape(b * t, n, d)).reshape(b, t, n, d)
        x = x + (1 + g1) * h if not self.cond_dim else x + g1 * h
        h = self.n2(x) * (1 + sc2) + s2
        h = h.permute(0, 2, 1, 3).reshape(b * n, t, d)
        h = self.tattn(h, causal=causal, window=window, rope=rope).reshape(b, n, t, d).permute(0, 2, 1, 3)
        x = x + (1 + g2) * h if not self.cond_dim else x + g2 * h
        h = self.n3(x) * (1 + sc3) + s3
        h = self.mlp(h)
        return x + (1 + g3) * h if not self.cond_dim else x + g3 * h


def ema_memory(z: torch.Tensor, decays: tuple, init: torch.Tensor | None = None) -> torch.Tensor:
    """Multi-scale causal EMA over time. z: (B,T,C,H,W) -> (B,T,len(decays)*C,H,W)."""
    if not decays:
        return z[:, :, :0]
    outs = []
    for k, a in enumerate(decays):
        m = z[:, 0] if init is None else init[:, k]
        seq = []
        for t in range(z.shape[1]):
            m = a * m + (1 - a) * z[:, t]
            seq.append(m)
        outs.append(torch.stack(seq, dim=1))
    return torch.cat(outs, dim=2)


class WorldModel(nn.Module):
    def __init__(self, cfg: WorldModelConfig, double_backward_safe: bool = False):
        super().__init__()
        self.cfg = cfg
        p, c = cfg.patch, cfg.latent_channels
        assert cfg.latent_size % p == 0
        self.grid = cfg.latent_size // p
        self.n_tokens = self.grid * self.grid
        in_dim = c * p * p * (1 + len(cfg.ema_decays))
        self.inp = nn.Linear(in_dim, cfg.dim)
        self.pos_s = nn.Parameter(torch.randn(self.n_tokens, cfg.dim) * 0.02)
        self.pos_t = nn.Parameter(torch.randn(cfg.max_frames, cfg.dim) * 0.02)
        self.noise_inp = nn.Linear(cfg.noise_dim, cfg.dim) if cfg.noise_dim else None
        self.blocks = nn.ModuleList([STBlock(cfg.dim, cfg.heads, cfg.mlp_ratio, cfg.cond_dim, double_backward_safe)
                                     for _ in range(cfg.depth)])
        if cfg.ltm_every:
            e = cfg.ltm_every
            self.ltm_blocks = nn.ModuleDict({str(i): MemoryCrossAttention(cfg.dim, cfg.heads)
                                             for i in range(cfg.depth) if i % e == e - 1})
            self.ltm_inp = nn.Linear(c * p * p, cfg.dim)
            self.ltm_age = nn.Embedding(cfg.ltm_age_buckets, cfg.dim)
            self.ltm_null = nn.Parameter(torch.randn(1, cfg.dim) * 0.02)  # always-valid "nothing stored" token
        else:
            self.ltm_blocks = None
        self.norm = RMSNorm(cfg.dim)
        self.head = zero_(nn.Linear(cfg.dim, c * p * p))

    # ----------------------------------------------------------------------------------
    def patchify(self, z: torch.Tensor) -> torch.Tensor:
        """(B,T,C,H,W) -> (B,T,N,C*p*p)"""
        b, t, c, h, w = z.shape
        p = self.cfg.patch
        z = z.reshape(b, t, c, h // p, p, w // p, p).permute(0, 1, 3, 5, 2, 4, 6)
        return z.reshape(b, t, (h // p) * (w // p), c * p * p)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        b, t, n, _ = x.shape
        p, c, g = self.cfg.patch, self.cfg.latent_channels, self.grid
        x = x.reshape(b, t, g, g, c, p, p).permute(0, 1, 4, 2, 5, 3, 6)
        return x.reshape(b, t, c, g * p, g * p)

    def embed_ltm(self, slots: torch.Tensor, age: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Memory slots -> key/value tokens. slots (B,K,C,H,W), age (B,K) frames, valid (B,K) bool.

        Returns tokens (B, 1 + K*N, D) and mask (B, 1 + K*N); the first token is a learned null slot.
        """
        b, k = slots.shape[:2]
        tok = self.ltm_inp(self.patchify(slots.to(self.ltm_inp.weight.dtype)))     # (B,K,N,D)
        bucket = (torch.log2(age.float().clamp(min=0) + 1) * 2).long().clamp(max=self.cfg.ltm_age_buckets - 1)
        tok = tok + self.pos_s[None, None] + self.ltm_age(bucket)[:, :, None]
        tok = torch.cat([self.ltm_null.expand(b, 1, -1), tok.reshape(b, -1, tok.shape[-1])], dim=1)
        mask = torch.cat([torch.ones(b, 1, dtype=torch.bool, device=slots.device),
                          valid[:, :, None].expand(b, k, self.n_tokens).reshape(b, -1)], dim=1)
        return tok, mask

    def forward(self, z: torch.Tensor, cond: torch.Tensor | None = None, noise: torch.Tensor | None = None,
                memory: torch.Tensor | None = None, ltm: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        """z: (B,T,C,H,W) normalised latents. Returns predictions of z[:, 1:] for every t (shape of z).

        ``ltm``: output of ``embed_ltm`` (long-term memory tokens + mask), read by cross-attention blocks.
        """
        b, t = z.shape[:2]
        assert self.cfg.rope_time or t <= self.cfg.max_frames, f"{t} frames > max_frames {self.cfg.max_frames}"
        mem = ema_memory(z, self.cfg.ema_decays) if memory is None else memory
        x = self.inp(self.patchify(torch.cat([z, mem], dim=2)))
        x = x + self.pos_s[None, None]
        rope = None
        if self.cfg.rope_time:
            rope = rope_tables(t, self.cfg.dim // self.cfg.heads, z.device)
        else:
            x = x + self.pos_t[None, :t, None]
        if self.noise_inp is not None:
            if noise is None:
                noise = torch.randn(b, t, self.n_tokens, self.cfg.noise_dim, device=z.device, dtype=x.dtype)
            x = x + self.noise_inp(noise.to(x.dtype))
        for i, blk in enumerate(self.blocks):
            x = blk(x, cond, causal=True, window=self.cfg.time_window, rope=rope)
            if self.ltm_blocks is not None and ltm is not None and str(i) in self.ltm_blocks:
                x = self.ltm_blocks[str(i)](x, ltm[0].to(x.dtype), ltm[1])
        delta = self.unpatchify(self.head(self.norm(x)))
        return z + delta if self.cfg.residual else delta

    def rollout(self, context: torch.Tensor, steps: int, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Autoregressive prediction of ``steps`` future latents (B,steps,C,H,W).

        The EMA memory always covers the full history; attention sees the last
        ``max_frames`` frames.
        """
        seq = context
        preds = []
        w = self.cfg.max_frames
        for _ in range(steps):
            mem_full = ema_memory(seq, self.cfg.ema_decays)
            nxt = self.forward(seq[:, -w:], cond, memory=mem_full[:, -w:])[:, -1:]
            preds.append(nxt)
            seq = torch.cat([seq, nxt], dim=1)
        return torch.cat(preds, dim=1)
