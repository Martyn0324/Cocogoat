"""Video data: Colored Moving MNIST (synthetic) and frame arrays (e.g. Shelterdataset.npz).

Clips never straddle a scene cut: cuts are detected from the mean absolute difference
between consecutive frames (robust z-score threshold), and every clip is drawn inside
one segment.  Frames are kept on the GPU as uint8.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# Colored Moving MNIST
# --------------------------------------------------------------------------------------

def load_mnist_digits(cache_dir: str | Path, train: bool = True) -> np.ndarray:
    import torchvision

    ds = torchvision.datasets.MNIST(str(cache_dir), train=train, download=True)
    return ds.data.numpy().astype(np.uint8)  # (N, 28, 28)


def synthetic_digits(n: int = 512, seed: int = 0) -> np.ndarray:
    """Blob 'digits' for offline tests (no download)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:28, :28]
    out = np.zeros((n, 28, 28), dtype=np.uint8)
    for i in range(n):
        img = np.zeros((28, 28))
        for _ in range(3):
            cy, cx, r = rng.uniform(6, 22), rng.uniform(6, 22), rng.uniform(2, 5)
            img += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r))
        out[i] = np.clip(img * 255, 0, 255).astype(np.uint8)
    return out


def colored_moving_mnist(digits: np.ndarray, n_seq: int, frames: int = 20, size: int = 64, n_digits: int = 2,
                         seed: int = 0) -> np.ndarray:
    """(n_seq, frames, size, size, 3) uint8. Each digit gets a random bright colour and
    bounces off the borders with constant speed (the classic Moving-MNIST dynamics)."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n_seq, frames, size, size, 3), dtype=np.uint8)
    lim = size - 28
    for s in range(n_seq):
        canvas = np.zeros((frames, size, size, 3), dtype=np.float32)
        for _ in range(n_digits):
            d = digits[rng.integers(len(digits))].astype(np.float32) / 255.0
            color = rng.uniform(0.35, 1.0, size=3)
            color[rng.integers(3)] = 1.0
            pos = rng.uniform(0, lim, size=2)
            theta = rng.uniform(0, 2 * np.pi)
            speed = rng.uniform(2.0, 4.0)
            vel = np.array([np.sin(theta), np.cos(theta)]) * speed
            for t in range(frames):
                y, x = int(round(pos[0])), int(round(pos[1]))
                patch = d[..., None] * color[None, None, :]
                region = canvas[t, y:y + 28, x:x + 28]
                np.maximum(region, patch, out=region)
                pos = pos + vel
                for k in range(2):
                    if pos[k] < 0:
                        pos[k], vel[k] = -pos[k], -vel[k]
                    elif pos[k] > lim:
                        pos[k], vel[k] = 2 * lim - pos[k], -vel[k]
        out[s] = np.clip(canvas * 255, 0, 255).astype(np.uint8)
    return out


# --------------------------------------------------------------------------------------
# Frame banks
# --------------------------------------------------------------------------------------

def detect_cuts(frames_uint8: np.ndarray, z_thresh: float = 6.0, min_abs: float = 0.08) -> np.ndarray:
    """Indices i such that a cut happens between frame i-1 and i."""
    n = len(frames_uint8)
    diffs = np.zeros(n, dtype=np.float32)
    for s in range(1, n, 4096):
        e = min(n, s + 4096)
        a = frames_uint8[s - 1:e - 1].astype(np.float32) / 255
        b = frames_uint8[s:e].astype(np.float32) / 255
        diffs[s:e] = np.abs(a - b).mean(axis=(1, 2, 3))
    d = diffs[1:]
    med = np.median(d)
    mad = np.median(np.abs(d - med)) + 1e-6
    z = (d - med) / (1.4826 * mad)
    return np.where((z > z_thresh) & (d > min_abs))[0] + 1


class ClipBank:
    """Video frames on GPU + valid clip start indices (clips stay inside segments).

    ``frames``: uint8 (N,H,W,3) with segment ids (N,) - consecutive frames of a segment
    are temporally contiguous.
    """

    def __init__(self, frames: np.ndarray | torch.Tensor, segment_ids: np.ndarray, device="cuda"):
        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(np.ascontiguousarray(frames))
        self.device = torch.device(device)
        self.frames = frames.permute(0, 3, 1, 2).contiguous().to(self.device)
        self.seg = torch.as_tensor(np.asarray(segment_ids), device=self.device)
        self._starts = {}

    def __len__(self) -> int:
        return self.frames.shape[0]

    @staticmethod
    def to_float(x: torch.Tensor) -> torch.Tensor:
        return x.float().div_(127.5).sub_(1.0)

    def valid_starts(self, length: int, stride: int = 1) -> torch.Tensor:
        key = (length, stride)
        if key not in self._starts:
            span = (length - 1) * stride
            n = len(self)
            idx = torch.arange(max(n - span, 0), device=self.device)
            ok = self.seg[idx] == self.seg[idx + span] if span > 0 else torch.ones_like(idx, dtype=torch.bool)
            self._starts[key] = idx[ok]
        return self._starts[key]

    def sample_frames(self, batch: int, xflip: bool = True) -> torch.Tensor:
        idx = torch.randint(len(self), (batch,), device=self.device)
        x = self.to_float(self.frames[idx])
        if xflip:
            flip = torch.rand(batch, device=self.device) < 0.5
            x = torch.where(flip[:, None, None, None], x.flip(3), x)
        return x

    def sample_clips(self, batch: int, length: int, stride: int = 1, xflip: bool = True,
                     generator: torch.Generator | None = None) -> torch.Tensor:
        """(B, T, 3, H, W) float in [-1, 1]."""
        starts = self.valid_starts(length, stride)
        if len(starts) == 0:
            raise ValueError(f"no clip of length {length} (stride {stride}) fits inside a segment")
        pick = starts[torch.randint(len(starts), (batch,), device=self.device, generator=generator)]
        idx = pick[:, None] + torch.arange(length, device=self.device)[None] * stride
        x = self.to_float(self.frames[idx])
        if xflip:
            flip = torch.rand(batch, device=self.device, generator=generator) < 0.5
            x = torch.where(flip[:, None, None, None, None], x.flip(4), x)
        return x

    def fixed_clips(self, n: int, length: int, stride: int = 1, seed: int = 0) -> torch.Tensor:
        starts = self.valid_starts(length, stride)
        g = torch.Generator(device="cpu").manual_seed(seed)
        sel = starts.cpu()[torch.randperm(len(starts), generator=g)[:n]].to(self.device)
        idx = sel[:, None] + torch.arange(length, device=self.device)[None] * stride
        return self.to_float(self.frames[idx])

    @classmethod
    def from_sequences(cls, seqs: np.ndarray, device="cuda") -> "ClipBank":
        """(S, T, H, W, 3) array of independent sequences -> bank with one segment per sequence."""
        s, t = seqs.shape[:2]
        return cls(seqs.reshape(s * t, *seqs.shape[2:]), np.repeat(np.arange(s), t), device)


def frames_to_segments(frames: np.ndarray, test_fraction: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut detection + temporal train/test split. Returns (segment_ids, train_mask, cuts)."""
    cuts = detect_cuts(frames)
    seg = np.zeros(len(frames), dtype=np.int64)
    seg[cuts] = 1
    seg = np.cumsum(seg)
    split = int(len(frames) * (1 - test_fraction))
    train = np.arange(len(frames)) < split
    seg = seg + np.where(train, 0, 10 ** 6)  # never join a clip across the split point
    return seg, train, cuts


def resize_frames_gpu(frames_uint8: torch.Tensor, res: int) -> torch.Tensor:
    """(N,3,H,W) uint8 -> (N,3,res,res) uint8 (antialiased)."""
    x = F.interpolate(frames_uint8.float(), size=(res, res), mode="bicubic", antialias=True, align_corners=False)
    return x.round().clamp(0, 255).to(torch.uint8)
