"""Image datasets.

Small/medium datasets are decoded once into a uint8 array and kept *on the GPU*
(``GPUImageBank``): sampling a batch is then a single gather, with no DataLoader.
At 64x64, 100k images take 1.2 GB; at 256x256, 70k images take 13.8 GB - both fit
easily on a 96 GB card.
"""

from __future__ import annotations

import io
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_CACHE = Path(os.environ.get("COCOGOAT_CACHE", "/content/cache"))


# --------------------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------------------

def localize(path: str | Path, cache_dir: str | Path = DEFAULT_CACHE) -> Path:
    """Copy a (Drive) file to the local disk once; random access on Drive FUSE is slow."""
    path = Path(path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / path.name
    if not local.exists() or local.stat().st_size != path.stat().st_size:
        print(f"copying {path} -> {local} ({path.stat().st_size / 2**20:.0f} MB)", flush=True)
        tmp = local.with_suffix(local.suffix + ".part")
        shutil.copyfile(path, tmp)
        os.replace(tmp, local)
    return local


def to_uint8_nhwc(arr: np.ndarray) -> np.ndarray:
    """Normalise an image array to uint8 NHWC with 3 channels.

    Accepts NHWC or NCHW, uint8 or float in [0,1] / [-1,1] / [0,255], grayscale or RGB(A).
    """
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[..., None]
    if arr.ndim != 4:
        raise ValueError(f"expected 4-D image array, got shape {arr.shape}")
    if arr.shape[1] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = arr.transpose(0, 2, 3, 1)
    if arr.dtype != np.uint8:
        sample = arr[: min(len(arr), 256)].astype(np.float32)
        lo, hi = float(sample.min()), float(sample.max())
        a = arr.astype(np.float32)
        if lo < -0.01 and hi <= 1.01:
            a = (a + 1.0) * 127.5
        elif hi <= 1.01:
            a = a * 255.0
        arr = np.clip(np.round(a), 0, 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return np.ascontiguousarray(arr)


def resize_uint8(arr: np.ndarray, res: int, batch: int = 2048, device: str | torch.device = "cpu") -> np.ndarray:
    """Center-crop to square and resize (antialiased bicubic) a uint8 NHWC array."""
    n, h, w, _ = arr.shape
    if h == res and w == res:
        return arr
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    out = np.empty((n, res, res, 3), dtype=np.uint8)
    for i in range(0, n, batch):
        x = torch.from_numpy(np.array(arr[i:i + batch, top:top + s, left:left + s])).to(device)
        x = x.permute(0, 3, 1, 2).float()
        x = F.interpolate(x, size=(res, res), mode="bicubic", antialias=True, align_corners=False)
        out[i:i + batch] = x.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    return out


def cache_looks_valid(arr: np.ndarray, n_samples: int = 64) -> bool:
    """False for a uint8 cache whose pixels never exceed 15 (frames sampled across the file):
    the signature of float data converted with the wrong value range (near-black images)."""
    if len(arr) == 0:
        return False
    idx = np.unique(np.linspace(0, len(arr) - 1, n_samples).astype(np.int64))
    return int(np.asarray(arr[idx]).max()) > 15


def load_array_images(path: str | Path, key: str | None = None, cache_dir: str | Path | None = DEFAULT_CACHE,
                      drive_cache: str | Path | None = None, chunk: int = 1024) -> np.ndarray:
    """Load images from .npy / .npz (any common layout/dtype) as a uint8 NHWC array.

    Float arrays are converted chunk by chunk (never fully materialised in RAM) and the
    uint8 result is cached as ``<stem>[_key]_u8.npy`` in ``cache_dir`` (local disk) and,
    if given, ``drive_cache`` (persistent across Colab sessions).  Returns a read-only
    memory map of the cached uint8 file.  5-D arrays (videos) are flattened to frames.
    """
    from cocogoat.data.arrays import channels_last, chunk_to_uint8, iter_array, scan_value_range, uint8_conversion

    path = Path(path)
    tag = f"{path.stem}{'_' + key if key else ''}_u8.npy"
    for cdir in (cache_dir, drive_cache):
        if cdir is not None and (Path(cdir) / tag).exists():
            arr = np.load(Path(cdir) / tag, mmap_mode="r")
            if not cache_looks_valid(arr):
                print(f"{Path(cdir) / tag}: pixel values never exceed 15 - a cache written by an older, "
                      f"faulty range conversion; rebuilding it", flush=True)
                (Path(cdir) / tag).unlink()
                continue
            if cdir == drive_cache and cache_dir is not None:
                local = localize(Path(cdir) / tag, cache_dir)
                arr = np.load(local, mmap_mode="r")
            return arr
    target_dir = Path(cache_dir if cache_dir is not None else path.parent)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / tag
    src, tmp_copy = path, None
    if str(path).startswith("/content/drive") and cache_dir is not None:
        # sequential copy from Drive FUSE is much faster than strided memmap reads
        src = tmp_copy = localize(path, Path(cache_dir) / "_src")
    shape, dtype, chunks = iter_array(src, key, chunk_items=chunk)
    n_items = int(np.prod(shape[:2])) if len(shape) == 5 else shape[0]
    conversion = None
    if dtype != np.uint8:
        lo, hi = scan_value_range(src, key, chunk_items=chunk)
        conversion = uint8_conversion(lo, hi)
        print(f"{path.name}: shape={shape} dtype={dtype} per-channel min={np.round(lo, 4).tolist()} "
              f"max={np.round(hi, 4).tolist()} -> {conversion['name']} conversion", flush=True)
    else:
        print(f"{path.name}: shape={shape} dtype=uint8", flush=True)
    out, i = None, 0
    for ch in chunks:
        if ch.ndim == 5:
            ch = ch.reshape(-1, *ch.shape[2:])
        u8 = to_uint8_nhwc(chunk_to_uint8(channels_last(ch), conversion))
        if out is None:
            tmp = target.with_suffix(".npy.part")
            out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(n_items,) + u8.shape[1:])
        out[i:i + len(u8)] = u8
        i += len(u8)
    assert out is not None and i == n_items, (i, n_items)
    out.flush()
    if not cache_looks_valid(out):
        raise ValueError(f"{path.name}: after conversion no sampled pixel exceeds 15 of 255 - the images are "
                         f"(almost) black; check the array's contents (dtype {dtype})")
    del out
    import os as _os

    _os.replace(target.with_suffix(".npy.part"), target)
    if tmp_copy is not None:
        tmp_copy.unlink(missing_ok=True)
    if drive_cache is not None:
        Path(drive_cache).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(target, Path(drive_cache) / tag)
    return np.load(target, mmap_mode="r")


# --------------------------------------------------------------------------------------
# GPU-resident dataset
# --------------------------------------------------------------------------------------

def to_device_nchw(images, device, chunk: int = 1024) -> torch.Tensor:
    """uint8 NHWC (numpy / memmap / tensor) -> contiguous uint8 NCHW on ``device``, copied in chunks so a
    40+ GB memmap never needs a second full-size host copy."""
    n, h, w, c = images.shape
    assert c == 3, images.shape
    out = torch.empty((n, 3, h, w), dtype=torch.uint8, device=device)
    for i in range(0, n, chunk):
        part = images[i:i + chunk]
        part = torch.from_numpy(np.array(part)) if not torch.is_tensor(part) else part
        out[i:i + len(part)] = part.to(device).permute(0, 3, 1, 2)
    return out

class GPUImageBank:
    """uint8 images (+ optional labels / condition vectors) resident on one device."""

    def __init__(self, images: np.ndarray | torch.Tensor, labels: np.ndarray | torch.Tensor | None = None,
                 device: str | torch.device = "cuda", xflip: bool = True):
        self.device = torch.device(device)
        self.images = to_device_nchw(images, self.device)  # N,3,H,W uint8
        if labels is not None and isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)
        self.labels = labels.to(self.device) if labels is not None else None
        self.xflip = xflip

    def __len__(self) -> int:
        return self.images.shape[0]

    @property
    def resolution(self) -> int:
        return self.images.shape[-1]

    @staticmethod
    def to_float(x_uint8: torch.Tensor) -> torch.Tensor:
        return x_uint8.float().div_(127.5).sub_(1.0)

    def sample(self, batch: int, generator: torch.Generator | None = None):
        idx = torch.randint(len(self), (batch,), device=self.device, generator=generator)
        x = self.to_float(self.images[idx])
        if self.xflip:
            flip = torch.rand(batch, device=self.device, generator=generator) < 0.5
            x = torch.where(flip[:, None, None, None], x.flip(3), x)
        y = self.labels[idx] if self.labels is not None else None
        return x, y

    def iterate(self, batch: int, limit: int | None = None):
        n = len(self) if limit is None else min(limit, len(self))
        for i in range(0, n, batch):
            x = self.to_float(self.images[i:min(i + batch, n)])
            y = self.labels[i:min(i + batch, n)] if self.labels is not None else None
            yield x, y

    def label_prior(self) -> torch.Tensor | None:
        """Empirical class distribution (for sampling labels of fakes)."""
        if self.labels is None or self.labels.dtype != torch.long:
            return None
        counts = torch.bincount(self.labels, minlength=int(self.labels.max()) + 1).float()
        return counts / counts.sum()


# --------------------------------------------------------------------------------------
# Specific datasets
# --------------------------------------------------------------------------------------

def load_tinyimagenet(cache_dir: str | Path = DEFAULT_CACHE, split: str = "train") -> tuple[np.ndarray, np.ndarray]:
    """Tiny-ImageNet (200 classes, 64x64) from the Hugging Face mirror ``zh-plus/tiny-imagenet``."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"tinyimagenet_{split}.npz"
    if cached.exists():
        with np.load(cached) as z:
            return z["images"], z["labels"]
    import pandas as pd
    from huggingface_hub import hf_hub_download, list_repo_files
    from PIL import Image

    files = sorted(f for f in list_repo_files("zh-plus/tiny-imagenet", repo_type="dataset")
                   if f.endswith(".parquet") and (split in f or (split == "valid" and "val" in f)))
    if not files:
        raise FileNotFoundError(f"no parquet files for split {split!r} in zh-plus/tiny-imagenet")
    images, labels = [], []
    for f in files:
        df = pd.read_parquet(hf_hub_download("zh-plus/tiny-imagenet", f, repo_type="dataset"))
        img_col = [c for c in df.columns if "image" in c.lower()][0]
        lab_col = [c for c in df.columns if "label" in c.lower()][0]
        for cell, lab in zip(df[img_col], df[lab_col]):
            raw = cell["bytes"] if isinstance(cell, dict) else cell
            with Image.open(io.BytesIO(raw)) as im:
                images.append(np.asarray(im.convert("RGB").resize((64, 64), Image.BICUBIC)))
            labels.append(int(lab))
    images = np.stack(images).astype(np.uint8)
    labels = np.asarray(labels, dtype=np.int64)
    np.savez(cached, images=images, labels=labels)
    return images, labels
