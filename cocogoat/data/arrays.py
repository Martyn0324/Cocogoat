"""Memory-friendly access to large .npy / .npz arrays.

The user's datasets are big (e.g. ``Complete_dataset.npy`` = 45,935 x 100 x 100 x 3
float64 = 11 GB), so we never materialise them in RAM as floating point: arrays are
read in chunks (memory-mapped .npy, or streamed out of the zip for .npz members) and
converted to uint8 on the fly.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Iterator

import numpy as np


def npy_header_from_stream(f) -> tuple[np.dtype, tuple[int, ...], bool]:
    version = np.lib.format.read_magic(f)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
    else:
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
    return np.dtype(dtype), tuple(shape), fortran


def npz_members(path: str | Path) -> dict[str, dict]:
    """Header-only description of every array inside an .npz (no data is loaded)."""
    out = {}
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if not info.filename.endswith(".npy"):
                continue
            with zf.open(info) as f:
                dtype, shape, fortran = npy_header_from_stream(f)
            out[info.filename[:-4]] = {"dtype": str(dtype), "shape": list(shape), "fortran_order": fortran,
                                       "compressed": info.compress_type != zipfile.ZIP_STORED,
                                       "stored_bytes": info.compress_size, "raw_bytes": info.file_size}
    return out


def iter_npz_member(path: str | Path, key: str, chunk_items: int = 1024) -> Iterator[np.ndarray]:
    """Stream an .npz member in chunks along axis 0 (C-order arrays only)."""
    with zipfile.ZipFile(path) as zf:
        with zf.open(key + ".npy") as f:
            dtype, shape, fortran = npy_header_from_stream(f)
            if fortran:
                raise ValueError("fortran-ordered arrays are not supported for streaming")
            item = int(np.prod(shape[1:])) if len(shape) > 1 else 1
            n = shape[0]
            for start in range(0, n, chunk_items):
                cnt = min(chunk_items, n - start)
                buf = f.read(cnt * item * dtype.itemsize)
                yield np.frombuffer(buf, dtype=dtype).reshape((cnt,) + tuple(shape[1:]))


def iter_array(path: str | Path, key: str | None = None, chunk_items: int = 1024) -> tuple[tuple, np.dtype, Iterator[np.ndarray]]:
    """(shape, dtype, chunk iterator) for an .npy file or an .npz member."""
    path = Path(path)
    if path.suffix == ".npy":
        arr = np.load(path, mmap_mode="r")

        def gen():
            for s in range(0, arr.shape[0], chunk_items):
                yield np.asarray(arr[s:s + chunk_items])

        return arr.shape, arr.dtype, gen()
    members = npz_members(path)
    if key is None:
        cands = [k for k, v in members.items() if len(v["shape"]) >= 3]
        key = max(cands or list(members), key=lambda k: members[k]["raw_bytes"])
    meta = members[key]
    return tuple(meta["shape"]), np.dtype(meta["dtype"]), iter_npz_member(path, key, chunk_items)


def channels_last(chunk: np.ndarray) -> np.ndarray:
    """(N,H,W) / (N,C,H,W) / (N,H,W,C) -> (N,H,W,C) (same layout rule as ``to_uint8_nhwc``)."""
    if chunk.ndim == 3:
        chunk = chunk[..., None]
    if chunk.ndim == 4 and chunk.shape[1] in (1, 3, 4) and chunk.shape[-1] not in (1, 3, 4):
        chunk = chunk.transpose(0, 2, 3, 1)
    return chunk


def scan_value_range(path: str | Path, key: str | None = None, chunk_items: int = 1024,
                     max_chunks: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel (min, max) over the whole array.

    The first frames of a dataset are not representative (a music video starts with black
    frames), so the range is measured over the whole file: every chunk of an .npz member
    (which can only be streamed), or ``max_chunks`` evenly spaced chunks of a memory-mapped .npy.
    """
    path = Path(path)
    if path.suffix == ".npy":
        arr = np.load(path, mmap_mode="r")
        starts = np.unique(np.linspace(0, max(arr.shape[0] - 1, 0), max_chunks).astype(np.int64))
        chunks = (np.asarray(arr[s:s + chunk_items]) for s in starts)
    else:
        chunks = iter_array(path, key, chunk_items)[2]
    lo = hi = None
    for ch in chunks:
        ch = channels_last(ch.reshape(-1, *ch.shape[2:]) if ch.ndim == 5 else ch)
        flat = ch.reshape(-1, ch.shape[-1]).astype(np.float64)
        c_lo, c_hi = np.nanmin(flat, 0), np.nanmax(flat, 0)
        lo = c_lo if lo is None else np.minimum(lo, c_lo)
        hi = c_hi if hi is None else np.maximum(hi, c_hi)
    if lo is None:
        raise ValueError(f"{path} contains no data")
    return lo, hi


def uint8_conversion(lo: np.ndarray, hi: np.ndarray) -> dict:
    """How to map float pixels to [0, 255]: ``u8 = round(x * scale + offset)`` per channel.

    Standard conventions are recognised from the global range: [0, 1], [-1, 1] and [0, 255].
    Anything else (e.g. ImageNet-standardised pixels, roughly [-2.1, 2.6]) is mapped per channel
    from its own min/max, which inverts any per-channel affine normalisation of full-range images.
    """
    lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
    glo, ghi = float(lo.min()), float(hi.max())
    ones = np.ones_like(lo)
    if glo >= -0.01 and ghi <= 1.01:
        return {"name": "unit [0,1]", "scale": 255.0 * ones, "offset": 0.0 * ones}
    if glo >= -1.01 and ghi <= 1.01:
        return {"name": "signed [-1,1]", "scale": 127.5 * ones, "offset": 127.5 * ones}
    if glo >= -0.5 and 10.0 < ghi <= 255.5:
        return {"name": "[0,255]", "scale": ones, "offset": 0.0 * ones}
    span = np.maximum(hi - lo, 1e-8)
    return {"name": "per-channel min/max", "scale": 255.0 / span, "offset": -lo * 255.0 / span}


def chunk_to_uint8(chunk: np.ndarray, conversion: dict | None) -> np.ndarray:
    """Channels-last chunk -> uint8 using a ``uint8_conversion`` (None: already uint8)."""
    if chunk.dtype == np.uint8 or conversion is None:
        return chunk
    a = chunk.astype(np.float32) * conversion["scale"].astype(np.float32) + conversion["offset"].astype(np.float32)
    return np.clip(np.round(a), 0, 255).astype(np.uint8)
