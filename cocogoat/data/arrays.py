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


def detect_value_range(sample: np.ndarray) -> str:
    """'uint8', 'unit' ([0,1]), 'signed' ([-1,1]) or '255' (float in [0,255])."""
    if sample.dtype == np.uint8:
        return "uint8"
    lo, hi = float(np.nanmin(sample)), float(np.nanmax(sample))
    if lo < -0.01 and hi <= 1.01:
        return "signed"
    if hi <= 1.01:
        return "unit"
    return "255"


def chunk_to_uint8(chunk: np.ndarray, value_range: str) -> np.ndarray:
    if value_range == "uint8":
        return chunk
    a = chunk.astype(np.float32)
    if value_range == "signed":
        a = (a + 1.0) * 127.5
    elif value_range == "unit":
        a = a * 255.0
    return np.clip(np.round(a), 0, 255).astype(np.uint8)
