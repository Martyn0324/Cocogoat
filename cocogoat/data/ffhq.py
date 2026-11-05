"""FFHQ-1024 from the Hugging Face WebDataset mirror ``gaunernst/ffhq-1024-wds``.

70 tar shards (~1000 WebP 1024x1024 images each, ~64 GB). Shards are downloaded once to the
Colab disk, then decoded in parallel (one process per shard) straight into two uint8
memory-mapped arrays: every image at ``low_res`` (for the base GAN) and the first
``keep_hr`` images at full 1024 resolution (for the super-resolution stage).
"""

from __future__ import annotations

import io
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = "gaunernst/ffhq-1024-wds"
IMG_EXT = (".webp", ".png", ".jpg", ".jpeg")


def download_shards(cache_dir: str | Path, max_shards: int | None = None, repo: str = REPO) -> list[Path]:
    from huggingface_hub import HfApi, snapshot_download

    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.endswith(".tar"))
    if max_shards:
        files = files[:max_shards]
    local = Path(cache_dir) / "ffhq_wds"
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(local), allow_patterns=files, max_workers=16)
    return [local / f for f in files]


def _count(path: Path) -> list[str]:
    with tarfile.open(path) as tf:
        return [m.name for m in tf.getmembers() if m.isfile() and m.name.lower().endswith(IMG_EXT)]


def _decode_shard(args) -> int:
    path, names, start, low_path, low_res, hr_path, keep_hr, n_total = args
    from PIL import Image

    low = np.load(low_path, mmap_mode="r+")
    hr = np.load(hr_path, mmap_mode="r+") if hr_path is not None else None
    wanted = {n: start + i for i, n in enumerate(names)}
    done = 0
    with tarfile.open(path) as tf:
        for m in tf:
            if m.name not in wanted:
                continue
            idx = wanted[m.name]
            with Image.open(io.BytesIO(tf.extractfile(m).read())) as im:
                im = im.convert("RGB")
                if hr is not None and idx < keep_hr:
                    arr = np.asarray(im)
                    if arr.shape[:2] != hr.shape[1:3]:
                        arr = np.asarray(im.resize(hr.shape[1:3][::-1], Image.BICUBIC))
                    hr[idx] = arr
                low[idx] = np.asarray(im.resize((low_res, low_res), Image.LANCZOS, reducing_gap=3.0))
            done += 1
    low.flush()
    if hr is not None:
        hr.flush()
    return done


def decode_shards(shards: list[Path], out_dir: str | Path, low_res: int = 256, keep_hr: int = 15000,
                  hr_res: int = 1024, workers: int | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Decode shards into ``ffhq_{low_res}.npy`` and ``ffhq_hr{hr_res}.npy`` (cached)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ns = f"{len(shards)}s"  # shard count in the names: a --max_shards subset never masquerades as the full set
    low_path = out_dir / f"ffhq_{low_res}_{ns}.npy"
    hr_path = out_dir / f"ffhq_hr{hr_res}_{keep_hr}_{ns}.npy" if keep_hr > 0 else None
    done_flag = out_dir / f"ffhq_{low_res}_{keep_hr}_{ns}.done"
    if done_flag.exists():
        return np.load(low_path, mmap_mode="r"), (np.load(hr_path, mmap_mode="r") if hr_path else None)
    workers = workers or min(len(shards), os.cpu_count() or 8)
    with ProcessPoolExecutor(workers) as ex:
        names = list(ex.map(_count, shards))
    counts = [len(n) for n in names]
    n_total = sum(counts)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(int)
    np.lib.format.open_memmap(low_path, mode="w+", dtype=np.uint8, shape=(n_total, low_res, low_res, 3)).flush()
    n_hr = min(keep_hr, n_total)
    if hr_path is not None:
        np.lib.format.open_memmap(hr_path, mode="w+", dtype=np.uint8, shape=(n_hr, hr_res, hr_res, 3)).flush()
    jobs = [(p, nm, int(s), str(low_path), low_res, str(hr_path) if hr_path else None, n_hr, n_total)
            for p, nm, s in zip(shards, names, starts)]
    with ProcessPoolExecutor(workers) as ex:
        decoded = sum(ex.map(_decode_shard, jobs))
    assert decoded == n_total, (decoded, n_total)
    done_flag.write_text(str(n_total))
    return np.load(low_path, mmap_mode="r"), (np.load(hr_path, mmap_mode="r") if hr_path else None)


def make_fake_shards(out_dir: str | Path, n_shards: int = 2, per_shard: int = 6, res: int = 128) -> list[Path]:
    """Synthetic WebDataset shards for offline tests."""
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    paths = []
    for s in range(n_shards):
        p = out_dir / f"{s:05d}.tar"
        with tarfile.open(p, "w") as tf:
            for i in range(per_shard):
                yy, xx = np.mgrid[:res, :res] / res
                img = np.stack([np.sin(6 * xx + rng.random() * 6), np.cos(5 * yy + rng.random() * 6),
                                np.sin(4 * (xx + yy))], -1) * 0.5 + 0.5
                buf = io.BytesIO()
                Image.fromarray((img * 255).astype(np.uint8)).save(buf, format="WEBP", quality=90)
                data = buf.getvalue()
                info = tarfile.TarInfo(name=f"{s * per_shard + i:05d}.webp")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
        paths.append(p)
    return paths
