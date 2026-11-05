"""Inter4K (1,000 UHD 60 fps clips of 5 s; CC BY-NC 4.0) from the HF mirror
``ermu2001/Inter4kPllavaRecaption`` (files under DATAS/Recaption/Inter4K/60fps/UHD/).

Every clip is decoded once (one process per clip): frames are temporally subsampled
(``stride``), center-cropped to a square and resized to ``low_res`` for the latent models;
every ``hr_every``-th kept frame is also stored at ``hr_res`` (native 4K source => genuine
1024 px detail) for the super-resolution stage.  Results are uint8 memory maps plus a
segment id per low-res frame (one segment per clip, so clips never straddle videos).
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = "ermu2001/Inter4kPllavaRecaption"
SUBDIR = "DATAS/Recaption/Inter4K/60fps/UHD"


def download_clips(cache_dir: str | Path, max_clips: int | None = None, repo: str = REPO) -> list[Path]:
    from huggingface_hub import HfApi, snapshot_download

    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset")
                   if f.startswith(SUBDIR) and f.endswith(".mp4"))
    files = sorted(files, key=lambda f: int(Path(f).stem) if Path(f).stem.isdigit() else f)
    if max_clips:
        files = files[:max_clips]
    local = Path(cache_dir) / "inter4k"
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(local), allow_patterns=files, max_workers=16)
    return [local / f for f in files]


def _square(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    s = min(h, w)
    return frame[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]


def _probe(path: str) -> int:
    import cv2

    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(n, 0)


def _decode_clip(args) -> tuple[int, int, int]:
    (path, clip_idx, start, n_keep, stride, low_path, low_res, hr_path, hr_start, n_hr, hr_every, hr_res) = args
    import cv2

    cv2.setNumThreads(1)
    low = np.load(low_path, mmap_mode="r+")
    hr = np.load(hr_path, mmap_mode="r+") if hr_path else None
    cap = cv2.VideoCapture(path)
    kept, hr_kept, fidx = 0, 0, 0
    last = None
    while kept < n_keep:
        ok, frame = cap.read()
        if not ok:
            break
        if fidx % stride == 0:
            sq = _square(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            last = cv2.resize(sq, (low_res, low_res), interpolation=cv2.INTER_AREA)
            low[start + kept] = last
            if hr is not None and kept % hr_every == 0 and hr_kept < n_hr:
                interp = cv2.INTER_AREA if sq.shape[0] >= hr_res else cv2.INTER_CUBIC
                hr[hr_start + hr_kept] = cv2.resize(sq, (hr_res, hr_res), interpolation=interp)
                hr_kept += 1
            kept += 1
        fidx += 1
    cap.release()
    decoded = kept
    while kept < n_keep and last is not None:  # pad short clips by repeating the last frame
        low[start + kept] = last
        kept += 1
    low.flush()
    if hr is not None:
        hr.flush()
    return clip_idx, hr_kept, decoded


def decode_clips(clips: list[Path], out_dir: str | Path, low_res: int = 256, stride: int = 4, frames_per_clip: int = 72,
                 hr_res: int = 1024, hr_every: int = 8, workers: int | None = None):
    """Returns (low uint8 (N,low,low,3), segment ids (N,), hr uint8 (M,hr,hr,3), hr clip ids (M,))."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{len(clips)}c_{low_res}_{stride}_{frames_per_clip}_{hr_res}_{hr_every}"
    low_path, hr_path = out_dir / f"inter4k_low_{tag}.npy", out_dir / f"inter4k_hr_{tag}.npy"
    meta_path, done = out_dir / f"inter4k_meta_{tag}.npz", out_dir / f"inter4k_{tag}.done"
    if done.exists():
        with np.load(meta_path) as m:
            return (np.load(low_path, mmap_mode="r"), m["segments"], np.load(hr_path, mmap_mode="r"), m["hr_clip"])
    workers = workers or min(len(clips), os.cpu_count() or 8)
    with ProcessPoolExecutor(workers) as ex:
        counts = list(ex.map(_probe, [str(c) for c in clips]))
    keep = [i for i, n in enumerate(counts) if n >= stride * 8]  # drop unreadable / tiny clips
    clips = [clips[i] for i in keep]
    n_hr_per = (frames_per_clip + hr_every - 1) // hr_every
    n_low = len(clips) * frames_per_clip
    n_hr = len(clips) * n_hr_per
    np.lib.format.open_memmap(low_path, mode="w+", dtype=np.uint8, shape=(n_low, low_res, low_res, 3)).flush()
    np.lib.format.open_memmap(hr_path, mode="w+", dtype=np.uint8, shape=(n_hr, hr_res, hr_res, 3)).flush()
    jobs = [(str(c), i, i * frames_per_clip, frames_per_clip, stride, str(low_path), low_res, str(hr_path),
             i * n_hr_per, n_hr_per, hr_every, hr_res) for i, c in enumerate(clips)]
    with ProcessPoolExecutor(workers) as ex:
        results = list(ex.map(_decode_clip, jobs))
    segments = np.repeat(np.arange(len(clips)), frames_per_clip)
    hr_clip = np.repeat(np.arange(len(clips)), n_hr_per)
    bad = []
    for clip_idx, hr_kept, decoded in results:
        if hr_kept < n_hr_per:  # HR slots that were never written: mark invalid (-1)
            hr_clip[clip_idx * n_hr_per + hr_kept:(clip_idx + 1) * n_hr_per] = -1
        if decoded < 8:  # undecodable clip: drop it entirely
            bad.append(clip_idx)
    if bad:
        keep_frames = ~np.isin(segments, bad)
        keep_hr = ~np.isin(np.repeat(np.arange(len(clips)), n_hr_per), bad)
        low_arr = np.load(low_path, mmap_mode="r")
        hr_arr = np.load(hr_path, mmap_mode="r")
        np.save(low_path.with_suffix(".tmp.npy"), low_arr[keep_frames])
        np.save(hr_path.with_suffix(".tmp.npy"), hr_arr[keep_hr])
        del low_arr, hr_arr
        os.replace(low_path.with_suffix(".tmp.npy"), low_path)
        os.replace(hr_path.with_suffix(".tmp.npy"), hr_path)
        remap = {old: new for new, old in enumerate(c for c in range(len(clips)) if c not in set(bad))}
        segments = np.array([remap[s_] for s_ in segments[keep_frames]])
        hr_clip = np.array([remap.get(h, -1) if h >= 0 else -1 for h in hr_clip[keep_hr]])
        clips = [c for i, c in enumerate(clips) if i not in set(bad)]
        print(f"dropped {len(bad)} undecodable clips", flush=True)
    np.savez(meta_path, segments=segments, hr_clip=hr_clip, names=np.array([c.name for c in clips]))
    done.write_text(str(len(clips)))
    return np.load(low_path, mmap_mode="r"), segments, np.load(hr_path, mmap_mode="r"), hr_clip


def make_fake_clips(out_dir: str | Path, n: int = 3, frames: int = 40, size: tuple[int, int] = (96, 64)) -> list[Path]:
    """Tiny synthetic mp4 clips (moving gradients) for offline tests."""
    import cv2

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    yy, xx = np.mgrid[:size[1], :size[0]]
    for i in range(n):
        p = out_dir / f"{i + 1}.mp4"
        w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 30, size)
        for t in range(frames):
            img = np.stack([np.sin((xx + 3 * t) / (6 + i)) * 0.5 + 0.5, np.cos((yy + 2 * t) / 7) * 0.5 + 0.5,
                            np.full_like(xx, (i + 1) / (n + 1), dtype=float)], -1)
            w.write((img * 255).astype(np.uint8))
        w.release()
        paths.append(p)
    return paths
