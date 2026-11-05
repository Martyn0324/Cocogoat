"""Training utilities: EMA, schedules, checkpoint/resume, logging, sample grids.

All run artefacts live in one directory (normally on Google Drive):

    <run_dir>/config.json      resolved configuration
    <run_dir>/log.txt          human-readable log (appended across sessions)
    <run_dir>/metrics.jsonl    one JSON object per logging/eval event
    <run_dir>/status.json      latest progress snapshot (overwritten)
    <run_dir>/ckpt/latest.pt   resumable training state
    <run_dir>/samples/*.png    sample grids
    <run_dir>/error.txt        traceback of the last crash (if any)
    <run_dir>/DONE             written when the stage finished its budget
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import json
import math
import os
import random
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch


# --------------------------------------------------------------------------------------
# Reproducibility / device
# --------------------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_torch(device: str | None = None) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    return torch.device(device)


def autocast(device: torch.device, enabled: bool = True, dtype: torch.dtype = torch.bfloat16):
    if device.type != "cuda" or not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and "cuda" in state:
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception as exc:  # noqa: BLE001 - RNG restore is best effort (device count may differ)
        print(f"[warn] could not restore RNG state: {exc}")


# --------------------------------------------------------------------------------------
# EMA of model weights
# --------------------------------------------------------------------------------------

class EMA:
    """Exponential moving average of a model's parameters (buffers are copied)."""

    def __init__(self, model: torch.nn.Module):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module, beta: float) -> None:
        ema_p = list(self.model.parameters())
        src_p = [p.detach() for p in model.parameters()]
        if ema_p and all(a.dtype == b.dtype and a.device == b.device for a, b in zip(ema_p, src_p)):
            torch._foreach_lerp_(ema_p, src_p, 1.0 - beta)  # one fused launch group instead of one per tensor
        else:
            for p_ema, p in zip(ema_p, src_p):
                p_ema.lerp_(p.to(p_ema.dtype), 1.0 - beta)
        for b_ema, b in zip(self.model.buffers(), model.buffers()):
            b_ema.copy_(b)

    def state_dict(self) -> dict:
        return self.model.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        self.model.load_state_dict(sd)


def ema_beta(batch_size: int, half_life_kimg: float, nimg: int, rampup: float | None = 0.05) -> float:
    """StyleGAN-style EMA decay from a half-life in thousands of images."""
    hl = half_life_kimg * 1000.0
    if rampup is not None:
        hl = min(hl, nimg * rampup)
    if hl <= 0:
        return 0.0
    return 0.5 ** (batch_size / max(hl, 1e-8))


# --------------------------------------------------------------------------------------
# Schedules
# --------------------------------------------------------------------------------------

def cosine_interp(start: float, end: float, progress: float) -> float:
    """Cosine interpolation from ``start`` (progress=0) to ``end`` (progress>=1)."""
    t = min(max(progress, 0.0), 1.0)
    return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * t))


def log_cosine_interp(start: float, end: float, progress: float) -> float:
    """Cosine interpolation in log-space (for quantities spanning decades, e.g. gamma)."""
    if start <= 0 or end <= 0:
        return cosine_interp(start, end, progress)
    return math.exp(cosine_interp(math.log(start), math.log(end), progress))


def set_optimizer_hparams(opt: torch.optim.Optimizer, lr: float | None = None, beta2: float | None = None) -> None:
    for g in opt.param_groups:
        if lr is not None:
            g["lr"] = lr
        if beta2 is not None and "betas" in g:
            g["betas"] = (g["betas"][0], beta2)


# --------------------------------------------------------------------------------------
# Run directory, logging, checkpoints
# --------------------------------------------------------------------------------------

def _jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist() if obj.numel() < 64 else f"tensor{tuple(obj.shape)}"
    return str(obj)


class RunDir:
    """Owns a run directory: logging, status, checkpoints, error capture."""

    def __init__(self, path: str | Path, config: Any | None = None):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "ckpt").mkdir(exist_ok=True)
        (self.path / "samples").mkdir(exist_ok=True)
        self._log = open(self.path / "log.txt", "a", buffering=1)
        self._metrics = open(self.path / "metrics.jsonl", "a", buffering=1)
        if config is not None:
            (self.path / "config.json").write_text(json.dumps(config, indent=1, default=_jsonable))

    # -- logging -------------------------------------------------------------------
    def print(self, *args: Any) -> None:
        msg = " ".join(str(a) for a in args)
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        self._log.write(line + "\n")

    def log_metrics(self, **kv: Any) -> None:
        kv = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **kv}
        self._metrics.write(json.dumps(kv, default=_jsonable) + "\n")

    def write_status(self, **kv: Any) -> None:
        kv = {"updated": time.strftime("%Y-%m-%d %H:%M:%S"), **kv}
        tmp = self.path / "status.json.tmp"
        tmp.write_text(json.dumps(kv, indent=1, default=_jsonable))
        os.replace(tmp, self.path / "status.json")

    def write_json(self, name: str, obj: Any) -> None:
        tmp = self.path / (name + ".tmp")
        tmp.write_text(json.dumps(obj, indent=1, default=_jsonable))
        os.replace(tmp, self.path / name)

    # -- checkpoints ---------------------------------------------------------------
    def save_checkpoint(self, state: dict, name: str = "latest.pt") -> Path:
        """Atomic save: write next to the target, then rename (safe on Drive FUSE)."""
        target = self.path / "ckpt" / name
        tmp = target.with_suffix(target.suffix + ".tmp")
        torch.save(state, tmp)
        os.replace(tmp, target)
        return target

    def load_checkpoint(self, name: str = "latest.pt") -> dict | None:
        target = self.path / "ckpt" / name
        if not target.exists():
            return None
        return torch.load(target, map_location="cpu", weights_only=False)

    def has_checkpoint(self, name: str = "latest.pt") -> bool:
        return (self.path / "ckpt" / name).exists()

    # -- completion / errors -------------------------------------------------------
    @property
    def done(self) -> bool:
        return (self.path / "DONE").exists()

    def mark_done(self, summary: dict | None = None) -> None:
        (self.path / "DONE").write_text(json.dumps(summary or {}, indent=1, default=_jsonable))

    @contextlib.contextmanager
    def capture_errors(self) -> Iterator[None]:
        try:
            yield
        except BaseException as exc:  # noqa: BLE001 - record everything, then re-raise
            tb = traceback.format_exc()
            (self.path / "error.txt").write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n{tb}")
            self.print(f"ERROR: {type(exc).__name__}: {exc}")
            raise

    def close(self) -> None:
        self._log.close()
        self._metrics.close()


# --------------------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------------------

class TrainClock:
    """Tracks *training* wall time across sessions (excludes eval/checkpoint time)."""

    def __init__(self, elapsed: float = 0.0):
        self.elapsed = elapsed
        self._t = None

    def start(self) -> None:
        self._t = time.time()

    def pause(self) -> None:
        if self._t is not None:
            self.elapsed += time.time() - self._t
            self._t = None

    def now(self) -> float:
        return self.elapsed + (time.time() - self._t if self._t is not None else 0.0)


class Every:
    """Fires every ``seconds`` of wall time (first fire after one period)."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self.last = time.time()

    def __call__(self) -> bool:
        if time.time() - self.last >= self.seconds:
            self.last = time.time()
            return True
        return False


# --------------------------------------------------------------------------------------
# Images
# --------------------------------------------------------------------------------------

def to_uint8(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] float NCHW -> uint8 NCHW."""
    return ((x.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1: fixed-size bitmap font
        return ImageFont.load_default()


def annotate_grid(pil, tile_w: int, tile_h: int, ncol: int, row_labels: list[str] | None = None,
                  col_labels: list[str] | None = None, col_colors: list[tuple] | None = None,
                  legend: list[tuple[str, tuple]] | None = None):
    """Add margins with labels to a grid image.

    ``row_labels``: one per grid row (left margin). ``col_labels``: one per column ('' = none, top margin).
    ``col_colors``: RGB per column, drawn as a bar above the column. ``legend``: (text, RGB) pairs on top.
    """
    from PIL import Image, ImageDraw

    fs = max(10, min(18, tile_h // 4))
    font = _font(fs)
    left = 0
    if row_labels:
        left = max(int(ImageDraw.Draw(pil).textlength(t, font=font)) for t in row_labels) + 8
    top = (fs + 6 if legend else 0) + (fs + 4 if col_labels else 0) + (6 if col_colors else 0)
    out = Image.new("RGB", (pil.size[0] + left, pil.size[1] + top), (255, 255, 255))
    out.paste(pil.convert("RGB"), (left, top))
    d = ImageDraw.Draw(out)
    y = 2
    if legend:
        x = left + 2
        for text, colour in legend:
            d.rectangle([x, y + 2, x + fs - 4, y + fs - 2], fill=tuple(colour))
            x += fs
            d.text((x, y), text, fill=(0, 0, 0), font=font)
            x += int(d.textlength(text, font=font)) + fs
        y += fs + 6
    if col_labels:
        for j, t in enumerate(col_labels[:ncol]):
            if t:
                d.text((left + j * tile_w + 2, y), t, fill=(0, 0, 0), font=font)
        y += fs + 4
    if col_colors:
        for j, colour in enumerate(col_colors[:ncol]):
            d.rectangle([left + j * tile_w, y, left + (j + 1) * tile_w - 1, y + 4], fill=tuple(colour))
    if row_labels:
        for i, t in enumerate(row_labels):
            d.text((2, top + i * tile_h + max(0, (tile_h - fs) // 2)), t, fill=(0, 0, 0), font=font)
    return out


def save_grid(images: torch.Tensor, path: str | Path, nrow: int = 8, max_side: int | None = None,
              row_labels: list[str] | None = None, col_labels: list[str] | None = None,
              col_colors: list[tuple] | None = None, legend: list[tuple[str, tuple]] | None = None) -> None:
    """Save an NCHW image batch in [-1, 1] as a PNG grid (optionally labelled, see ``annotate_grid``)."""
    from PIL import Image

    imgs = to_uint8(images).cpu()
    n, c, h, w = imgs.shape
    ncol = min(nrow, n)
    nrows = math.ceil(n / ncol)
    grid = torch.zeros((c, nrows * h, ncol * w), dtype=torch.uint8)
    for i in range(n):
        r, cc = divmod(i, ncol)
        grid[:, r * h:(r + 1) * h, cc * w:(cc + 1) * w] = imgs[i]
    arr = grid.permute(1, 2, 0).numpy()
    if arr.shape[2] == 1:
        arr = arr[:, :, 0]
    pil = Image.fromarray(arr)
    if row_labels or col_labels or col_colors or legend:
        pil = annotate_grid(pil, w, h, ncol, row_labels, col_labels, col_colors, legend)
    if max_side is not None and max(pil.size) > max_side:
        s = max_side / max(pil.size)
        pil = pil.resize((int(pil.size[0] * s), int(pil.size[1] * s)), Image.LANCZOS)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pil.save(path)


def save_video_gif(videos: torch.Tensor, path: str | Path, nrow: int = 4, fps: float = 8.0,
                   borders: torch.Tensor | None = None, border: int = 2, max_side: int = 1024,
                   row_labels: list[str] | None = None, captions: list[str] | None = None,
                   legend: list[tuple[str, tuple]] | None = None) -> None:
    """Save videos as one animated GIF in which every frame is a grid of the videos' frames.

    ``videos``: (N, T, C, H, W) in [-1, 1]. ``borders``: optional (N, T, 3) uint8 RGB colours of a
    ``border``-pixel frame drawn around each tile (e.g. to mark real vs generated frames).
    ``row_labels``: one per grid row; ``captions``: one text per time step (e.g. "t=5 generated");
    ``legend``: (text, RGB) pairs shown on top.
    """
    from PIL import Image

    n, t, c, h, w = videos.shape
    v = to_uint8(videos.flatten(0, 1)).view(n, t, c, h, w).cpu()
    if c == 1:
        v = v.expand(-1, -1, 3, -1, -1)
    if borders is not None:
        b = border
        canvas = borders.to(torch.uint8).cpu()[:, :, :, None, None].expand(n, t, 3, h + 2 * b, w + 2 * b).clone()
        canvas[:, :, :, b:b + h, b:b + w] = v[:, :, :3]
        v, h, w = canvas, h + 2 * b, w + 2 * b
    ncol = min(nrow, n)
    nrows = math.ceil(n / ncol)
    grid = torch.zeros((t, 3, nrows * h, ncol * w), dtype=torch.uint8)
    for i in range(n):
        r, cc = divmod(i, ncol)
        grid[:, :, r * h:(r + 1) * h, cc * w:(cc + 1) * w] = v[i, :, :3]
    frames = []
    for k, f in enumerate(grid):
        pil = Image.fromarray(f.permute(1, 2, 0).numpy())
        if row_labels or captions or legend:
            leg = list(legend or [])
            if captions:
                leg = leg + [(captions[k], (255, 255, 255))]
            pil = annotate_grid(pil, w, h, ncol, row_labels, None, None, leg or None)
        frames.append(pil)
    side = max(frames[0].size)
    if side > max_side:
        s = max_side / side
        size = (max(1, int(frames[0].size[0] * s)), max(1, int(frames[0].size[1] * s)))
        frames = [f.resize(size, Image.LANCZOS) for f in frames]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=max(20, int(round(1000 / fps))), loop=0)


def save_images(images: torch.Tensor, folder: str | Path, prefix: str = "img") -> None:
    from PIL import Image

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    imgs = to_uint8(images).cpu().permute(0, 2, 3, 1).numpy()
    for i, arr in enumerate(imgs):
        Image.fromarray(arr).save(folder / f"{prefix}_{i:04d}.png")


def grad_norm(params: Callable[[], Iterator[torch.nn.Parameter]] | Iterator[torch.nn.Parameter]) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().float().square().sum())
    return math.sqrt(total)


def describe_env() -> dict:
    info = {"python": sys.version.split()[0], "torch": torch.__version__}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, gpu_mem_gb=round(p.total_memory / 2 ** 30, 1))
    return info


def copy_file(src: str | Path, dst: str | Path) -> None:
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
