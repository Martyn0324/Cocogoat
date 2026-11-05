"""Helpers shared by the task entry points (config overrides, data, evaluation)."""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

LOCAL_CACHE = Path("/content/cache") if Path("/content").exists() else Path.home() / ".cache" / "cocogoat"
DRIVE_ROOT = Path("/content/drive/MyDrive/Cocogoat")
USER_DATA = Path("/content/drive/MyDrive/Python/Projects/GANs")


def apply_overrides(cfg: Any, overrides: dict) -> Any:
    """Recursively apply a dict of overrides onto (nested) dataclasses."""
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise KeyError(f"unknown config key {k!r} for {type(cfg).__name__}")
        cur = getattr(cfg, k)
        if dataclasses.is_dataclass(cur) and isinstance(v, dict):
            apply_overrides(cur, v)
        else:
            if isinstance(cur, tuple) and isinstance(v, list):
                v = tuple(v)
            setattr(cfg, k, v)
    return cfg


def parse_override(s: str | None) -> dict:
    return json.loads(s) if s else {}


def rotate_old_error(run_dir: Path) -> None:
    err = Path(run_dir) / "error.txt"
    if err.exists():
        err.rename(Path(run_dir) / f"error_{time.strftime('%Y%m%d_%H%M%S')}.txt")


class RandomFeatures(torch.nn.Module):
    """Deterministic random-projection 'feature extractor' - ONLY for CPU smoke tests."""

    def __init__(self, dim: int = 64, seed: int = 0):
        super().__init__()
        self.dim = dim
        self.seed = seed
        self.proj = None

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.adaptive_avg_pool2d(x.float(), 8).flatten(1)
        if self.proj is None or self.proj.shape[0] != x.shape[1]:
            g = torch.Generator().manual_seed(self.seed)
            self.proj = torch.randn(x.shape[1], self.dim, generator=g).to(x.device)
        return torch.tanh(x @ self.proj.to(x.device))


def make_feature_fn(kind: str, device: torch.device) -> Callable[[torch.Tensor], torch.Tensor]:
    if kind == "inception":
        from cocogoat.metrics import InceptionFeatures

        return InceptionFeatures(device)
    if kind == "random":
        return RandomFeatures().to(device)
    raise ValueError(kind)


def sample_labels(prior: torch.Tensor | None, n: int, device: torch.device) -> torch.Tensor | None:
    if prior is None:
        return None
    return torch.multinomial(prior.to(device), n, replacement=True)


def summarize_metrics_md(title: str, metrics: dict, extra: list[str] | None = None) -> str:
    lines = [f"# {title}", "", "| metric | value |", "|---|---|"]
    for k, v in metrics.items():
        if isinstance(v, float):
            v = f"{v:.4g}"
        lines.append(f"| {k} | {v} |")
    if extra:
        lines += [""] + extra
    return "\n".join(lines) + "\n"


def np_load_or(path: Path, fn: Callable[[], np.ndarray]) -> np.ndarray:
    if path.exists():
        return np.load(path)
    arr = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return arr
