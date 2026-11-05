"""Helpers shared by the task entry points (config overrides, data, evaluation)."""

from __future__ import annotations

import dataclasses
import json
import shutil
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


def fresh_smoke_run(path: str | Path, root: str | Path | None = None) -> None:
    """Smoke tests start from scratch: delete a previous smoke run (only folders named ``*_smoke``)."""
    p = Path(path)
    if Path(root if root is not None else p).name.endswith("_smoke") and p.exists():
        shutil.rmtree(p)
        print(f"smoke test: removed the previous smoke run in {p}", flush=True)


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


def resume_or_skip(rd, requested_hours: float, force: bool = False) -> bool:
    """True if the stage should be skipped.

    A stage marked DONE is re-opened (DONE removed) when a *larger* time budget is requested,
    so ``--hours`` can be raised to continue training; ``--force`` always re-opens it.
    """
    if not rd.done:
        return False
    import json as _json

    try:
        done_info = _json.loads((rd.path / "DONE").read_text() or "{}")
    except ValueError:
        done_info = {}
    trained_budget = float(done_info.get("budget_hours", 0.0) or 0.0)
    if force or requested_hours > trained_budget + 1e-6:
        rd.print(f"re-opening finished stage (trained budget {trained_budget} h -> requested {requested_hours} h)")
        (rd.path / "DONE").unlink()
        return False
    rd.print("stage already DONE (raise --hours to continue training, or use --force)")
    return True


def load_compatible(rd, modules: dict, state: dict | None) -> bool:
    """Load state_dicts; if shapes/keys no longer match the current architecture, archive the
    checkpoint and return False (caller starts fresh)."""
    if state is None:
        return False
    try:
        for name, module in modules.items():
            module.load_state_dict(state[name])
        return True
    except (RuntimeError, KeyError) as exc:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        src = rd.path / "ckpt" / "latest.pt"
        if src.exists():
            src.rename(rd.path / "ckpt" / f"incompatible_{stamp}.pt")
        rd.print(f"checkpoint does not match the current architecture ({type(exc).__name__}); "
                 f"archived as ckpt/incompatible_{stamp}.pt and starting fresh")
        return False
