"""Artifact statistics for decoded images (no human inspection needed)."""

from __future__ import annotations

import torch


@torch.no_grad()
def seam_ratio(x: torch.Tensor, cell: int = 8) -> dict:
    """Seams of a latent grid in decoded images.

    Per image: the mean absolute jump between neighbouring pixels across latent-cell boundaries
    (columns/rows ``cell-1|cell``, ``2cell-1|2cell``, ...) divided by the mean jump between all other
    neighbours; returns the batch medians for vertical seams (column boundaries, ``seam_v``) and
    horizontal seams (row boundaries, ``seam_h``). Natural images score ~1.0; a decoder that renders
    each latent cell on its own scores > 1 (1.5 = jumps 50% larger at the cell edges).

    ``x``: (N, C, H, W), any value range.
    """
    x = x.float()
    dx = (x[..., :, 1:] - x[..., :, :-1]).abs().mean(dim=(1, 2))  # (N, W-1): jump between columns j and j+1
    dy = (x[..., 1:, :] - x[..., :-1, :]).abs().mean(dim=(1, 3))  # (N, H-1): jump between rows i and i+1

    def ratio(d: torch.Tensor) -> torch.Tensor:
        edge = torch.zeros(d.shape[1], dtype=torch.bool, device=d.device)
        edge[cell - 1::cell] = True
        return d[:, edge].mean(1) / d[:, ~edge].mean(1).clamp_min(1e-8)

    return {"seam_v": float(ratio(dx).median()), "seam_h": float(ratio(dy).median())}
