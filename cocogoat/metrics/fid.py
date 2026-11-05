"""Distribution metrics on feature embeddings.

* FID  (Heusel et al. 2017) - Frechet distance between Gaussian fits.
* KID  (Binkowski et al. 2018) - unbiased MMD^2 with the cubic polynomial kernel,
  averaged over random subsets (reported x1000 in the summaries, as is customary).
* Precision / Recall (Kynkaanniemi et al. 2019) - k-NN manifold estimates.
  Recall is the direct mode-coverage measure: the fraction of real samples that fall
  inside the estimated manifold of generated samples.

Features are produced by ``InceptionFeatures`` (FID-compatible InceptionV3 weights
from ``pytorch-fid``) unless a different extractor is injected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn


class InceptionFeatures(nn.Module):
    """pool3 (2048-d) InceptionV3 features with the TF-FID weights."""

    def __init__(self, device: str | torch.device = "cuda"):
        super().__init__()
        from pytorch_fid.inception import InceptionV3

        self.net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]], resize_input=True,
                               normalize_input=True).to(device).eval().requires_grad_(False)
        self.device = torch.device(device)

    @torch.no_grad()
    def forward(self, x_m11: torch.Tensor) -> torch.Tensor:
        x01 = ((x_m11.float().clamp(-1, 1) + 1) / 2).to(self.device)
        return self.net(x01)[0].flatten(1).float()


def gaussian_stats(feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    feats = np.asarray(feats, dtype=np.float64)
    return feats.mean(axis=0), np.cov(feats, rowvar=False)


def frechet_distance(mu1, sigma1, mu2, sigma2) -> float:
    from scipy import linalg

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def kid(f_real: np.ndarray, f_fake: np.ndarray, n_subsets: int = 100, subset_size: int = 1000,
        seed: int = 0) -> tuple[float, float]:
    """Kernel Inception Distance: mean and std of unbiased MMD^2 over subsets."""
    rng = np.random.default_rng(seed)
    f_real = np.asarray(f_real, dtype=np.float64)
    f_fake = np.asarray(f_fake, dtype=np.float64)
    m = min(subset_size, len(f_real), len(f_fake))
    d = f_real.shape[1]
    vals = []
    for _ in range(n_subsets):
        x = f_real[rng.choice(len(f_real), m, replace=False)]
        y = f_fake[rng.choice(len(f_fake), m, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3
        kyy = (y @ y.T / d + 1) ** 3
        kxy = (x @ y.T / d + 1) ** 3
        mmd = ((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
               - 2 * kxy.mean())
        vals.append(mmd)
    return float(np.mean(vals)), float(np.std(vals))


@torch.no_grad()
def _kth_nn_radius(x: torch.Tensor, k: int, chunk: int = 2048) -> torch.Tensor:
    radii = []
    for i in range(0, x.shape[0], chunk):
        d = torch.cdist(x[i:i + chunk], x)
        radii.append(d.kthvalue(k + 1, dim=1).values)  # +1 skips the zero self-distance
    return torch.cat(radii)


@torch.no_grad()
def _coverage_fraction(query: torch.Tensor, ref: torch.Tensor, ref_radii: torch.Tensor, chunk: int = 2048) -> float:
    inside = []
    for i in range(0, query.shape[0], chunk):
        d = torch.cdist(query[i:i + chunk], ref)
        inside.append((d <= ref_radii[None, :]).any(dim=1))
    return float(torch.cat(inside).float().mean())


@torch.no_grad()
def precision_recall(f_real: np.ndarray | torch.Tensor, f_fake: np.ndarray | torch.Tensor, k: int = 3,
                     device: str | torch.device | None = None) -> tuple[float, float]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    r = torch.as_tensor(np.asarray(f_real), dtype=torch.float32, device=device)
    f = torch.as_tensor(np.asarray(f_fake), dtype=torch.float32, device=device)
    rr = _kth_nn_radius(r, k)
    fr = _kth_nn_radius(f, k)
    precision = _coverage_fraction(f, r, rr)
    recall = _coverage_fraction(r, f, fr)
    return precision, recall


class ImageEvaluator:
    """Caches real-data features once, then scores generators with FID/KID/P/R."""

    def __init__(self, feature_fn: Callable[[torch.Tensor], torch.Tensor], real_batches, cache: str | Path | None = None,
                 max_real: int = 50000, pr_size: int = 10000):
        self.feature_fn = feature_fn
        self.pr_size = pr_size
        cache = Path(cache) if cache else None
        if cache is not None and cache.exists():
            with np.load(cache) as z:
                self.mu, self.sigma, self.real_sub = z["mu"], z["sigma"], z["sub"]
            return
        feats, n = [], 0
        for x in real_batches:
            feats.append(self.feature_fn(x).cpu().numpy())
            n += len(feats[-1])
            if n >= max_real:
                break
        feats = np.concatenate(feats)[:max_real]
        self.mu, self.sigma = gaussian_stats(feats)
        rng = np.random.default_rng(0)
        self.real_sub = feats[rng.choice(len(feats), min(pr_size, len(feats)), replace=False)]
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, mu=self.mu, sigma=self.sigma, sub=self.real_sub)

    @torch.no_grad()
    def features_of(self, sample_fn: Callable[[int], torch.Tensor], n: int, batch: int = 500) -> np.ndarray:
        feats, done = [], 0
        while done < n:
            b = min(batch, n - done)
            feats.append(self.feature_fn(sample_fn(b)).cpu().numpy())
            done += b
        return np.concatenate(feats)

    def score_features(self, fake: np.ndarray, kid_subset: int = 1000) -> dict:
        mu, sigma = gaussian_stats(fake)
        out = {"fid": frechet_distance(self.mu, self.sigma, mu, sigma)}
        km, ks = kid(self.real_sub, fake, subset_size=kid_subset)
        out["kid_x1000"], out["kid_std_x1000"] = km * 1000, ks * 1000
        m = min(len(self.real_sub), len(fake), self.pr_size)
        p, r = precision_recall(self.real_sub[:m], fake[:m])
        out["precision"], out["recall"] = p, r
        out["n_fake"] = int(len(fake))
        return out

    def evaluate(self, sample_fn: Callable[[int], torch.Tensor], n: int = 10000, batch: int = 500) -> dict:
        return self.score_features(self.features_of(sample_fn, n, batch))
