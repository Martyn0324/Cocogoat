"""Pseudo-labels for unlabelled datasets: spherical k-means on CLIP image embeddings.

Conditioning a GAN on clusters of a pretrained embedding space ("self-conditioned" /
cluster-conditioned GANs, Liu et al. CVPR 2020; IC-GAN, Casanova et al. 2021) is a
strong guard against mode collapse: every cluster is explicitly requested from the
generator in proportion to its frequency in the data, so no cluster can be dropped
without the discriminator noticing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def spherical_kmeans(x: torch.Tensor, k: int, iters: int = 40, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosine k-means with k-means++ seeding. ``x`` must be L2-normalised (N, D)."""
    n = x.shape[0]
    g = torch.Generator(device=x.device).manual_seed(seed)
    first = torch.randint(n, (1,), generator=g, device=x.device)
    centers = [x[first].squeeze(0)]
    min_d = (1 - x @ centers[0]).clamp_min(0)
    for _ in range(1, k):
        probs = min_d / min_d.sum().clamp_min(1e-12)
        idx = torch.multinomial(probs, 1, generator=g)
        centers.append(x[idx].squeeze(0))
        min_d = torch.minimum(min_d, (1 - x @ centers[-1]).clamp_min(0))
    c = torch.stack(centers)
    assign = torch.zeros(n, dtype=torch.long, device=x.device)
    for _ in range(iters):
        assign = (x @ c.T).argmax(dim=1)
        new = torch.zeros_like(c).index_add_(0, assign, x)
        counts = torch.bincount(assign, minlength=k)
        empty = counts == 0
        if empty.any():  # re-seed empty clusters at random points
            new[empty] = x[torch.randint(n, (int(empty.sum()),), generator=g, device=x.device)]
        c = torch.nn.functional.normalize(new, dim=1)
    assign = (x @ c.T).argmax(dim=1)
    return assign, c


@torch.no_grad()
def clip_pseudo_labels(images_uint8_nhwc: np.ndarray, k: int, cache: str | Path | None = None,
                       clip_name: str = "openai/clip-vit-base-patch32", device: str = "cuda",
                       batch: int = 512, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Returns (labels[N], centroids[k, D]); cached as .npz when ``cache`` is given."""
    if cache is not None and Path(cache).exists():
        with np.load(cache) as z:
            if z["labels"].shape[0] == images_uint8_nhwc.shape[0] and z["centroids"].shape[0] == k:
                return z["labels"], z["centroids"]
    from cocogoat.models.pretrained import CLIPEncoder

    enc = CLIPEncoder(clip_name, device=device)
    embs = []
    for i in range(0, len(images_uint8_nhwc), batch):
        x = torch.from_numpy(np.ascontiguousarray(images_uint8_nhwc[i:i + batch])).to(device)
        x = x.permute(0, 3, 1, 2).float() / 127.5 - 1
        embs.append(enc.encode_images(x))
    emb = torch.cat(embs)
    labels, centroids = spherical_kmeans(emb, k, seed=seed)
    labels_np = labels.cpu().numpy().astype(np.int64)
    cent_np = centroids.cpu().numpy().astype(np.float32)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, labels=labels_np, centroids=cent_np)
    del enc
    torch.cuda.empty_cache()
    return labels_np, cent_np
