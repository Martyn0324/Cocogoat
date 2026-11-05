"""CPU unit tests for the shared building blocks, losses and metrics."""

import numpy as np
import pytest
import torch

from cocogoat.data import diff_augment
from cocogoat.data.labels import spherical_kmeans
from cocogoat.losses import gradient_penalty, rpgan_d_loss, rpgan_g_loss
from cocogoat.metrics.fid import frechet_distance, gaussian_stats, kid, precision_recall
from cocogoat.models.gan2d import Discriminator, DiscriminatorConfig, Generator, GeneratorConfig
from cocogoat.nn import InvertedBottleneck, MinibatchStd, SelfAttention2d
from cocogoat.train import EMA, cosine_interp, ema_beta, log_cosine_interp


def tiny_gd(res=32, num_classes=0, vec=0):
    chans = {4: 32, 8: 32, 16: 24, 32: 16, 64: 16, 128: 16, 256: 16}
    g = Generator(GeneratorConfig(resolution=res, z_dim=8, w_dim=16, cond_dim=8, blocks=1, expansion=2.0,
                                  channels={k: v for k, v in chans.items() if k <= res}, num_classes=num_classes,
                                  cond_vector_dim=vec))
    d = Discriminator(DiscriminatorConfig(resolution=res, blocks=1, expansion=2.0,
                                          channels={k: v for k, v in chans.items() if k <= res},
                                          num_classes=num_classes, cond_vector_dim=vec))
    return g, d


@pytest.mark.parametrize("res", [32, 64, 128])
def test_generator_discriminator_shapes(res):
    g, d = tiny_gd(res, num_classes=5)
    z = torch.randn(4, 8)
    c = torch.randint(0, 5, (4,))
    x = g(z, c)
    assert x.shape == (4, 3, res, res)
    assert d(x, c).shape == (4,)


def test_vector_conditioning():
    g, d = tiny_gd(32, vec=12)
    c = torch.randn(4, 12)
    x = g(torch.randn(4, 8), c)
    assert d(x, c).shape == (4,)


def test_blocks_start_as_identity():
    x = torch.randn(2, 16, 8, 8)
    assert torch.allclose(InvertedBottleneck(16)(x), x)
    assert torch.allclose(SelfAttention2d(16, heads=4)(x), x)


def test_r1_double_backward_through_discriminator():
    """R1/R2 need second-order grads through depthwise convs, attention, mbstd, unshuffle."""
    _, d = tiny_gd(32, num_classes=3)
    x = torch.randn(4, 3, 32, 32, requires_grad=True)
    c = torch.randint(0, 3, (4,))
    logits = d(diff_augment(x, "color,translation,cutout", 1.0), c)
    r1 = gradient_penalty(logits, x).mean()
    r1.backward()
    grads = [p.grad for p in d.parameters() if p.grad is not None]
    assert len(grads) > 0 and all(torch.isfinite(gr).all() for gr in grads)


def test_gradient_penalty_value():
    x = torch.randn(3, 5, requires_grad=True)
    w = torch.randn(5)
    out = x @ w  # d out_i / d x_i = w  -> ||w||^2 per sample
    gp = gradient_penalty(out, x)
    assert torch.allclose(gp, w.square().sum().expand(3), atol=1e-5)


def test_rpgan_losses():
    r = torch.tensor([2.0, 1.0])
    f = torch.tensor([0.0, 1.0])
    assert rpgan_d_loss(r, f) < rpgan_g_loss(r, f)
    assert torch.isclose(rpgan_d_loss(r, r), torch.log(torch.tensor(2.0)))


def test_minibatch_std_shape():
    x = torch.randn(8, 6, 4, 4)
    assert MinibatchStd(4)(x).shape == (8, 7, 4, 4)
    assert MinibatchStd(4)(torch.randn(6, 6, 4, 4)).shape == (6, 7, 4, 4)  # group shrinks to divide batch


def test_diff_augment_preserves_shape_and_p0():
    x = torch.randn(4, 3, 16, 16)
    assert diff_augment(x, "color,translation,cutout", 1.0).shape == x.shape
    assert torch.equal(diff_augment(x, "color", 0.0), x)
    lat = torch.randn(4, 8, 8, 8)  # colour ops skipped for non-RGB tensors
    assert diff_augment(lat, "color,translation", 1.0).shape == lat.shape


def test_ema_and_schedules():
    m = torch.nn.Linear(2, 2)
    e = EMA(m)
    with torch.no_grad():
        m.weight.add_(1.0)
    e.update(m, 0.5)
    assert torch.allclose(e.model.weight, m.weight - 0.5)
    assert cosine_interp(1, 0, 0) == 1 and cosine_interp(1, 0, 1) == 0
    assert abs(log_cosine_interp(2, 0.2, 0.5) - (2 * 0.2) ** 0.5) < 1e-9
    assert 0 < ema_beta(256, 10, 10**6) < 1


def test_metrics_sanity():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(800, 16))
    b = rng.normal(size=(800, 16))
    c = rng.normal(size=(800, 16)) + 3
    mu_a, s_a = gaussian_stats(a)
    assert frechet_distance(mu_a, s_a, *gaussian_stats(b)) < frechet_distance(mu_a, s_a, *gaussian_stats(c))
    assert abs(kid(a, b, n_subsets=10, subset_size=200)[0]) < kid(a, c, n_subsets=10, subset_size=200)[0]
    p, r = precision_recall(a, b, k=3, device="cpu")
    assert p > 0.5 and r > 0.5
    p, r = precision_recall(a, a[:20].repeat(40, axis=0) + 1e-3 * rng.normal(size=(800, 16)), k=3, device="cpu")
    assert r < 0.3  # collapsed generator -> low recall


def test_spherical_kmeans():
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(4, 8))
    x = np.concatenate([c + 0.05 * rng.normal(size=(50, 8)) for c in centers])
    x = torch.nn.functional.normalize(torch.tensor(x, dtype=torch.float32), dim=1)
    labels, c = spherical_kmeans(x, 4, iters=20)
    assert len(torch.unique(labels)) == 4
    for k in range(4):  # each true cluster maps to one label
        assert len(torch.unique(labels[k * 50:(k + 1) * 50])) == 1
