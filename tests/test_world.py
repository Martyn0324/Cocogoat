"""CPU tests for the VAE, world model and video utilities."""

import numpy as np
import torch

from cocogoat.data.video import ClipBank, colored_moving_mnist, detect_cuts, frames_to_segments, synthetic_digits
from cocogoat.metrics.video import psnr, ssim
from cocogoat.models.vae import VAE, VAEConfig, kl_divergence
from cocogoat.models.world import WorldModel, WorldModelConfig, ema_memory


def test_vae_roundtrip_shapes_and_normalisation():
    vae = VAE(VAEConfig(channels=(16, 16, 32, 32), blocks=(1, 1, 1, 1), mid_blocks=1, heads=2, expansion=2.0))
    x = torch.rand(4, 3, 64, 64) * 2 - 1
    rec, mean, logvar, z = vae(x)
    assert rec.shape == x.shape and z.shape == (4, 16, 8, 8)
    assert kl_divergence(mean, logvar) >= 0
    vae.fit_latent_stats([x])
    zn = vae.encode_normalized(x)
    assert torch.allclose(zn.mean(dim=(0, 2, 3)), torch.zeros(16), atol=1e-4)
    assert torch.allclose(vae.denormalize(vae.normalize(z)), z, atol=1e-5)


def test_world_model_is_causal_and_starts_as_copy():
    wm = WorldModel(WorldModelConfig(latent_channels=4, latent_size=4, dim=32, depth=2, heads=4, max_frames=8))
    z = torch.randn(2, 6, 4, 4, 4)
    out = wm(z)
    assert out.shape == z.shape
    assert torch.allclose(out, z)  # zero-init head => predicts "copy last frame" at init
    with torch.no_grad():
        wm.head.weight.normal_()
    out1 = wm(z)
    z2 = z.clone()
    z2[:, 4:] += 1.0  # change the future: outputs for t < 4 must not change
    out2 = wm(z2)
    assert torch.allclose(out1[:, :4], out2[:, :4], atol=1e-5)
    assert not torch.allclose(out1[:, 4:], out2[:, 4:])
    roll = wm.rollout(z[:, :3], steps=7)  # longer than max_frames - window slides
    assert roll.shape == (2, 7, 4, 4, 4)


def test_ema_memory():
    z = torch.ones(1, 3, 2, 1, 1)
    z[:, 1:] = 0
    m = ema_memory(z, (0.5,))
    assert torch.allclose(m[0, :, 0, 0, 0], torch.tensor([1.0, 0.5, 0.25]))


def test_moving_mnist_and_clip_bank():
    seqs = colored_moving_mnist(synthetic_digits(16), n_seq=6, frames=10, size=64)
    assert seqs.shape == (6, 10, 64, 64, 3) and seqs.dtype == np.uint8 and seqs.max() > 0
    bank = ClipBank.from_sequences(seqs, device="cpu")
    clips = bank.sample_clips(5, 10)
    assert clips.shape == (5, 10, 3, 64, 64)
    starts = bank.valid_starts(10)
    assert len(starts) == 6  # exactly one 10-frame clip per 10-frame sequence


def test_cut_detection():
    rng = np.random.default_rng(0)
    a = np.full((20, 8, 8, 3), 40, np.uint8) + rng.integers(0, 3, (20, 8, 8, 3)).astype(np.uint8)
    b = np.full((20, 8, 8, 3), 200, np.uint8) + rng.integers(0, 3, (20, 8, 8, 3)).astype(np.uint8)
    frames = np.concatenate([a, b])
    assert list(detect_cuts(frames)) == [20]
    seg, train, cuts = frames_to_segments(frames, test_fraction=0.25)
    assert seg[19] != seg[20] and train.sum() == 30


def test_psnr_ssim():
    x = torch.rand(2, 3, 32, 32) * 2 - 1
    assert torch.all(psnr(x, x) > 90)
    assert torch.allclose(ssim(x, x), torch.ones(2), atol=1e-4)
    assert torch.all(ssim(x, -x) < 0.5)


def test_latent_generator_residual_identity_and_conditioning():
    from cocogoat.models.latent_gan import LatentGenerator, LatentGeneratorConfig

    g = LatentGenerator(LatentGeneratorConfig(latent_channels=4, latent_size=4, z_global_dim=8, w_dim=16, cond_dim=8,
                                              num_classes=3, width=16, depth=2, heads=2, expansion=2.0))
    z = torch.randn(5, 4, 4, 4)
    out = g(z, torch.randn(5, 8), torch.randint(0, 3, (5,)))
    assert out.shape == z.shape and torch.allclose(out, z)  # zero-init output => z' = z at init
