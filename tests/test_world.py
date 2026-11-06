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


def test_bulk_vae_passes_are_sized_by_pixels(monkeypatch):
    """No-grad bulk encode/decode must be split by pixel count: frame counts tuned for 64x64 run out of
    GPU memory at 256x256 (Task 5: 512 frames in one encoder pass -> a 32 GiB activation)."""
    import cocogoat.models.vae as vae_mod

    monkeypatch.setattr(vae_mod, "BULK_PIXELS", 8 * 32 * 32)          # => 8 frames per pass at 32x32
    torch.manual_seed(0)
    vae = VAE(VAEConfig(channels=(8, 8, 8, 8), blocks=(1, 1, 1, 1), mid_blocks=1, heads=2, expansion=2.0)).eval()
    seen = {"enc": [], "dec": []}
    vae.encoder.register_forward_pre_hook(lambda m, a: seen["enc"].append(a[0].shape[0]))
    vae.decoder.register_forward_pre_hook(lambda m, a: seen["dec"].append(a[0].shape[0]))
    x = torch.randn(20, 3, 32, 32)

    with torch.no_grad():
        ref_mean = vae.encode(x)[0]                                     # one pass, for reference
    seen["enc"].clear()
    vae.fit_latent_stats([x])
    assert max(seen["enc"]) <= 8
    assert torch.allclose(vae.latent_mean, ref_mean.double().mean(dim=(0, 2, 3)).float(), atol=1e-5)

    seen["enc"].clear()
    zn = vae.encode_normalized(x)
    assert max(seen["enc"]) <= 8 and zn.shape == (20, 16, 4, 4)
    assert torch.allclose(zn, vae.normalize(ref_mean), atol=1e-5)

    with torch.no_grad():
        out = vae.decode_normalized(zn)
    assert max(seen["dec"]) <= 8 and out.shape == (20, 3, 32, 32)
    seen["dec"].clear()
    zg = zn.clone().requires_grad_(True)                               # training path: one pass, graph intact
    vae.decode_normalized(zg).mean().backward()
    assert seen["dec"] == [20] and zg.grad is not None


def test_eval_length_shrinks_to_the_held_out_shots():
    import numpy as np

    from cocogoat.data.video import ClipBank
    from cocogoat.tasks.task2_world import Task2Config, fit_eval_length

    class _Log:
        def print(self, *a):
            pass

    seg = np.repeat(np.arange(10), 40)  # ten 40-frame shots
    bank = ClipBank(np.zeros((400, 8, 8, 3), np.uint8), seg, "cpu")
    cfg = Task2Config()
    cfg.eval_context, cfg.eval_horizon, cfg.clip_stride = 64, 64, 1
    fit_eval_length(cfg, bank, _Log())
    assert cfg.eval_context + cfg.eval_horizon <= 40 and abs(cfg.eval_context - cfg.eval_horizon) <= 1
    assert len(bank.valid_starts(cfg.eval_context + cfg.eval_horizon, 1)) >= 16
    cfg.eval_context, cfg.eval_horizon = 4, 8  # fits: unchanged
    fit_eval_length(cfg, bank, _Log())
    assert (cfg.eval_context, cfg.eval_horizon) == (4, 8)


def _chunked(slots: int, window: int = 4):
    from cocogoat.models.ltm import ChunkedWorldModel, LTMConfig
    from cocogoat.models.world import WorldModel, WorldModelConfig

    torch.manual_seed(0)
    cfg = WorldModelConfig(latent_channels=4, latent_size=4, dim=32, depth=2, heads=2, max_frames=2 * window,
                           rope_time=True, ltm_every=1 if slots else 0)
    wm = WorldModel(cfg).eval()
    torch.nn.init.normal_(wm.head.weight, std=0.1)  # the output head is zero-initialised (no gradient flows at init)
    if slots:  # make the memory read non-trivial (it is zero-initialised)
        for blk in wm.ltm_blocks.values():
            torch.nn.init.normal_(blk.proj.weight, std=0.1)
    return ChunkedWorldModel(wm, LTMConfig(slots=slots, window=window))


def test_chunked_teacher_forcing_matches_step_by_step_generation():
    from cocogoat.models.world import ema_memory

    runner = _chunked(slots=0)
    z = torch.randn(2, 13, 4, 4, 4)
    preds = []
    runner.teacher_forced(z, z, backward_fn=lambda loss: preds.append(None))
    # reference: predict frame t+1 one step at a time from frames [s - W, t], s = start of t's chunk
    mem = ema_memory(z, runner.wm.cfg.ema_decays)
    with torch.no_grad():
        ref = torch.stack([runner.wm(z[:, max(0, 4 * (t // 4) - 4):t + 1],
                                     memory=mem[:, max(0, 4 * (t // 4) - 4):t + 1])[:, -1] for t in range(12)], dim=1)
        bank = runner.new_bank(z)
        chunked = torch.cat([runner.step_chunk(z, mem, s, min(s + 4, 12), bank) for s in range(0, 12, 4)], dim=1)
    assert torch.allclose(chunked, ref, atol=1e-5)
    # rollout of one step equals the teacher-forced prediction of the next frame
    with torch.no_grad():
        one = runner.rollout(z[:, :9], 1)
    assert torch.allclose(one[:, 0], ref[:, 8], atol=1e-5)


def test_surprise_memory_keeps_the_most_surprising_frames():
    from cocogoat.models.ltm import SurpriseMemory

    bank = SurpriseMemory(1, 2, (1, 1, 1), decay=1.0, device="cpu")
    z = torch.arange(5.0).view(1, 5, 1, 1, 1)
    bank.write(z, torch.arange(5), torch.tensor([[0.1, 3.0, 0.2, 2.0, 0.5]]), now=4)
    assert sorted(bank.z.flatten().tolist()) == [1.0, 3.0]
    bank.write(z[:, :1] + 10, torch.tensor([5]), torch.tensor([[2.5]]), now=5)
    assert sorted(bank.z.flatten().tolist()) == [1.0, 10.0]
    # with decay, an old surprise loses to a slightly smaller fresh one
    bank2 = SurpriseMemory(1, 1, (1, 1, 1), decay=0.5, device="cpu")
    bank2.write(z[:, :1], torch.tensor([0]), torch.tensor([[1.0]]), now=0)
    bank2.write(z[:, 1:2], torch.tensor([10]), torch.tensor([[0.5]]), now=10)
    assert bank2.z.flatten().tolist() == [1.0]


def test_long_term_memory_changes_predictions_and_gets_gradients():
    runner = _chunked(slots=2)
    z = torch.randn(2, 13, 4, 4, 4)
    runner.teacher_forced(z, z, backward_fn=lambda loss: loss.backward())
    g = sum(p.grad.abs().sum() for p in runner.wm.ltm_blocks.parameters() if p.grad is not None)
    assert g > 0
    assert runner.last_stats["ltm_fill"] == 1.0
    with torch.no_grad():
        a = runner.rollout(z[:, :9], 4)
        runner.cfg.slots = 0  # same weights, memory read switched off
        b = runner.rollout(z[:, :9], 4)
    assert not torch.allclose(a, b)


def test_memory_rollout_matches_teacher_forcing_at_chunk_boundaries():
    """With memory on, the first generated frame equals the teacher-forced prediction (same bank, same context)."""
    runner = _chunked(slots=2)
    z = torch.randn(2, 13, 4, 4, 4)
    from cocogoat.models.world import ema_memory

    with torch.no_grad():
        runner.teacher_forced(z, z)
        for T0 in (5, 8, 9, 12):  # generation starts mid-chunk and at a chunk boundary
            one = runner.rollout(z[:, :T0], 1)
            # teacher-forced reference for the prediction made at frame T0-1
            bank = runner.new_bank(z)
            mem = ema_memory(z, runner.wm.cfg.ema_decays)
            from cocogoat.models.ltm import surprise
            ref = None
            for start in range(0, 12, 4):
                end = min(start + 4, 12)
                pred = runner.step_chunk(z, mem, start, end, bank)
                if start <= T0 - 1 < end:
                    ref = pred[:, T0 - 1 - start]
                    break
                tgt = z[:, start + 1:end + 1]
                bank.write(tgt, torch.arange(start + 1, end + 1), surprise(pred, tgt), now=end)
            assert torch.allclose(one[:, 0], ref, atol=1e-5), T0
