"""Task 2 - per-frame beta-VAE + latent world model (next-latent prediction), 64x64 video.

Stage ``vae``: beta-VAE (L1 + LPIPS + beta*KL with warm-up + EQ-VAE scale equivariance + RpGAN
               adversarial term with VQGAN-style adaptive weight after warm-up).
Stage ``wm`` : frozen VAE -> normalised latents -> factorised space-time transformer with EMA memory,
               trained with teacher-forced MSE + context-noise augmentation + a short self-rollout loss.

Datasets: ``mmnist`` (Colored Moving MNIST, generated) or ``shelter`` (the user's Shelterdataset.npz).

Usage (Colab):
    python -m cocogoat.tasks.task2_world --dataset mmnist
    python -m cocogoat.tasks.task2_world --dataset shelter
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cocogoat.data.images import load_array_images
from cocogoat.data.video import (
    ClipBank,
    colored_moving_mnist,
    frames_to_segments,
    load_mnist_digits,
    resize_frames_gpu,
    synthetic_digits,
)
from cocogoat.losses import rpgan_g_loss
from cocogoat.metrics.video import psnr, ssim
from cocogoat.models.gan2d import Discriminator, DiscriminatorConfig
from cocogoat.models.vae import VAE, VAEConfig, frames_per_pass, kl_divergence
from cocogoat.models.world import WorldModel, WorldModelConfig
from cocogoat.nn import count_params
from cocogoat.tasks.common import (
    DRIVE_ROOT,
    fresh_smoke_run,
    LOCAL_CACHE,
    USER_DATA,
    apply_overrides,
    make_feature_fn,
    parse_override,
    rotate_old_error,
)
from cocogoat.train import (
    Every,
    RunDir,
    TrainClock,
    autocast,
    cosine_interp,
    describe_env,
    rng_state,
    save_grid,
    save_video_gif,
    seed_everything,
    set_optimizer_hparams,
    set_rng_state,
    setup_torch,
)
from cocogoat.train.adversarial import AdversarialTrainer, CriticSlot, make_adam
from cocogoat.tasks.task1_gan64 import PixelCritic


@dataclass
class Task2Config:
    run_root: str = str(DRIVE_ROOT / "runs" / "task2")
    dataset: str = "mmnist"                 # mmnist | shelter | array
    data_path: str = str(USER_DATA / "Shelterdataset.npz")
    data_key: str | None = None
    drive_cache: str = str(DRIVE_ROOT / "cache")
    resolution: int = 64
    stage: str = "all"                      # vae | wm | all
    # moving mnist
    mmnist_train_seqs: int = 10000
    mmnist_test_seqs: int = 512
    mmnist_frames: int = 20
    # VAE
    vae: VAEConfig = field(default_factory=VAEConfig)
    vae_hours: float = 0.75
    vae_batch: int = 128
    vae_lr: tuple = (2e-4, 2e-5)
    kl_weight: float = 1e-3
    kl_warmup: float = 0.15                 # fraction of the budget
    lpips_weight: float = 1.0
    eq_weight: float = 0.5
    eq_prob: float = 0.5
    adv_weight: float = 0.1
    adv_start: float = 0.35                 # fraction of the budget
    vae_gamma: float = 0.5
    vae_D: DiscriminatorConfig = field(default_factory=lambda: DiscriminatorConfig(
        channels={64: 64, 32: 128, 16: 256, 8: 256, 4: 256}, blocks=1, attn_res=(8,)))
    # world model
    wm: WorldModelConfig = field(default_factory=WorldModelConfig)
    wm_hours: float = 0.75
    wm_batch: int = 64
    clip_len: int = 16
    clip_stride: int = 1
    wm_lr: tuple = (3e-4, 3e-5)
    wm_warmup_steps: int = 300
    ctx_noise: float = 0.1
    rollout_k: int = 3
    rollout_weight: float = 0.5
    rollout_fraction: float = 0.25
    eval_context: int = 64                  # real frames given to the world model in evaluation
    eval_horizon: int = 64                  # frames it then generates autoregressively (any length)
    eval_clips: int = 256                   # held-out clips (capped so all of them fit in memory)
    eval_batch: int = 64                    # clips rolled out together (capped for long horizons)
    start_frame: int = 0                    # video datasets: drop the first N frames (e.g. Shelter's black intro)
    gif_clips: int = 4                      # evaluation rollouts also saved as an animated GIF (0 = none)
    gif_fps: float = 8.0
    # cadence / misc
    log_every: int = 50
    eval_minutes: float = 15.0
    ckpt_minutes: float = 15.0
    amp: bool = True
    metrics: str = "inception"
    seed: int = 0
    device: str | None = None
    smoke: bool = False
    force: bool = False
    max_steps: int = 0


def smoke_config(cfg: Task2Config) -> Task2Config:
    cfg.smoke = True
    cfg.vae = VAEConfig(channels=(16, 16, 32, 32), blocks=(1, 1, 1, 1), mid_blocks=1, heads=2, expansion=2.0)
    cfg.vae_D = DiscriminatorConfig(channels={64: 16, 32: 16, 16: 16, 8: 16, 4: 16}, blocks=1, attn_res=(8,),
                                    heads=2, expansion=2.0)
    cfg.wm = WorldModelConfig(dim=32, depth=1, heads=2)
    cfg.vae_batch, cfg.wm_batch = 8, 4
    cfg.mmnist_train_seqs, cfg.mmnist_test_seqs, cfg.mmnist_frames = 24, 8, 20
    cfg.clip_len = 8
    cfg.eval_context, cfg.eval_horizon, cfg.eval_clips = 4, 4, 4
    cfg.max_steps = cfg.max_steps or 6
    cfg.adv_start = 0.0
    cfg.log_every = 2
    cfg.eval_minutes = cfg.ckpt_minutes = 0.0
    cfg.metrics = "random"
    cfg.amp = False
    return cfg


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------

def load_banks(cfg: Task2Config, device: torch.device, rd: RunDir) -> tuple[ClipBank, ClipBank, dict]:
    if cfg.dataset == "mmnist":
        cache = LOCAL_CACHE / "mnist"
        if cfg.smoke:
            digits_tr, digits_te = synthetic_digits(256, 0), synthetic_digits(64, 1)
        else:
            digits_tr, digits_te = load_mnist_digits(cache, True), load_mnist_digits(cache, False)
        tr = colored_moving_mnist(digits_tr, cfg.mmnist_train_seqs, cfg.mmnist_frames, cfg.resolution, seed=0)
        te_frames = max(cfg.mmnist_frames, cfg.eval_context + cfg.eval_horizon)  # long enough for evaluation
        te = colored_moving_mnist(digits_te, cfg.mmnist_test_seqs, te_frames, cfg.resolution, seed=1)
        info = {"train_frames": int(tr.shape[0] * tr.shape[1]), "test_frames": int(te.shape[0] * te.shape[1]),
                "train_seqs": int(tr.shape[0])}
        return ClipBank.from_sequences(tr, device), ClipBank.from_sequences(te, device), info
    path = Path(cfg.data_path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    drive_cache = Path(cfg.drive_cache) if Path(cfg.drive_cache).parent.exists() else None
    frames = np.asarray(load_array_images(path, cfg.data_key, cache_dir=LOCAL_CACHE, drive_cache=drive_cache))
    rd.print(f"frames {frames.shape} from {path.name}")
    if cfg.start_frame:
        if cfg.start_frame >= len(frames):
            raise ValueError(f"start_frame {cfg.start_frame} >= number of frames {len(frames)}")
        frames = frames[cfg.start_frame:]
        rd.print(f"dropped the first {cfg.start_frame} frames -> {len(frames)} frames")
    if frames.shape[1] != cfg.resolution or frames.shape[2] != cfg.resolution:
        t = torch.from_numpy(frames).permute(0, 3, 1, 2)
        out = []
        for i in range(0, len(t), 2048):
            out.append(resize_frames_gpu(t[i:i + 2048].to(device), cfg.resolution).cpu())
        frames = torch.cat(out).permute(0, 2, 3, 1).numpy()
    seg, train_mask, cuts = frames_to_segments(frames)
    info = {"frames": int(len(frames)), "cuts": int(len(cuts)), "train_frames": int(train_mask.sum()),
            "test_frames": int((~train_mask).sum())}
    rd.print(f"scene cuts detected: {len(cuts)}; train {info['train_frames']} / test {info['test_frames']} frames")
    train = ClipBank(frames[train_mask], seg[train_mask], device)
    test = ClipBank(frames[~train_mask], seg[~train_mask], device)
    return train, test, info


# --------------------------------------------------------------------------------------
# Stage 1: VAE
# --------------------------------------------------------------------------------------

def adaptive_weight(nll: torch.Tensor, g_loss: torch.Tensor, last_layer: torch.Tensor) -> torch.Tensor:
    """VQGAN: balance adversarial and reconstruction gradients at the decoder's last layer."""
    g1 = torch.autograd.grad(nll, last_layer, retain_graph=True)[0]
    g2 = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
    return (g1.norm() / (g2.norm() + 1e-4)).clamp(0.0, 1e4).detach()


def train_vae(cfg: Task2Config, train: ClipBank, test: ClipBank, device: torch.device, run_dir: Path) -> Path:
    rd = RunDir(run_dir, config=cfg)
    out_path = run_dir / "vae.pt"
    with rd.capture_errors():
        if rd.done and out_path.exists() and not cfg.force:
            rd.print("VAE stage already DONE")
            return out_path
        from cocogoat.losses.perceptual import PerceptualLoss

        vae = VAE(cfg.vae).to(device)
        cfg.vae_D.resolution = cfg.resolution
        D = Discriminator(cfg.vae_D).to(device)
        rd.print(f"VAE params {count_params(vae) / 1e6:.2f}M, D params {count_params(D) / 1e6:.2f}M")
        lpips = PerceptualLoss(device, random_weights=cfg.metrics == "random")
        opt = torch.optim.AdamW(vae.parameters(), lr=cfg.vae_lr[0], betas=(0.5, 0.9), weight_decay=0.0)
        opt_D = make_adam(D.parameters(), cfg.vae_lr[0], 0.99)
        critic = PixelCritic(D, "translation,cutout", 0.3)
        slot = CriticSlot("vaeD", critic, opt_D, gamma=cfg.vae_gamma)
        trainer = AdversarialTrainer([slot], device, amp=cfg.amp, reg_every=4)
        step, elapsed = 0, 0.0
        st = rd.load_checkpoint()
        if st is not None:
            vae.load_state_dict(st["vae"])
            D.load_state_dict(st["D"])
            opt.load_state_dict(st["opt"])
            opt_D.load_state_dict(st["opt_D"])
            step, elapsed = st["step"], st["elapsed"]
            set_rng_state(st["rng"])
            rd.print(f"resumed VAE at step {step}")
        clock = TrainClock(elapsed)
        budget = cfg.vae_hours * 3600
        fixed = test.sample_frames(32, xflip=False) if len(test) >= 32 else train.sample_frames(32, xflip=False)
        ckpt_timer, eval_timer = Every(cfg.ckpt_minutes * 60), Every(cfg.eval_minutes * 60)

        def checkpoint():
            rd.save_checkpoint({"vae": vae.state_dict(), "D": D.state_dict(), "opt": opt.state_dict(),
                                "opt_D": opt_D.state_dict(), "step": step, "elapsed": clock.now(), "rng": rng_state()})

        def recon_grid(tag):
            vae.eval()
            with torch.no_grad(), autocast(device, cfg.amp):
                rec = vae(fixed, sample=False)[0].float()
            vae.train()
            pair = torch.stack([fixed, rec], 1).flatten(0, 1)  # alternate original / reconstruction
            save_grid(pair, rd.path / "samples" / f"recon_{tag}.png", nrow=8)

        t_log = time.time()
        clock.start()
        while True:
            progress = clock.now() / budget
            if (cfg.max_steps and step >= cfg.max_steps) or (not cfg.max_steps and progress >= 1):
                break
            lr = cosine_interp(*cfg.vae_lr, progress)
            set_optimizer_hparams(opt, lr=lr)
            set_optimizer_hparams(opt_D, lr=lr)
            x = train.sample_frames(cfg.vae_batch)
            with autocast(device, cfg.amp):
                x_rec, mean, logvar, z = vae(x)
            x_rec = x_rec.float()
            rec = (x - x_rec).abs().mean()
            lp = lpips(x_rec, x)
            kl = kl_divergence(mean, logvar)
            beta = cfg.kl_weight * min(1.0, progress / max(cfg.kl_warmup, 1e-8))
            nll = rec + cfg.lpips_weight * lp
            loss = nll + beta * kl
            stats = {"rec_l1": float(rec), "lpips": float(lp), "kl": float(kl), "beta": beta, "lr": lr,
                     "post_std": float((0.5 * logvar).exp().mean())}
            if cfg.eq_weight > 0 and torch.rand(()) < cfg.eq_prob:
                # EQ-VAE: decoding a 2x-downscaled latent must give the 2x-downscaled image
                with autocast(device, cfg.amp):
                    x_half = vae.decode(F.avg_pool2d(z, 2)).float()
                target = F.interpolate(x, scale_factor=0.5, mode="bilinear", antialias=True, align_corners=False)
                eq = (x_half - target).abs().mean() + cfg.lpips_weight * lpips(x_half, target)
                loss = loss + cfg.eq_weight * eq
                stats["eq"] = float(eq)
            adv_on = progress >= cfg.adv_start
            if adv_on:
                for p in D.parameters():
                    p.requires_grad_(False)
                with autocast(device, cfg.amp):
                    lr_ = D(critic.encode({"img": x})["img"]).detach()
                    lf_ = D(critic.encode({"img": x_rec})["img"])
                g_adv = rpgan_g_loss(lr_, lf_)
                w = adaptive_weight(nll, g_adv, vae.decoder.out.weight) * cfg.adv_weight
                loss = loss + w * g_adv
                stats.update(g_adv=float(g_adv), adv_w=float(w))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            opt.step()
            if adv_on:
                stats.update(trainer.d_step({"img": x}, {"img": x_rec.detach()}, None, step))
            step += 1
            if step % cfg.log_every == 0:
                stats["it_per_s"] = cfg.log_every / max(time.time() - t_log, 1e-9)
                t_log = time.time()
                rd.log_metrics(kind="vae_train", step=step, hours=clock.now() / 3600, **stats)
                rd.write_status(stage="vae", step=step, hours=clock.now() / 3600, budget_hours=cfg.vae_hours, **stats)
                if step % (cfg.log_every * 10) == 0 or cfg.max_steps:
                    rd.print(f"vae step {step} " + " ".join(f"{k}={v:.4g}" for k, v in stats.items()))
            if eval_timer():
                clock.pause()
                recon_grid(f"step{step:06d}")
                clock.start()
            if ckpt_timer():
                clock.pause()
                checkpoint()
                clock.start()
        clock.pause()
        checkpoint()
        vae.eval()
        vae.fit_latent_stats(train.sample_frames(512, xflip=False) for _ in range(8 if not cfg.smoke else 1))
        torch.save({"vae": vae.state_dict(), "vae_config": dataclasses.asdict(cfg.vae)}, out_path)
        recon_grid("final")
        metrics = evaluate_vae(cfg, vae, test, device, rd)
        rd.write_json("vae_metrics.json", metrics)
        rd.mark_done(metrics)
        rd.print(f"VAE done: {metrics}")
        return out_path


@torch.no_grad()
def evaluate_vae(cfg: Task2Config, vae: VAE, test: ClipBank, device: torch.device, rd: RunDir) -> dict:
    from cocogoat.metrics.artifacts import seam_ratio
    from cocogoat.metrics.fid import ImageEvaluator
    from cocogoat.metrics.video import LPIPSMetric

    lp = LPIPSMetric(device, random_weights=cfg.metrics == "random")
    n = min(len(test), 10000 if not cfg.smoke else 32)
    bs = min(256, frames_per_pass(cfg.resolution, cfg.resolution))  # 256 frames at 64 px, 32 at 256 px
    ps, ss, ls = [], [], []
    recs = []
    for i in range(0, n, bs):
        x = ClipBank.to_float(test.frames[i:min(i + bs, n)])
        with autocast(device, cfg.amp):
            r = vae(x, sample=False)[0].float().clamp(-1, 1)
        ps.append(psnr(r, x))
        ss.append(ssim(r, x))
        ls.append(lp(r, x))
        recs.append(r)
    feat = make_feature_fn(cfg.metrics, device)
    real_iter = (ClipBank.to_float(test.frames[i:min(i + 256, n)]) for i in range(0, n, 256))
    ev = ImageEvaluator(feat, real_iter, max_real=n)
    recs = torch.cat(recs)
    fake_feats = np.concatenate([feat(recs[i:i + 256]).cpu().numpy() for i in range(0, len(recs), 256)])
    rfid = ev.score_features(fake_feats, kid_subset=min(1000, n))["fid"]
    seams = seam_ratio(recs[:1024], vae.cfg.downsample)  # latent-cell seams of reconstructions (1.0 = none)
    seams_real = seam_ratio(ClipBank.to_float(test.frames[:min(n, 1024)]), vae.cfg.downsample)
    return {"psnr": float(torch.cat(ps).mean()), "ssim": float(torch.cat(ss).mean()),
            "lpips": float(torch.cat(ls).mean()), "rfid": rfid, "n_frames": n,
            "seam_v": seams["seam_v"], "seam_h": seams["seam_h"],
            "seam_v_real": seams_real["seam_v"], "seam_h_real": seams_real["seam_h"],
            "latent_mean": vae.latent_mean.tolist(), "latent_std": vae.latent_std.tolist()}


# --------------------------------------------------------------------------------------
# Stage 2: world model
# --------------------------------------------------------------------------------------

@torch.no_grad()
def encode_clips(vae: VAE, clips: torch.Tensor, amp: bool, device) -> torch.Tensor:
    b, t = clips.shape[:2]
    with autocast(device, amp):
        z = vae.encode_normalized(clips.flatten(0, 1), sample=False)
    return z.float().view(b, t, *z.shape[1:])


@torch.no_grad()
def decode_latents(vae: VAE, z: torch.Tensor, amp: bool, device) -> torch.Tensor:
    b, t = z.shape[:2]
    with autocast(device, amp):
        x = vae.decode_normalized(z.flatten(0, 1))
    return x.float().clamp(-1, 1).view(b, t, *x.shape[1:])


def fit_eval_length(cfg: Task2Config, test: ClipBank, rd: RunDir, min_clips: int = 16) -> None:
    """Shorten eval_context + eval_horizon when the held-out shots are shorter than that.

    Evaluation needs ground-truth frames for every generated frame, so a clip may not cross a scene
    cut. If too few cut-free held-out clips of the requested length exist, the longest length with at
    least ``min_clips`` clips (or the longest possible) is used, split between context and horizon in
    the requested proportion.
    """
    c, h = cfg.eval_context, cfg.eval_horizon
    want = min(cfg.eval_clips, min_clips)
    n = lambda length: len(test.valid_starts(length, cfg.clip_stride))  # noqa: E731
    if n(c + h) >= want:
        return
    lengths = [L for L in range(c + h - 1, 1, -1) if n(L) > 0]
    if not lengths:
        raise ValueError("no held-out clip has even 2 frames without a scene cut")
    good = [L for L in lengths if n(L) >= want]
    L = good[0] if good else lengths[0]
    if n(c + h) > 0 and L < c + h and not good:
        return  # some clips of the requested length exist and no shorter length helps
    new_c = min(c, max(1, round(L * c / (c + h))))
    new_h = L - new_c
    if new_h < 1:
        new_c, new_h = L - 1, 1
    rd.print(f"WARNING: only {n(c + h)} held-out clips have {c + h} frames without a scene cut; evaluating with "
             f"eval_context {new_c} + eval_horizon {new_h} ({n(L)} clips) instead of {c} + {h}")
    cfg.eval_context, cfg.eval_horizon = new_c, new_h


def train_world_model(cfg: Task2Config, vae_path: Path, train: ClipBank, test: ClipBank, device: torch.device,
                      run_dir: Path) -> dict:
    rd = RunDir(run_dir, config=cfg)
    with rd.capture_errors():
        if rd.done and not cfg.force:
            rd.print("world-model stage already DONE")
            return {}
        from cocogoat.models.vae import load_vae

        vae = load_vae(str(vae_path), device).requires_grad_(False)
        f = vae.cfg.downsample
        cfg.wm.latent_channels = vae.cfg.latent_channels
        cfg.wm.latent_size = cfg.resolution // f
        cfg.wm.max_frames = max(cfg.clip_len, cfg.eval_context + 1)
        wm = WorldModel(cfg.wm).to(device)
        rd.print(f"world model params {count_params(wm) / 1e6:.2f}M; latent {cfg.wm.latent_channels}x"
                 f"{cfg.wm.latent_size}x{cfg.wm.latent_size}; clip {cfg.clip_len}")
        opt = torch.optim.AdamW(wm.parameters(), lr=cfg.wm_lr[0], betas=(0.9, 0.95), weight_decay=0.01)
        step, elapsed, best = 0, 0.0, float("inf")
        st = rd.load_checkpoint()
        if st is not None:
            wm.load_state_dict(st["wm"])
            opt.load_state_dict(st["opt"])
            step, elapsed, best = st["step"], st["elapsed"], st.get("best", best)
            set_rng_state(st["rng"])
            rd.print(f"resumed world model at step {step}")
        fit_eval_length(cfg, test, rd)
        eval_len = cfg.eval_context + cfg.eval_horizon
        n_avail = len(test.valid_starts(eval_len, cfg.clip_stride))
        # clips are kept on the GPU as float: cap the total number of frames (~0.8 GB at 64 px)
        n_eval = max(1, min(cfg.eval_clips, n_avail, (16384 * 64 * 64) // (eval_len * cfg.resolution ** 2)))
        eval_clips = test.fixed_clips(n_eval, eval_len, cfg.clip_stride, seed=0)
        rd.print(f"evaluation: {n_eval} held-out clips x ({cfg.eval_context} context + {cfg.eval_horizon} "
                 f"generated) frames")
        clock = TrainClock(elapsed)
        budget = cfg.wm_hours * 3600
        ckpt_timer, eval_timer = Every(cfg.ckpt_minutes * 60), Every(cfg.eval_minutes * 60)

        def checkpoint():
            rd.save_checkpoint({"wm": wm.state_dict(), "opt": opt.state_dict(), "step": step, "elapsed": clock.now(),
                                "best": best, "rng": rng_state(), "wm_config": dataclasses.asdict(cfg.wm)})

        def evaluate(tag: str) -> dict:
            nonlocal best
            wm.eval()
            m = evaluate_world_model(cfg, vae, wm, eval_clips, device, rd, tag)
            wm.train()
            rd.log_metrics(kind="wm_eval", step=step, hours=clock.now() / 3600, tag=tag, **m)
            score = float(np.mean(m["model_latent_mse"]))
            if score < best:
                best = score
                torch.save({"wm": wm.state_dict(), "wm_config": dataclasses.asdict(cfg.wm)}, rd.path / "wm_best.pt")
            rd.print(f"[wm eval {tag}] latent_mse model={np.mean(m['model_latent_mse']):.4f} "
                     f"copy={np.mean(m['copy_latent_mse']):.4f} psnr model={np.mean(m['model_psnr']):.2f} "
                     f"copy={np.mean(m['copy_psnr']):.2f} vae_upper={np.mean(m['vae_psnr']):.2f}")
            return m

        t_log = time.time()
        clock.start()
        while True:
            progress = clock.now() / budget
            if (cfg.max_steps and step >= cfg.max_steps) or (not cfg.max_steps and progress >= 1):
                break
            warm = min(1.0, (step + 1) / cfg.wm_warmup_steps)
            lr = cosine_interp(*cfg.wm_lr, progress) * warm
            set_optimizer_hparams(opt, lr=lr)
            clips = train.sample_clips(cfg.wm_batch, cfg.clip_len, cfg.clip_stride)
            z = encode_clips(vae, clips, cfg.amp, device)
            sigma = torch.rand(z.shape[0], 1, 1, 1, 1, device=device) * cfg.ctx_noise
            z_in = z + sigma * torch.randn_like(z)
            with autocast(device, cfg.amp):
                pred = wm(z_in).float()
            loss_tf = F.mse_loss(pred[:, :-1], z[:, 1:])
            loss = loss_tf
            stats = {"tf_mse": float(loss_tf), "lr": lr}
            if cfg.rollout_k > 0 and cfg.rollout_weight > 0:
                br = max(1, int(cfg.wm_batch * cfg.rollout_fraction))
                k = cfg.rollout_k
                ctx = z_in[:br, :cfg.clip_len - k]
                seq, preds = ctx, []
                with autocast(device, cfg.amp):
                    for _ in range(k):
                        nxt = wm(seq)[:, -1:].float()
                        preds.append(nxt)
                        seq = torch.cat([seq, nxt], dim=1)
                loss_ro = F.mse_loss(torch.cat(preds, 1), z[:br, cfg.clip_len - k:])
                loss = loss + cfg.rollout_weight * loss_ro
                stats["rollout_mse"] = float(loss_ro)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            stats["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0))
            opt.step()
            step += 1
            if step % cfg.log_every == 0:
                stats["it_per_s"] = cfg.log_every / max(time.time() - t_log, 1e-9)
                t_log = time.time()
                rd.log_metrics(kind="wm_train", step=step, hours=clock.now() / 3600, **stats)
                rd.write_status(stage="wm", step=step, hours=clock.now() / 3600, budget_hours=cfg.wm_hours, **stats)
                if step % (cfg.log_every * 10) == 0 or cfg.max_steps:
                    rd.print(f"wm step {step} " + " ".join(f"{k}={v:.4g}" for k, v in stats.items()))
            if eval_timer():
                clock.pause()
                evaluate(f"step{step:06d}")
                clock.start()
            if ckpt_timer():
                clock.pause()
                checkpoint()
                clock.start()
        clock.pause()
        checkpoint()
        final = evaluate("final")
        torch.save({"wm": wm.state_dict(), "wm_config": dataclasses.asdict(cfg.wm)}, rd.path / "wm_final.pt")
        rd.write_json("wm_metrics.json", final)
        write_wm_report(rd, final)
        rd.mark_done({"final": final})
        return final


@torch.no_grad()
def evaluate_world_model(cfg: Task2Config, vae: VAE, wm: WorldModel, clips: torch.Tensor, device, rd: RunDir,
                         tag: str) -> dict:
    from cocogoat.metrics.video import LPIPSMetric

    lp = LPIPSMetric(device, random_weights=cfg.metrics == "random")
    c, h = cfg.eval_context, cfg.eval_horizon
    res = {k: [] for k in ["model_latent_mse", "copy_latent_mse", "model_psnr", "copy_psnr", "vae_psnr",
                           "model_ssim", "copy_ssim", "model_lpips", "copy_lpips"]}
    per_clip = {k: [] for k in res}
    strips = []
    nb = max(1, min(cfg.eval_batch, 1024 // (c + h)))  # batched rollout; ~1k frames per pass for long horizons
    for i in range(0, clips.shape[0], nb):
        x = clips[i:i + nb]
        z = encode_clips(vae, x, cfg.amp, device)
        with autocast(device, cfg.amp):
            pred = wm.rollout(z[:, :c], h).float()
        gt_z = z[:, c:]
        copy_z = z[:, c - 1:c].expand_as(gt_z)
        x_pred = decode_latents(vae, pred, cfg.amp, device)
        x_vae = decode_latents(vae, gt_z, cfg.amp, device)
        x_gt = x[:, c:]
        x_copy = x[:, c - 1:c].expand_as(x_gt)
        for name, val in [("model_latent_mse", (pred - gt_z).square().mean(dim=(2, 3, 4))),
                          ("copy_latent_mse", (copy_z - gt_z).square().mean(dim=(2, 3, 4)))]:
            per_clip[name].append(val)
        flat = lambda t: t.flatten(0, 1)  # noqa: E731
        bsz = x_gt.shape[0]
        per_clip["model_psnr"].append(psnr(flat(x_pred), flat(x_gt)).view(bsz, h))
        per_clip["copy_psnr"].append(psnr(flat(x_copy), flat(x_gt)).view(bsz, h))
        per_clip["vae_psnr"].append(psnr(flat(x_vae), flat(x_gt)).view(bsz, h))
        per_clip["model_ssim"].append(ssim(flat(x_pred), flat(x_gt)).view(bsz, h))
        per_clip["copy_ssim"].append(ssim(flat(x_copy), flat(x_gt)).view(bsz, h))
        per_clip["model_lpips"].append(lp(flat(x_pred), flat(x_gt)).view(bsz, h))
        per_clip["copy_lpips"].append(lp(flat(x_copy), flat(x_gt)).view(bsz, h))
        if not strips:
            for j in range(min(max(4, cfg.gif_clips), bsz)):
                strips += [x[j, :c].cpu(), x_gt[j].cpu(), x_pred[j].cpu()]
    for k in res:
        res[k] = torch.cat(per_clip[k]).mean(0).tolist()  # per-horizon averages
    # visual strips: for 4 clips -> rows: context, ground truth future, predicted future
    rows = []
    for j in range(0, len(strips), 3):
        ctx, gt, pr = strips[j:j + 3]
        rows += [torch.cat([ctx, gt]), torch.cat([ctx, pr])]
    if rows:
        save_grid(torch.cat(rows[:8]), rd.path / "samples" / f"rollout_{tag}.png", nrow=c + h, max_side=8192)
    n_gif = min(cfg.gif_clips, len(rows) // 2)
    if n_gif:
        # one column per clip: top = real video, bottom = context + world-model rollout.
        # Border: grey = real frame, green = context given to the model, red = generated frame.
        vids = torch.stack(rows[0:2 * n_gif:2] + rows[1:2 * n_gif:2])  # (2*n, c+h, 3, H, W)
        colours = torch.zeros(2 * n_gif, c + h, 3, dtype=torch.uint8)
        colours[:n_gif] = torch.tensor([128, 128, 128], dtype=torch.uint8)
        colours[n_gif:, :c] = torch.tensor([0, 200, 0], dtype=torch.uint8)
        colours[n_gif:, c:] = torch.tensor([220, 0, 0], dtype=torch.uint8)
        save_video_gif(vids, rd.path / "samples" / f"rollout_{tag}.gif", nrow=n_gif, fps=cfg.gif_fps,
                       borders=colours)
    res["context"], res["horizon"] = c, h
    return res


def write_wm_report(rd: RunDir, m: dict) -> None:
    h = m["horizon"]
    lines = ["# Task 2 - world model rollout metrics", "",
             f"Context {m['context']} frames, autoregressive horizon {h}. 'copy' = repeat the last observed frame.",
             "", "| t+k | latent MSE model | latent MSE copy | PSNR model | PSNR copy | PSNR VAE-recon (upper bound) |"
             " SSIM model | SSIM copy | LPIPS model | LPIPS copy |", "|" + "---|" * 10]
    for k in range(h):
        lines.append(f"| {k + 1} | {m['model_latent_mse'][k]:.4f} | {m['copy_latent_mse'][k]:.4f} | "
                     f"{m['model_psnr'][k]:.2f} | {m['copy_psnr'][k]:.2f} | {m['vae_psnr'][k]:.2f} | "
                     f"{m['model_ssim'][k]:.3f} | {m['copy_ssim'][k]:.3f} | {m['model_lpips'][k]:.3f} | "
                     f"{m['copy_lpips'][k]:.3f} |")
    (rd.path / "report.md").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------------------

def run(cfg: Task2Config) -> dict:
    device = setup_torch(cfg.device)
    seed_everything(cfg.seed)
    root = Path(cfg.run_root) / cfg.dataset
    root.mkdir(parents=True, exist_ok=True)
    for sub in ("vae", "wm"):
        (root / sub).mkdir(exist_ok=True)
        rotate_old_error(root / sub)
    rd = RunDir(root, config=cfg)
    with rd.capture_errors():
        rd.print(f"env: {describe_env()}")
        if cfg.dataset == "mmnist":
            if cfg.smoke:  # training clips must fit in the (short) training sequences
                cfg.clip_len = min(max(cfg.clip_len, cfg.eval_context + cfg.eval_horizon), cfg.mmnist_frames)
        train, test, info = load_banks(cfg, device, rd)
        rd.write_json("data_info.json", info)
        vae_path = root / "vae" / "vae.pt"
        if cfg.stage in ("vae", "all"):
            vae_path = train_vae(cfg, train, test, device, root / "vae")
        out = {}
        if cfg.stage in ("wm", "all"):
            out = train_world_model(cfg, vae_path, train, test, device, root / "wm")
        rd.print("task 2 finished for", cfg.dataset)
        return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--data_path", default=None)
    ap.add_argument("--stage", default=None)
    ap.add_argument("--run_root", default=None)
    ap.add_argument("--vae_hours", type=float, default=None)
    ap.add_argument("--wm_hours", type=float, default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--eval_context", type=int, default=None, help="real frames given before generation")
    ap.add_argument("--eval_horizon", type=int, default=None, help="frames generated in evaluation")
    ap.add_argument("--eval_clips", type=int, default=None)
    ap.add_argument("--eval_batch", type=int, default=None, help="clips generated together")
    ap.add_argument("--start_frame", type=int, default=None, help="drop the first N frames of a video dataset")
    ap.add_argument("--gif_clips", type=int, default=None, help="evaluation clips saved as an animated GIF (0 = none)")
    ap.add_argument("--gif_fps", type=float, default=None, help="frame rate of the evaluation GIFs")
    ap.add_argument("--override", default=None)
    args = ap.parse_args(argv)
    cfg = Task2Config()
    if args.smoke:
        cfg = smoke_config(cfg)
    for k in ["dataset", "data_path", "stage", "run_root", "vae_hours", "wm_hours", "max_steps", "device"]:
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)
    if cfg.dataset == "mmnist" and not args.smoke:
        cfg.clip_len, cfg.eval_context, cfg.eval_horizon = 20, 10, 10
    for k in ["eval_context", "eval_horizon", "eval_clips", "eval_batch", "start_frame", "gif_clips", "gif_fps"]:
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)
    cfg.force = args.force
    apply_overrides(cfg, parse_override(args.override))
    if args.smoke:  # smoke tests always start from scratch
        fresh_smoke_run(Path(cfg.run_root) / cfg.dataset, root=cfg.run_root)
    return run(cfg)


if __name__ == "__main__":
    main()
