"""Task 5 - latent GAN + adversarial world model, single images AND videos, frames at >= 1024 px.

One system, two modes (the world-model toggle):
  image mode  (world model OFF): z -> G_img -> latent -> VAE decoder -> 256 px -> SR x4 -> 1024 px image
  video mode  (world model ON):  first latent from G_img (or the latent of a given frame), then the world
              model G_wm rolls the latent forward frame by frame (stochastic, conditioned) -> decode -> SR.

Data: Inter4K (1,000 UHD clips) at 15 fps; 256 px frames for the latent models, native 4K->1024 frames
for the SR stage.  Conditioning: CLIP ViT-L/14 embedding of each clip (language-free LAFITE training),
so both modes accept text prompts at inference.

Stages (resumable): data -> vae (per-frame beta-VAE, f=8 -> 32x32x16 latents, Task 2 trainer)
-> worldgan (G_img + G_wm trained jointly with three RpGAN+R1/R2 critics) -> sr (Task 4 trainer)
-> eval (1024 px image metrics, video rollout metrics, mp4 + GIF videos, CLIP score).

Critics of the world-GAN stage:
  D_lat  single-frame latent critic  (real frame latents vs G_img outputs and world-model predictions)
  D_img  decoded-frame critic        (Dec(real latents) vs Dec(G_img outputs); decoded vs decoded)
  D_T    temporal critic             (windows [z_{t-3..t}, z_{t+1}]: real continuation vs predicted one)
The world model also gets a decaying MSE anchor to the true future and is trained on its own
rollouts (self-forcing), so errors it makes during generation are seen and penalised in training.

Usage (Colab):
    python -m cocogoat.tasks.task5_world_gan
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cocogoat.data.inter4k import decode_clips, download_clips, make_fake_clips
from cocogoat.data.video import ClipBank
from cocogoat.metrics.fid import frechet_distance, gaussian_stats, kid, precision_recall
from cocogoat.metrics.video import psnr
from cocogoat.models.gan2d import Discriminator, DiscriminatorConfig
from cocogoat.models.latent_gan import LatentGenerator, LatentGeneratorConfig
from cocogoat.models.sr import SRConfig, SRNet
from cocogoat.models.vae import VAEConfig, load_vae
from cocogoat.models.world import WorldModel, WorldModelConfig
from cocogoat.nn import count_params
from cocogoat.tasks.common import (
    DRIVE_ROOT,
    fresh_smoke_run,
    LOCAL_CACHE,
    apply_overrides,
    load_compatible,
    make_feature_fn,
    parse_override,
    resume_or_skip,
    rotate_old_error,
    summarize_metrics_md,
)
from cocogoat.tasks.task1_gan64 import PixelCritic
from cocogoat.tasks.task2_world import Task2Config, train_vae
from cocogoat.tasks.task3_latent_gan import LatentCritic
from cocogoat.tasks.task4_hires import EVAL_PROMPTS as _FACE_PROMPTS  # noqa: F401  (kept for reference)
from cocogoat.tasks.task4_hires import Task4Config, gap_prompt_bank, perturb, train_sr
from cocogoat.train import (
    EMA,
    Every,
    RunDir,
    TrainClock,
    autocast,
    cosine_interp,
    describe_env,
    ema_beta,
    log_cosine_interp,
    rng_state,
    save_grid,
    seed_everything,
    set_optimizer_hparams,
    set_rng_state,
    setup_torch,
)
from cocogoat.train.adversarial import AdversarialTrainer, CriticSlot, make_adam, to_floats

SCENE_PROMPTS = [
    "a drone shot flying over a green forest",
    "ocean waves crashing on a rocky beach",
    "a busy city street at night with car lights",
    "snowy mountains under a blue sky",
    "a slow pan across a field of flowers",
    "clouds moving over a lake at sunset",
    "a waterfall in a tropical jungle",
    "people walking in a market square",
]


def scene_prompt_bank() -> list[str]:
    subjects = ["a city", "a forest", "the ocean", "mountains", "a river", "a beach", "a street", "a field",
                "a building", "people", "a car", "the sky", "a lake", "a park"]
    styles = ["a video of {}", "a drone shot of {}", "a time-lapse of {}", "a slow pan over {}", "footage of {}"]
    return [s.format(x) for s in styles for x in subjects]


def _vae_stage() -> Task2Config:
    c = Task2Config()
    c.vae = VAEConfig(channels=(64, 128, 256, 256), blocks=(1, 2, 2, 2))
    c.vae_hours = 1.5
    c.vae_batch = 32
    c.vae_D = DiscriminatorConfig(channels={256: 32, 128: 64, 64: 128, 32: 256, 16: 256, 8: 256, 4: 256}, blocks=1,
                                  expansion=2.0, attn_res=(16,))
    c.vae_gamma = 5.0
    return c


@dataclass
class Task5Config:
    run_dir: str = str(DRIVE_ROOT / "runs" / "task5")
    stage: str = "all"                     # data | vae | worldgan | sr | eval | all
    local_clips: str | None = None
    max_clips: int = 0                     # 0 = all 1000
    low_res: int = 256
    stride: int = 4                        # 60 fps -> 15 fps
    frames_per_clip: int = 72
    hr_res: int = 1024
    hr_every: int = 8
    holdout_clips: int = 50
    clip_cond: str = "openai/clip-vit-large-patch14"
    clip_eval: str = "openai/clip-vit-base-patch32"
    cond_noise: float = 0.25
    vae_stage: Task2Config = field(default_factory=_vae_stage)
    # ---- world GAN ----
    G_img: LatentGeneratorConfig = field(default_factory=lambda: LatentGeneratorConfig(
        latent_channels=16, latent_size=32, cond_vector_dim=768, width=384, depth=8, heads=6, expansion=3.0,
        attn_every=2))
    wm: WorldModelConfig = field(default_factory=lambda: WorldModelConfig(
        latent_channels=16, latent_size=32, patch=2, dim=512, depth=8, heads=8, max_frames=8, cond_dim=768,
        noise_dim=16))
    D_lat: DiscriminatorConfig = field(default_factory=lambda: DiscriminatorConfig(
        resolution=32, in_channels=16, channels={32: 256, 16: 384, 8: 512, 4: 512}, blocks=2, attn_res=(16, 8),
        expansion=2.0, cond_vector_dim=768))
    D_img: DiscriminatorConfig = field(default_factory=lambda: DiscriminatorConfig(
        resolution=256, channels={256: 32, 128: 64, 64: 128, 32: 256, 16: 384, 8: 512, 4: 512},
        blocks={256: 1, 128: 1, 64: 2, 32: 2, 16: 2, 8: 2, 4: 2}, expansion=2.0, attn_res=(16,), cond_vector_dim=768))
    D_T: DiscriminatorConfig = field(default_factory=lambda: DiscriminatorConfig(
        resolution=32, in_channels=64, channels={32: 256, 16: 384, 8: 512, 4: 512}, blocks=2, attn_res=(16, 8),
        expansion=2.0, cond_vector_dim=768))
    window: int = 4                        # frames seen by D_T (window-1 context + 1 next)
    ctx: int = 4                           # real context frames for world-model training rollouts
    roll: int = 4                          # self-forcing rollout steps per training clip
    batch: int = 32
    gan_hours: float = 5.5
    lr: tuple = (2e-4, 5e-5)
    beta2: tuple = (0.9, 0.99)
    gamma_lat: tuple = (1.0, 0.1)
    gamma_img: tuple = (30.0, 3.0)
    gamma_T: tuple = (1.0, 0.1)
    mse_weight: tuple = (1.0, 0.05)        # world-model regression anchor, decays as critics take over
    ot_weight: tuple = (0.1, 0.01)
    ctx_noise: float = 0.05
    ema_kimg: tuple = (10.0, 200.0)
    aug_p: tuple = (0.2, 0.5)
    reg_every: int = 8
    amp: bool = True
    # ---- SR (Task 4 trainer on Inter4K HR frames) ----
    sr: SRConfig = field(default_factory=SRConfig)
    sr_hours: float = 2.0
    # ---- eval / cadence ----
    eval_images: int = 5000
    eval_videos: int = 48
    video_frames: int = 24                 # length of generated videos in the final evaluation
    eval_horizon: int = 8                  # frames the world model generates in periodic evaluations
    eval_clips: int = 256                  # held-out clips for those evaluations (capped by memory)
    fps: int = 15
    log_every: int = 50
    sample_minutes: float = 20.0
    eval_minutes: float = 60.0
    ckpt_minutes: float = 20.0
    metrics: str = "inception"
    seed: int = 0
    device: str | None = None
    smoke: bool = False
    force: bool = False
    max_steps: int = 0


def smoke_config(cfg: Task5Config) -> Task5Config:
    from cocogoat.tasks import task2_world

    cfg.smoke = True
    cfg.low_res, cfg.hr_res, cfg.frames_per_clip, cfg.hr_every, cfg.holdout_clips = 64, 128, 12, 4, 1
    cfg.vae_stage = task2_world.smoke_config(Task2Config())
    cfg.vae_stage.max_steps = 4
    cfg.G_img = LatentGeneratorConfig(latent_channels=16, latent_size=8, z_global_dim=8, w_dim=16, cond_dim=8,
                                      cond_vector_dim=768, width=16, depth=1, heads=2, expansion=2.0)
    cfg.wm = WorldModelConfig(latent_channels=16, latent_size=8, patch=1, dim=16, depth=1, heads=2, max_frames=8,
                              cond_dim=768, noise_dim=4)
    small = dict(blocks=1, expansion=2.0, heads=2, cond_vector_dim=768)
    cfg.D_lat = DiscriminatorConfig(resolution=8, in_channels=16, channels={8: 16, 4: 16}, attn_res=(8,), **small)
    cfg.D_img = DiscriminatorConfig(resolution=64, channels={64: 16, 32: 16, 16: 16, 8: 16, 4: 16}, attn_res=(8,),
                                    **small)
    cfg.D_T = DiscriminatorConfig(resolution=8, in_channels=64, channels={8: 16, 4: 16}, attn_res=(8,), **small)
    cfg.sr = SRConfig(channels=16, blocks=1, hr_channels=8, hr_blocks=1)
    cfg.batch, cfg.max_steps = 2, 3
    cfg.log_every = 1
    cfg.sample_minutes = cfg.eval_minutes = cfg.ckpt_minutes = 0.0
    cfg.eval_images, cfg.eval_videos, cfg.video_frames = 8, 2, 6
    cfg.metrics = "random"
    cfg.amp = False
    return cfg


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------

def prepare_data(cfg: Task5Config, rd: RunDir, device):
    cache = LOCAL_CACHE / "inter4k"
    if cfg.local_clips:
        clips = sorted(Path(cfg.local_clips).glob("*.mp4"))
    else:
        t0 = time.time()
        clips = download_clips(cache, cfg.max_clips or None)
        rd.print(f"downloaded {len(clips)} clips in {(time.time() - t0) / 60:.1f} min")
    t0 = time.time()
    low, seg, hr, hr_clip = decode_clips(clips, cache / "decoded", cfg.low_res, cfg.stride, cfg.frames_per_clip,
                                         cfg.hr_res, cfg.hr_every)
    n_clips = int(seg.max()) + 1
    rd.print(f"decoded {n_clips} clips: low {low.shape}, hr {hr.shape} in {(time.time() - t0) / 60:.1f} min")
    emb_path, gap_path = rd.path / "clip_emb.npy", rd.path / "clip_gap.npz"
    if emb_path.exists() and gap_path.exists() and np.load(emb_path).shape[0] == n_clips:
        emb = np.load(emb_path).astype(np.float32)
        with np.load(gap_path) as z:
            gap = {"mu_img": z["mu_img"], "mu_txt": z["mu_txt"]}
        return low, seg, hr, hr_clip, emb, gap
    mid = np.arange(n_clips) * cfg.frames_per_clip + cfg.frames_per_clip // 2
    if cfg.smoke:
        emb = np.random.default_rng(0).normal(size=(n_clips, 768)).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        gap = {"mu_img": emb.mean(0), "mu_txt": np.zeros(768, np.float32)}
    else:
        from cocogoat.models.pretrained import CLIPEncoder

        enc = CLIPEncoder(cfg.clip_cond, device)
        out = []
        for i in range(0, n_clips, 256):
            x = torch.from_numpy(np.ascontiguousarray(low[mid[i:i + 256]])).to(device).permute(0, 3, 1, 2)
            out.append(enc.encode_images(x.float() / 127.5 - 1).cpu())
        emb = torch.cat(out).numpy().astype(np.float32)
        gap = {"mu_img": emb.mean(0), "mu_txt": enc.encode_text(scene_prompt_bank() + gap_prompt_bank()[:0])
               .mean(0).cpu().numpy()}
        del enc
        torch.cuda.empty_cache()
    np.save(emb_path, emb)
    np.savez(gap_path, **gap)
    return low, seg, hr, hr_clip, emb, gap


# --------------------------------------------------------------------------------------
# world-GAN stage
# --------------------------------------------------------------------------------------

class WindowCritic(LatentCritic):
    """Temporal critic: input = window of latent frames stacked along channels."""

    keys = ("win",)

    def encode(self, sample):
        return {"win": sample["win"]}

    def head(self, enc, cond):
        net = self.D_fast if (self.fast and self.D_fast is not None) else self.D
        return net(enc["win"], cond)


def train_worldgan(cfg: Task5Config, vae_path: Path, low, seg, emb, device, run_dir: Path) -> Path:
    rd = RunDir(run_dir, config=cfg)
    out_path = run_dir / "world_gan.pt"
    with rd.capture_errors():
        if resume_or_skip(rd, cfg.gan_hours, cfg.force) and out_path.exists():
            return out_path
        vae = load_vae(str(vae_path), device).requires_grad_(False).eval()
        lc, ls = vae.cfg.latent_channels, cfg.low_res // vae.cfg.downsample
        n_clips = int(seg.max()) + 1
        train_clips = n_clips - cfg.holdout_clips
        train_mask = seg < train_clips
        bank = ClipBank(np.asarray(low[train_mask]), seg[train_mask], device)
        test = ClipBank(np.asarray(low[~train_mask]), seg[~train_mask], device)
        emb_t = torch.from_numpy(emb).to(device)
        T = cfg.ctx + cfg.roll

        cfg.G_img.latent_channels, cfg.G_img.latent_size = lc, ls
        cfg.wm.latent_channels, cfg.wm.latent_size = lc, ls
        cfg.wm.max_frames = max(cfg.wm.max_frames, T)
        cfg.D_lat.resolution, cfg.D_lat.in_channels = ls, lc
        cfg.D_T.resolution, cfg.D_T.in_channels = ls, lc * cfg.window
        cfg.D_img.resolution = cfg.low_res
        G_img = LatentGenerator(cfg.G_img).to(device)
        G_wm = WorldModel(cfg.wm).to(device)
        D_lat, D_img, D_T = Discriminator(cfg.D_lat).to(device), Discriminator(cfg.D_img).to(device), \
            Discriminator(cfg.D_T).to(device)
        E_img, E_wm = EMA(G_img), EMA(G_wm)
        rd.print(f"G_img {count_params(G_img) / 1e6:.1f}M, G_wm {count_params(G_wm) / 1e6:.1f}M, D_lat "
                 f"{count_params(D_lat) / 1e6:.1f}M, D_img {count_params(D_img) / 1e6:.1f}M, D_T "
                 f"{count_params(D_T) / 1e6:.1f}M; latents {lc}x{ls}x{ls}; train clips {train_clips}")
        opt_G = make_adam(list(G_img.parameters()) + list(G_wm.parameters()), cfg.lr[0], cfg.beta2[0])
        opts = {k: make_adam(m.parameters(), cfg.lr[0], cfg.beta2[0]) for k, m in
                (("lat", D_lat), ("img", D_img), ("T", D_T))}
        c_lat = LatentCritic(D_lat, "translation", cfg.aug_p[0])
        c_img = PixelCritic(D_img, "color,translation,cutout", cfg.aug_p[0])
        c_T = WindowCritic(D_T, "", 0.0)
        slots = [CriticSlot("lat", c_lat, opts["lat"], gamma=cfg.gamma_lat[0]),
                 CriticSlot("img", c_img, opts["img"], gamma=cfg.gamma_img[0]),
                 CriticSlot("T", c_T, opts["T"], gamma=cfg.gamma_T[0])]
        trainer = AdversarialTrainer(slots, device, amp=cfg.amp, reg_every=cfg.reg_every)

        step, nimg, elapsed, best = 0, 0, 0.0, float("inf")
        st = rd.load_checkpoint()
        if load_compatible(rd, {"G_img": G_img, "G_wm": G_wm, "D_lat": D_lat, "D_img": D_img, "D_T": D_T,
                                "E_img": E_img, "E_wm": E_wm, "opt_G": opt_G, "opt_lat": opts["lat"],
                                "opt_img": opts["img"], "opt_T": opts["T"]}, st):
            step, nimg, elapsed, best = st["step"], st["nimg"], st["elapsed"], st.get("best", best)
            set_rng_state(st["rng"])
            rd.print(f"resumed world-GAN at step {step}")

        def encode(clips):  # (B,T,3,H,W) -> (B,T,C,h,w) normalised latents (posterior means)
            b, t = clips.shape[:2]
            with torch.no_grad(), autocast(device, cfg.amp):
                z = vae.encode_normalized(clips.flatten(0, 1))
            return z.float().view(b, t, lc, ls, ls)

        def decode(z):
            return vae.decode_normalized(z)

        def clip_cond(starts_seg: torch.Tensor, perturbed: bool = True):
            c = emb_t[starts_seg]
            return perturb(c, cfg.cond_noise) if perturbed else F.normalize(c, dim=-1)

        def sample_clips(b):
            starts = bank.valid_starts(T)
            pick = starts[torch.randint(len(starts), (b,), device=device)]
            idx = pick[:, None] + torch.arange(T, device=device)[None]
            x = bank.to_float(bank.frames[idx])
            flip = torch.rand(b, device=device) < 0.5
            x = torch.where(flip[:, None, None, None, None], x.flip(4), x)
            return x, bank.seg[pick]

        def rollout(g_wm, ctx_z, steps, c):
            seq, preds = ctx_z, []
            for _ in range(steps):
                nxt = g_wm(seq[:, -cfg.wm.max_frames:], c)[:, -1:].float()
                preds.append(nxt)
                seq = torch.cat([seq, nxt], dim=1)
            return torch.cat(preds, dim=1), seq

        def windows(seq):  # (B,T,C,h,w) -> (B*roll, window*C, h, w) for windows ending at predicted positions
            w = cfg.window
            outs = [seq[:, k - w + 1:k + 1].flatten(1, 2) for k in range(cfg.ctx, T)]
            return torch.stack(outs, 1).flatten(0, 1)

        def make_batch(g_img, g_wm, with_grad: bool):
            x, segs = sample_clips(cfg.batch)
            z = encode(x)
            c = clip_cond(segs)
            ctx_z = z[:, :cfg.ctx] + cfg.ctx_noise * torch.randn_like(z[:, :cfg.ctx])
            ctxm = torch.enable_grad() if with_grad else torch.no_grad()
            noise = torch.randn(cfg.batch, lc, ls, ls, device=device)
            with ctxm, autocast(device, cfg.amp):
                zi = g_img(noise, torch.randn(cfg.batch, cfg.G_img.z_global_dim, device=device), c).float()
                preds, seq = rollout(g_wm, ctx_z, cfg.roll, c)
                xi = decode(zi).float()
            real_seq = torch.cat([z[:, :cfg.ctx], z[:, cfg.ctx:]], dim=1)
            j = torch.randint(T, (cfg.batch,), device=device)
            real_frames = z[torch.arange(cfg.batch, device=device), j]
            jp = torch.randint(cfg.roll, (cfg.batch,), device=device)
            pred_frames = preds[torch.arange(cfg.batch, device=device), jp]
            with torch.no_grad(), autocast(device, cfg.amp):
                x_real_dec = decode(real_frames).float()
            fake_seq = torch.cat([z[:, :cfg.ctx], preds], dim=1)
            real = {"lat": torch.cat([real_frames, z[torch.arange(cfg.batch, device=device), cfg.ctx + jp]]),
                    "img": x_real_dec, "win": windows(real_seq)}
            fake = {"lat": torch.cat([zi, pred_frames]), "img": xi, "win": windows(fake_seq)}
            cond = {"lat": torch.cat([c, c]), "img": c, "T": c.repeat_interleave(cfg.roll, 0)}
            extra = {"preds": preds, "target": z[:, cfg.ctx:], "zi": zi, "noise": noise}
            return real, fake, cond, extra

        H = cfg.eval_horizon
        n_avail = len(test.valid_starts(cfg.ctx + H))
        n_hold = max(1, min(cfg.eval_clips, n_avail, 3072 // (cfg.ctx + H)))  # <= ~3k frames (2.4 GB at 256 px)
        holdout = test.fixed_clips(n_hold, cfg.ctx + H, seed=0) if n_avail else None
        if holdout is None:
            rd.print(f"no held-out clip has {cfg.ctx + H} frames: world-model rollout evaluation skipped")
        holdout_seg = None
        if holdout is not None:
            starts = test.valid_starts(cfg.ctx + H).cpu()
            g = torch.Generator().manual_seed(0)
            sel = starts[torch.randperm(len(starts), generator=g)[:holdout.shape[0]]]
            holdout_seg = test.seg[sel.to(device)]
        feat = make_feature_fn(cfg.metrics, device)
        real_feats = {}

        def real_frame_feats(n=5000):
            if "f" not in real_feats:
                idx = torch.randint(len(bank), (n,), device=device)
                fs = [feat(bank.to_float(bank.frames[idx[i:i + 250]])).cpu().numpy() for i in range(0, n, 250)]
                real_feats["f"] = np.concatenate(fs)
            return real_feats["f"]

        @torch.no_grad()
        def evaluate(tag):
            nonlocal best
            G_img_e, G_wm_e = E_img.model, E_wm.model
            m = {}
            n = min(cfg.eval_images, 5000)
            fs = []
            for i in range(0, n, 250):
                b = min(250, n - i)
                c = F.normalize(emb_t[torch.randint(len(emb_t), (b,), device=device)], dim=-1)
                with autocast(device, cfg.amp):
                    zi = G_img_e(torch.randn(b, lc, ls, ls, device=device),
                                 torch.randn(b, cfg.G_img.z_global_dim, device=device), c)
                    fs.append(feat(decode(zi).float().clamp(-1, 1)).cpu().numpy())
            fs = np.concatenate(fs)
            rf = real_frame_feats()
            mu, s = gaussian_stats(rf)
            m["image_fid_256"] = frechet_distance(mu, s, *gaussian_stats(fs))
            m["image_kid_256_x1000"] = kid(rf, fs, subset_size=min(1000, len(fs)))[0] * 1000
            if holdout is not None:
                z = encode(holdout)
                c = clip_cond(holdout_seg, perturbed=False)
                with autocast(device, cfg.amp):
                    preds, _ = rollout(G_wm_e, z[:, :cfg.ctx], H, c)
                gt = z[:, cfg.ctx:]
                copy = z[:, cfg.ctx - 1:cfg.ctx].expand_as(gt)
                m["wm_latent_mse"] = (preds - gt).square().mean(dim=(0, 2, 3, 4)).tolist()
                m["copy_latent_mse"] = (copy - gt).square().mean(dim=(0, 2, 3, 4)).tolist()
                with autocast(device, cfg.amp):
                    xp = decode(preds[:, -1]).float().clamp(-1, 1)
                    xg = decode(gt[:, -1]).float().clamp(-1, 1)
                m[f"wm_psnr_t{H}_vs_decoded_gt"] = float(psnr(xp, xg).mean())
                pf = feat(xp).cpu().numpy()
                m[f"wm_frame_kid_t{H}_x1000"] = kid(rf, pf, subset_size=min(1000, len(pf)))[0] * 1000
                nr = min(4, preds.shape[0])
                with autocast(device, cfg.amp):
                    pd = decode(preds[:nr].flatten(0, 1)).float().clamp(-1, 1).view(nr, H, 3, cfg.low_res, cfg.low_res)
                rows = []
                for r in range(nr):  # per clip: real context + real future, then real context + predicted future
                    rows += [torch.cat([holdout[r, :cfg.ctx], holdout[r, cfg.ctx:]]),
                             torch.cat([holdout[r, :cfg.ctx], pd[r]])]
                save_grid(torch.cat(rows), rd.path / "samples" / f"rollout_{tag}.png", nrow=cfg.ctx + H, max_side=8192)
            m.update(step=step, kimg=nimg / 1000, hours=clock.now() / 3600, tag=tag)
            rd.log_metrics(kind="eval", **m)
            rd.print(f"[world-gan eval {tag}] " + " ".join(
                f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items() if not isinstance(v, list)))
            if m["image_fid_256"] < best:
                best = m["image_fid_256"]
            return m

        def checkpoint():
            rd.save_checkpoint({"G_img": G_img.state_dict(), "G_wm": G_wm.state_dict(), "D_lat": D_lat.state_dict(),
                                "D_img": D_img.state_dict(), "D_T": D_T.state_dict(), "E_img": E_img.state_dict(),
                                "E_wm": E_wm.state_dict(), "opt_G": opt_G.state_dict(),
                                "opt_lat": opts["lat"].state_dict(), "opt_img": opts["img"].state_dict(),
                                "opt_T": opts["T"].state_dict(), "step": step, "nimg": nimg, "elapsed": clock.now(),
                                "best": best, "rng": rng_state()})

        budget = cfg.gan_hours * 3600
        clock = TrainClock(elapsed)
        timers = (Every(cfg.sample_minutes * 60), Every(cfg.eval_minutes * 60), Every(cfg.ckpt_minutes * 60))
        t_log, n_log, bad = time.time(), 0, 0
        clock.start()
        while True:
            progress = clock.now() / budget
            if (cfg.max_steps and step >= cfg.max_steps) or (not cfg.max_steps and progress >= 1):
                break
            lr = cosine_interp(*cfg.lr, progress)
            beta2 = cosine_interp(*cfg.beta2, progress)
            for o in [opt_G, *opts.values()]:
                set_optimizer_hparams(o, lr=lr, beta2=beta2)
            slots[0].gamma = log_cosine_interp(*cfg.gamma_lat, progress)
            slots[1].gamma = log_cosine_interp(*cfg.gamma_img, progress)
            slots[2].gamma = log_cosine_interp(*cfg.gamma_T, progress)
            c_lat.p = c_img.p = cosine_interp(*cfg.aug_p, progress)
            mse_w = log_cosine_interp(*cfg.mse_weight, progress)
            ot_w = log_cosine_interp(*cfg.ot_weight, progress)
            # critics
            real, fake, cond, _ = make_batch(G_img, G_wm, with_grad=False)
            d_stats = trainer.d_step(real, {k: v.detach() for k, v in fake.items()}, cond, step)
            # generators
            real, fake, cond, extra = make_batch(G_img, G_wm, with_grad=True)
            adv, g_stats = trainer.g_adversarial_loss(real, fake, cond)
            mse = (extra["preds"] - extra["target"]).square().mean()
            ot = (extra["zi"] - extra["noise"]).square().mean()
            loss = adv + mse_w * mse + ot_w * ot
            opt_G.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(G_img.parameters()) + list(G_wm.parameters()), 10.0)
            opt_G.step()
            nimg += cfg.batch
            step += 1
            beta = ema_beta(cfg.batch, cosine_interp(*cfg.ema_kimg, progress), nimg)
            E_img.update(G_img, beta)
            E_wm.update(G_wm, beta)
            n_log += 1
            if step % cfg.log_every == 0:
                dt = time.time() - t_log
                stats = {**to_floats(d_stats), **to_floats(g_stats), "wm_mse": float(mse), "mse_w": mse_w,
                         "ot": float(ot), "ot_w": ot_w, "lr": lr,
                         "it_per_s": n_log / max(dt, 1e-9)}
                bad = bad + 1 if not all(math.isfinite(v) for v in stats.values()) else 0
                if bad >= 3:
                    raise FloatingPointError("world-GAN diverged (non-finite losses); resume from last checkpoint")
                rd.log_metrics(kind="train", step=step, kimg=nimg / 1000, hours=clock.now() / 3600, **stats)
                rd.write_status(stage="worldgan", step=step, hours=clock.now() / 3600, budget_hours=cfg.gan_hours,
                                **stats)
                if step % (cfg.log_every * 10) == 0 or cfg.max_steps:
                    rd.print(f"wgan step {step} " + " ".join(f"{k}={v:.4g}" for k, v in stats.items()))
                t_log, n_log = time.time(), 0
            if timers[0]() or (cfg.max_steps and step == cfg.max_steps):
                clock.pause()
                with torch.no_grad(), autocast(device, cfg.amp):
                    c = F.normalize(emb_t[:16 % len(emb_t) or 16][:16], dim=-1)
                    c = c[torch.arange(16, device=device) % len(c)]
                    zi = E_img.model(torch.randn(16, lc, ls, ls, device=device),
                                     torch.randn(16, cfg.G_img.z_global_dim, device=device), c)
                    save_grid(decode(zi).float().clamp(-1, 1), rd.path / "samples" / f"images_step{step:06d}.png",
                              nrow=4, max_side=2048)
                clock.start()
            if timers[1]():
                clock.pause()
                evaluate(f"step{step:06d}")
                clock.start()
            if timers[2]():
                clock.pause()
                checkpoint()
                clock.start()
        clock.pause()
        checkpoint()
        final = evaluate("final")
        torch.save({"E_img": E_img.state_dict(), "E_wm": E_wm.state_dict(),
                    "G_img_config": dataclasses.asdict(cfg.G_img), "wm_config": dataclasses.asdict(cfg.wm),
                    "metrics": final, "vae_path": str(vae_path)}, out_path)
        rd.mark_done({"final": final, "budget_hours": cfg.gan_hours, "steps": step})
        del bank, test, trainer
        gc.collect()
        torch.cuda.empty_cache()
        return out_path


# --------------------------------------------------------------------------------------
# generation API (the "one system with a world-model toggle")
# --------------------------------------------------------------------------------------

class WorldGANSystem:
    """Loads VAE + G_img + world model + SR and generates images or videos."""

    def __init__(self, world_path: Path, sr_path: Path, device, amp: bool = True):
        st = torch.load(world_path, map_location="cpu", weights_only=False)
        self.vae = load_vae(st["vae_path"], device).requires_grad_(False).eval()
        self.G_img = LatentGenerator(LatentGeneratorConfig(**st["G_img_config"])).to(device).eval()
        self.G_img.load_state_dict(st["E_img"])
        self.G_wm = WorldModel(WorldModelConfig(**st["wm_config"])).to(device).eval()
        self.G_wm.load_state_dict(st["E_wm"])
        ss = torch.load(sr_path, map_location="cpu", weights_only=False)
        self.sr = SRNet(SRConfig(**ss["sr_config"])).to(device).eval()
        self.sr.load_state_dict(ss["net"])
        self.device, self.amp = device, amp
        self.lc, self.ls = self.G_img.cfg.latent_channels, self.G_img.cfg.latent_size

    @torch.no_grad()
    def first_latents(self, c: torch.Tensor) -> torch.Tensor:
        b = c.shape[0]
        with autocast(self.device, self.amp):
            return self.G_img(torch.randn(b, self.lc, self.ls, self.ls, device=self.device),
                              torch.randn(b, self.G_img.cfg.z_global_dim, device=self.device), c).float()

    @torch.no_grad()
    def render(self, z: torch.Tensor, hires: bool = True) -> torch.Tensor:
        with autocast(self.device, self.amp):
            x = self.vae.decode_normalized(z).float().clamp(-1, 1)
            return self.sr(x).float().clamp(-1, 1) if hires else x

    @torch.no_grad()
    def generate(self, c: torch.Tensor, frames: int = 1, world_model: bool = True, hires: bool = True,
                 first_latent: torch.Tensor | None = None) -> torch.Tensor:
        """frames=1 or world_model=False -> images (B,3,H,W); otherwise videos (B,T,3,H,W)."""
        z0 = self.first_latents(c) if first_latent is None else first_latent
        if frames <= 1 or not world_model:
            return self.render(z0, hires)
        seq = z0[:, None]
        w = self.G_wm.cfg.max_frames
        while seq.shape[1] < frames:
            with autocast(self.device, self.amp):
                nxt = self.G_wm(seq[:, -w:], c)[:, -1:].float()
            seq = torch.cat([seq, nxt], dim=1)
        out = [self.render(seq[:, t], hires) for t in range(frames)]
        return torch.stack(out, dim=1)


def write_video(frames: torch.Tensor, path: Path, fps: int = 15) -> None:
    """(T,3,H,W) in [-1,1] -> mp4 (cv2) ; also a small GIF preview next to it."""
    import cv2
    from PIL import Image

    arr = ((frames.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    h, w = arr.shape[1:3]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in arr:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    small = [Image.fromarray(f).resize((256, 256), Image.BILINEAR) for f in arr]
    small[0].save(path.with_suffix(".gif"), save_all=True, append_images=small[1:], duration=int(1000 / fps), loop=0)


@torch.no_grad()
def final_eval(cfg: Task5Config, low, seg, hr, hr_clip, emb, gap, world_path: Path, sr_path: Path, device,
               run_dir: Path) -> dict:
    rd = RunDir(run_dir, config=cfg)
    with rd.capture_errors():
        sys_ = WorldGANSystem(world_path, sr_path, device, cfg.amp)
        feat = make_feature_fn(cfg.metrics, device)
        emb_t = F.normalize(torch.from_numpy(emb).to(device), dim=-1)
        n_clips = len(emb)
        hold = torch.arange(n_clips - cfg.holdout_clips, n_clips, device=device)
        crop = 299 if cfg.hr_res >= 299 else cfg.hr_res // 2

        def rand_crop(x):
            y = int(torch.randint(0, x.shape[-2] - crop + 1, ()))
            xx = int(torch.randint(0, x.shape[-1] - crop + 1, ()))
            return x[..., y:y + crop, xx:xx + crop]

        valid_hr = np.where(hr_clip >= 0)[0]
        rf_full, rf_patch = [], []
        for i in range(0, min(len(valid_hr), 5000), 32):
            x = torch.from_numpy(np.ascontiguousarray(hr[valid_hr[i:i + 32]])).to(device).permute(0, 3, 1, 2)
            x = x.float() / 127.5 - 1
            rf_full.append(feat(x).cpu().numpy())
            rf_patch.append(feat(rand_crop(x)).cpu().numpy())
        rf_full, rf_patch = np.concatenate(rf_full), np.concatenate(rf_patch)
        # ---- image mode (world model OFF) ----
        ff, fp = [], []
        for i in range(0, cfg.eval_images, 16):
            b = min(16, cfg.eval_images - i)
            c = emb_t[torch.randint(n_clips, (b,), device=device)]
            x = sys_.generate(c, frames=1)
            ff.append(feat(x).cpu().numpy())
            fp.append(feat(rand_crop(x)).cpu().numpy())
        ff, fp = np.concatenate(ff), np.concatenate(fp)
        m = {}
        sub = min(1000, len(rf_full), len(ff))
        mu, s = gaussian_stats(rf_full)
        m["image_fid_1024"] = frechet_distance(mu, s, *gaussian_stats(ff))
        m["image_kid_1024_x1000"] = kid(rf_full, ff, subset_size=sub)[0] * 1000
        m["image_patch_kid_x1000"] = kid(rf_patch, fp, subset_size=min(1000, len(rf_patch), len(fp)))[0] * 1000
        p, r = precision_recall(rf_full[:sub * 3], ff[:sub * 3])
        m["image_precision"], m["image_recall"] = p, r
        # ---- video mode (world model ON): frame quality over time + temporal smoothness ----
        vids_dir = rd.path / "videos"
        vids_dir.mkdir(parents=True, exist_ok=True)
        late_feats, step_lpips_gen = [], []
        n_v = cfg.eval_videos
        for i in range(0, n_v, 4):
            b = min(4, n_v - i)
            c = emb_t[hold[torch.randint(len(hold), (b,), device=device)]]
            v = sys_.generate(c, frames=cfg.video_frames)            # (b,T,3,H,W) at hr
            late = v[:, cfg.video_frames // 2:].flatten(0, 1)
            late_feats.append(feat(late).cpu().numpy())
            step_lpips_gen.append((v[:, 1:] - v[:, :-1]).abs().mean().item())
            if i == 0:
                for j in range(b):
                    write_video(v[j], vids_dir / f"holdout_cond_{j}.mp4", cfg.fps)
        late_feats = np.concatenate(late_feats)
        m["video_late_frame_kid_1024_x1000"] = kid(rf_full, late_feats, subset_size=min(1000, len(late_feats)))[0] * 1000
        m["video_mean_abs_frame_diff_1024"] = float(np.mean(step_lpips_gen))
        # temporal change per 15 fps step at 256 px: generated (decoder output) vs real held-out clips
        gen_d = []
        for i in range(0, min(n_v, 16), 4):
            c = emb_t[hold[torch.randint(len(hold), (4,), device=device)]]
            v = sys_.generate(c, frames=cfg.video_frames, hires=False)
            gen_d.append((v[:, 1:] - v[:, :-1]).abs().mean().item())
        hold_frames = np.where(seg >= n_clips - cfg.holdout_clips)[0]
        real_d = []
        for k in hold_frames[:-1:7][:200]:
            if seg[k] == seg[k + 1]:
                xa = torch.from_numpy(np.asarray(low[k])).float() / 127.5 - 1
                xb = torch.from_numpy(np.asarray(low[k + 1])).float() / 127.5 - 1
                real_d.append((xa - xb).abs().mean().item())
        m["video_mean_abs_frame_diff_256"] = float(np.mean(gen_d))
        m["real_mean_abs_frame_diff_256"] = float(np.mean(real_d)) if real_d else float("nan")
        # ---- prompts: images + videos ----
        prompts = SCENE_PROMPTS
        if not cfg.smoke:
            from cocogoat.models.pretrained import CLIPEncoder
            from cocogoat.tasks.task4_hires import text_conditions

            c_txt = text_conditions(prompts, gap, cfg.clip_cond, device)
            judge = CLIPEncoder(cfg.clip_eval, device)
            t_eval = judge.encode_text(prompts)
        else:
            c_txt, judge, t_eval = F.normalize(torch.randn(len(prompts), 768, device=device), dim=-1), None, None
        grid, match, other = [], [], []
        for j in range(len(prompts)):
            imgs = sys_.generate(c_txt[j:j + 1].expand(4, -1), frames=1)
            grid.append(F.interpolate(imgs, size=(256, 256), mode="bicubic", antialias=True))
            if judge is not None:
                e = judge.encode_images(imgs)
                sims = e @ t_eval.T
                match.append(float(sims[:, j].mean()))
                other.append(float(torch.cat([sims[:, :j], sims[:, j + 1:]], 1).mean()))
            vid = sys_.generate(c_txt[j:j + 1], frames=cfg.video_frames)[0]
            write_video(vid, vids_dir / f"prompt{j}.mp4", cfg.fps)
            save_grid(F.interpolate(vid[::max(1, cfg.video_frames // 8)], size=(256, 256), mode="bicubic",
                                    antialias=True), rd.path / "samples" / f"prompt{j}_strip.png", nrow=8)
        save_grid(torch.cat(grid), rd.path / "samples" / "prompts_images.png", nrow=4)
        if match:
            m["clip_score_matched"], m["clip_score_mismatched"] = float(np.mean(match)), float(np.mean(other))
        rd.write_json("final_metrics.json", m)
        (rd.path / "report.md").write_text(summarize_metrics_md(
            "Task 5 - latent world-GAN at 1024 px (image + video modes)", m,
            ["image_*: world model OFF (G_img -> decoder -> SR), vs real Inter4K frames at 1024 (Inception at 299, "
             "antialiased); image_patch_kid: native-resolution 299 px crops.",
             "video_*: world model ON; late-frame KID uses the second half of each generated clip (drift check); "
             "mean abs frame diff (256 px): generated vs real temporal change per 15 fps step (too low = frozen "
             "video, too high = flicker).",
             "Videos: videos/*.mp4 (+ .gif previews); prompt strips: samples/prompt*_strip.png; "
             "prompt images: samples/prompts_images.png.", "Prompts: " + "; ".join(prompts)]))
        rd.mark_done(m)
        rd.print("final: " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()))
        return m


def run(cfg: Task5Config) -> dict:
    device = setup_torch(cfg.device)
    seed_everything(cfg.seed)
    root = Path(cfg.run_dir)
    for sub in ("", "vae", "worldgan", "sr", "eval"):
        (root / sub).mkdir(parents=True, exist_ok=True)
        rotate_old_error(root / sub)
    rd = RunDir(root, config=cfg)
    with rd.capture_errors():
        rd.print(f"env: {describe_env()}")
        if cfg.smoke and not cfg.local_clips:
            cfg.local_clips = str(make_fake_clips(LOCAL_CACHE / "fake_inter4k", 3, 60, (160, 128))[0].parent)
        low, seg, hr, hr_clip, emb, gap = prepare_data(cfg, rd, device)
        if cfg.stage == "data":
            return {}
        vae_path = root / "vae" / "vae.pt"
        world_path = root / "worldgan" / "world_gan.pt"
        sr_path = root / "sr" / "sr.pt"
        n_clips = int(seg.max()) + 1
        if not cfg.smoke and cfg.holdout_clips > n_clips // 10:  # small --max_clips subsets: keep 90% for training
            cfg.holdout_clips = max(1, n_clips // 10)
            rd.print(f"holdout reduced to {cfg.holdout_clips} clips for a {n_clips}-clip subset")
        train_mask = seg < n_clips - cfg.holdout_clips
        if cfg.stage in ("vae", "all"):
            vcfg = cfg.vae_stage
            vcfg.resolution, vcfg.metrics, vcfg.amp, vcfg.force = cfg.low_res, cfg.metrics, cfg.amp, cfg.force
            tr = ClipBank(np.asarray(low[train_mask]), seg[train_mask], device)
            te = ClipBank(np.asarray(low[~train_mask][:2000]), seg[~train_mask][:2000], device)
            vae_path = train_vae(vcfg, tr, te, device, root / "vae")
            del tr, te
            torch.cuda.empty_cache()
        if cfg.stage in ("worldgan", "all"):
            world_path = train_worldgan(cfg, vae_path, low, seg, emb, device, root / "worldgan")
        if cfg.stage in ("sr", "all"):
            t4 = Task4Config(run_dir=str(root), low_res=cfg.low_res, hr_res=cfg.hr_res, sr=cfg.sr,
                             sr_hours=cfg.sr_hours, metrics=cfg.metrics, amp=cfg.amp, force=cfg.force,
                             max_steps=cfg.max_steps, log_every=cfg.log_every, eval_minutes=cfg.eval_minutes,
                             ckpt_minutes=cfg.ckpt_minutes)
            if cfg.smoke:
                t4.sr_D = DiscriminatorConfig(resolution=32, channels={32: 16, 16: 16, 8: 16, 4: 16}, blocks=1,
                                              expansion=2.0, attn_res=(8,), heads=2)
                t4.sr_crop, t4.sr_batch = 32, 2
            hr_train = np.where((hr_clip >= 0) & (hr_clip < n_clips - cfg.holdout_clips))[0]
            sr_path = train_sr(t4, hr[hr_train] if len(hr_train) < len(hr_clip) else hr, device, root / "sr")
        if cfg.stage in ("eval", "all"):
            return final_eval(cfg, low, seg, hr, hr_clip, emb, gap, world_path, sr_path, device, root / "eval")
        return {}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default=None)
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--gan_hours", type=float, default=None)
    ap.add_argument("--sr_hours", type=float, default=None)
    ap.add_argument("--vae_hours", type=float, default=None)
    ap.add_argument("--max_clips", type=int, default=None)
    ap.add_argument("--local_clips", default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--eval_horizon", type=int, default=None, help="frames generated in periodic evaluations")
    ap.add_argument("--video_frames", type=int, default=None, help="length of generated videos (final eval)")
    ap.add_argument("--override", default=None)
    args = ap.parse_args(argv)
    cfg = Task5Config()
    if args.smoke:
        cfg = smoke_config(cfg)
    for k in ["stage", "run_dir", "gan_hours", "sr_hours", "max_clips", "local_clips", "max_steps", "device",
              "eval_horizon", "video_frames"]:
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)
    if args.vae_hours is not None:
        cfg.vae_stage.vae_hours = args.vae_hours
    cfg.force = args.force
    apply_overrides(cfg, parse_override(args.override))
    if args.smoke:  # smoke tests always start from scratch
        fresh_smoke_run(cfg.run_dir)
    return run(cfg)


if __name__ == "__main__":
    main()
