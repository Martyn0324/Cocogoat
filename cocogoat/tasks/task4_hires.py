"""Task 4 - prompt-conditional 1024x1024 GAN trained in ~10 hours on one GPU.

Pipeline (each stage resumes independently):
  data  FFHQ-1024 (70k faces, HF WebDataset mirror) -> 256 px uint8 bank + 15k native 1024 px images;
        CLIP ViT-L/14 image embeddings of every image (the conditioning space).
  base  256x256 GAN conditioned on CLIP embeddings (language-free training, LAFITE: the image
        embedding is perturbed with noise during training; at inference a CLIP *text* embedding,
        shifted by the image/text modality-gap means, is used instead).  Critics: pixel D (R1/R2 on
        pixels) + projected D on frozen DINOv2-S features (R1/R2 on features); RpGAN for both.
  sr    x4 super-resolution GAN 256 -> 1024 trained from scratch on (degraded crop, real crop)
        pairs from the native 1024 images: L1 + LPIPS + RpGAN(+R1/R2).
  eval  10k prompt-free samples at 1024 (conditions = held-out real embeddings) -> KID/FID at 299 px,
        patch-KID/FID on native-resolution 299 px crops, precision/recall; CLIP score for text prompts
        (CLIP B/32 - a different model from the conditioning one); prompt grids + full-res PNGs.

The main generative process is the base GAN; the SR GAN only adds high-frequency detail.

Usage (Colab):
    python -m cocogoat.tasks.task4_hires            # all stages, ~10.5 h
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import itertools
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cocogoat.data import GPUImageBank
from cocogoat.data.images import to_device_nchw
from cocogoat.data.ffhq import decode_shards, download_shards, make_fake_shards
from cocogoat.metrics.fid import ImageEvaluator, frechet_distance, gaussian_stats, kid, precision_recall
from cocogoat.metrics.video import psnr
from cocogoat.models.gan2d import Discriminator, DiscriminatorConfig, Generator, GeneratorConfig
from cocogoat.models.projected import ProjectedCritic, TinyBackbone
from cocogoat.models.sr import SRConfig, SRNet, degrade
from cocogoat.nn import count_params
from cocogoat.tasks.common import (
    DRIVE_ROOT,
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
    save_images,
    seed_everything,
    set_optimizer_hparams,
    set_rng_state,
    setup_torch,
)
from cocogoat.train.adversarial import AdversarialTrainer, CriticSlot, make_adam, maybe_compile, to_floats

HR_CHANNELS = {4: 512, 8: 512, 16: 512, 32: 384, 64: 256, 128: 128, 256: 64}
HR_BLOCKS = {4: 2, 8: 2, 16: 2, 32: 2, 64: 2, 128: 1, 256: 1}
HR_EXPANSION = {4: 3.0, 8: 3.0, 16: 3.0, 32: 3.0, 64: 2.0, 128: 2.0, 256: 2.0}

EVAL_PROMPTS = [
    "a photo of a smiling young woman with long blonde hair",
    "a close-up portrait of an old man with a white beard",
    "a photo of a little boy with curly hair",
    "a portrait of a woman with short red hair and freckles",
    "a photo of a bald man wearing glasses",
    "a portrait of a laughing girl with dark skin",
    "a photo of a man with a black mustache and a serious face",
    "a portrait of an elderly woman with gray hair and earrings",
]


def gap_prompt_bank() -> list[str]:
    """Generic face descriptions used only to estimate the mean CLIP text embedding."""
    who = ["man", "woman", "boy", "girl", "old man", "old woman", "young man", "young woman", "child", "person"]
    what = ["", " with glasses", " smiling", " with long hair", " with short hair", " with a beard", " wearing a hat",
            " with curly hair"]
    style = ["a photo of a {}{}", "a portrait of a {}{}", "a close-up photo of the face of a {}{}"]
    return [s.format(w, a) for s, w, a in itertools.product(style, who, what)]


@dataclass
class Task4Config:
    run_dir: str = str(DRIVE_ROOT / "runs" / "task4")
    stage: str = "all"                       # data | base | sr | eval | all
    local_shards: str | None = None          # tests: use these .tar shards instead of downloading
    max_shards: int = 0                      # 0 = all 70 shards (~70k images)
    keep_hr: int = 15000
    low_res: int = 256
    hr_res: int = 1024
    holdout: int = 2000                      # images never used for training (eval conditions)
    clip_cond: str = "openai/clip-vit-large-patch14"
    clip_eval: str = "openai/clip-vit-base-patch32"
    cond_noise: float = 0.25                 # LAFITE perturbation strength xi
    # ---- base GAN ----
    G: GeneratorConfig = field(default_factory=lambda: GeneratorConfig(
        resolution=256, z_dim=128, w_dim=512, cond_dim=512, channels=dict(HR_CHANNELS), blocks=dict(HR_BLOCKS),
        expansion=dict(HR_EXPANSION), attn_res=(16,)))
    D: DiscriminatorConfig = field(default_factory=lambda: DiscriminatorConfig(
        resolution=256, channels=dict(HR_CHANNELS), blocks=dict(HR_BLOCKS), expansion=dict(HR_EXPANSION),
        attn_res=(16, 8)))
    dino: str = "facebook/dinov2-small"
    proj_layers: tuple = (3, 6, 9, 12)
    proj_width: int = 256
    proj_weight: float = 1.0
    batch: int = 64
    base_hours: float = 7.0
    lr: tuple = (2e-4, 5e-5)
    beta2: tuple = (0.9, 0.99)
    gamma_pix: tuple = (30.0, 3.0)
    gamma_proj: tuple = (1.0, 0.1)
    ema_kimg: tuple = (10.0, 300.0)
    aug_p: tuple = (0.2, 0.5)
    aug_policy: str = "color,translation,cutout"
    reg_every: int = 8
    reg_fp32: bool = False
    compile: bool = True
    amp: bool = True
    channels_last: bool = True
    # ---- super-resolution ----
    sr: SRConfig = field(default_factory=SRConfig)
    sr_D: DiscriminatorConfig = field(default_factory=lambda: DiscriminatorConfig(
        resolution=256, channels={256: 32, 128: 64, 64: 128, 32: 256, 16: 256, 8: 256, 4: 256}, blocks=1,
        expansion=2.0, attn_res=(16,)))
    sr_hours: float = 2.5
    sr_batch: int = 32
    sr_crop: int = 256
    sr_lr: tuple = (2e-4, 2e-5)
    sr_lpips: float = 1.0
    sr_adv: float = 0.05
    sr_adv_start: float = 0.1
    sr_gamma: tuple = (10.0, 1.0)
    # ---- evaluation / cadence ----
    eval_n: int = 10000
    final_n: int = 10000
    log_every: int = 100
    sample_minutes: float = 20.0
    eval_minutes: float = 60.0
    ckpt_minutes: float = 20.0
    metrics: str = "inception"
    seed: int = 0
    device: str | None = None
    smoke: bool = False
    force: bool = False
    max_steps: int = 0


def smoke_config(cfg: Task4Config) -> Task4Config:
    cfg.smoke = True
    small = {4: 32, 8: 32, 16: 24, 32: 16, 64: 16}
    cfg.low_res, cfg.hr_res, cfg.keep_hr, cfg.holdout = 64, 128, 8, 4
    cfg.G = GeneratorConfig(resolution=64, z_dim=8, w_dim=16, cond_dim=8, channels=small, blocks=1, expansion=2.0,
                            heads=2)
    cfg.D = DiscriminatorConfig(resolution=64, channels=small, blocks=1, expansion=2.0, heads=2)
    cfg.proj_layers = (1, 2)
    cfg.proj_width = 16
    cfg.sr = SRConfig(channels=16, blocks=1, hr_channels=8, hr_blocks=1)
    cfg.sr_D = DiscriminatorConfig(resolution=32, channels={32: 16, 16: 16, 8: 16, 4: 16}, blocks=1, expansion=2.0,
                                   attn_res=(8,), heads=2)
    cfg.sr_crop = 32
    cfg.batch, cfg.sr_batch, cfg.max_steps = 4, 4, 4
    cfg.log_every = 2
    cfg.sample_minutes = cfg.eval_minutes = cfg.ckpt_minutes = 0.0
    cfg.eval_n = cfg.final_n = 16
    cfg.metrics = "random"
    cfg.amp = False
    cfg.compile = False
    cfg.channels_last = False
    return cfg


# --------------------------------------------------------------------------------------
# conditioning helpers
# --------------------------------------------------------------------------------------

def perturb(e: torch.Tensor, xi: float) -> torch.Tensor:
    """LAFITE: h' = normalize(h + xi * eps * ||h|| / ||eps||)."""
    if xi <= 0:
        return F.normalize(e, dim=-1)
    eps = torch.randn_like(e)
    e = e + xi * eps * e.norm(dim=-1, keepdim=True) / eps.norm(dim=-1, keepdim=True)
    return F.normalize(e, dim=-1)


def text_conditions(prompts: list[str], gap: dict, clip_name: str, device) -> torch.Tensor:
    """CLIP text embeddings moved into the image-embedding region (modality-gap correction)."""
    from cocogoat.models.pretrained import CLIPEncoder

    enc = CLIPEncoder(clip_name, device)
    t = enc.encode_text(prompts)
    del enc
    mu_txt = torch.as_tensor(gap["mu_txt"], device=t.device, dtype=t.dtype)
    mu_img = torch.as_tensor(gap["mu_img"], device=t.device, dtype=t.dtype)
    return F.normalize(F.normalize(t, dim=-1) - mu_txt + mu_img, dim=-1)


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------

def prepare_data(cfg: Task4Config, rd: RunDir, device):
    cache = LOCAL_CACHE / "ffhq"
    if cfg.local_shards:
        shards = sorted(Path(cfg.local_shards).glob("*.tar"))
    else:
        t0 = time.time()
        shards = download_shards(cache, cfg.max_shards or None)
        rd.print(f"downloaded {len(shards)} shards in {(time.time() - t0) / 60:.1f} min")
    t0 = time.time()
    low, hr = decode_shards(shards, cache / "decoded", cfg.low_res, cfg.keep_hr, cfg.hr_res)
    rd.print(f"decoded: low {low.shape}, hr {None if hr is None else hr.shape} in {(time.time() - t0) / 60:.1f} min")
    emb_path = rd.path / f"clip_emb_{cfg.low_res}.npy"
    gap_path = rd.path / "clip_gap.npz"
    if emb_path.exists() and np.load(emb_path, mmap_mode="r").shape[0] == low.shape[0] and gap_path.exists():
        emb = np.load(emb_path)
        with np.load(gap_path) as z:
            gap = {"mu_img": z["mu_img"], "mu_txt": z["mu_txt"]}
        return low, hr, emb, gap
    if cfg.smoke:
        emb = np.random.default_rng(0).normal(size=(low.shape[0], 768)).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        gap = {"mu_img": emb.mean(0), "mu_txt": np.zeros(768, np.float32)}
    else:
        from cocogoat.models.pretrained import CLIPEncoder

        enc = CLIPEncoder(cfg.clip_cond, device)
        out = []
        for i in range(0, low.shape[0], 512):
            x = torch.from_numpy(np.ascontiguousarray(low[i:i + 512])).to(device).permute(0, 3, 1, 2).float() / 127.5 - 1
            with autocast(device, True):
                out.append(enc.encode_images(x).float().cpu())
        emb = torch.cat(out).numpy().astype(np.float32)
        mu_txt = enc.encode_text(gap_prompt_bank()).mean(0).cpu().numpy()
        gap = {"mu_img": emb.mean(0), "mu_txt": mu_txt}
        del enc
        torch.cuda.empty_cache()
    np.save(emb_path, emb.astype(np.float16))
    np.savez(gap_path, **gap)
    rd.print(f"CLIP embeddings {emb.shape}")
    return low, hr, emb.astype(np.float16), gap


# --------------------------------------------------------------------------------------
# stage: base GAN at 256
# --------------------------------------------------------------------------------------

def build_projected(cfg: Task4Config, device) -> ProjectedCritic:
    if cfg.smoke:
        backbone = TinyBackbone(dim=16, layers=len(cfg.proj_layers), grid=16).to(device)
        feat_dim, grid = 16, 16
    else:
        from cocogoat.models.pretrained import DINOv2Features

        backbone = DINOv2Features(cfg.dino, layers=tuple(cfg.proj_layers), device=device)
        feat_dim, grid = backbone.dim, backbone.size // backbone.patch
    return ProjectedCritic(backbone, feat_dim, len(cfg.proj_layers), grid, cfg.proj_width,
                           cond_vector_dim=768, policy=cfg.aug_policy, p=cfg.aug_p[0]).to(device)


def train_base(cfg: Task4Config, low: np.ndarray, emb: np.ndarray, device, run_dir: Path) -> Path:
    rd = RunDir(run_dir, config=cfg)
    out_path = run_dir / "G_ema_base.pt"
    with rd.capture_errors():
        if resume_or_skip(rd, cfg.base_hours, cfg.force) and out_path.exists():
            return out_path
        n_train = low.shape[0] - cfg.holdout
        bank = GPUImageBank(np.asarray(low[:n_train]), torch.from_numpy(np.asarray(emb[:n_train]).astype(np.float32)),
                            device=device, xflip=True)
        hold_emb = torch.from_numpy(np.asarray(emb[n_train:]).astype(np.float32)).to(device)
        cfg.G.cond_vector_dim = cfg.D.cond_vector_dim = 768
        cfg.G.resolution = cfg.D.resolution = cfg.low_res
        G = Generator(cfg.G).to(device)
        D = Discriminator(cfg.D).to(device)
        if cfg.channels_last and device.type == "cuda":
            G = G.to(memory_format=torch.channels_last)
            D = D.to(memory_format=torch.channels_last)
        G_ema = EMA(G)
        proj = build_projected(cfg, device)
        rd.print(f"base G {count_params(G) / 1e6:.1f}M, pixel D {count_params(D) / 1e6:.1f}M, projected heads "
                 f"{sum(p.numel() for p in proj.trainable_parameters()) / 1e6:.1f}M params; train images {n_train}")
        opt_G = make_adam(G.parameters(), cfg.lr[0], cfg.beta2[0])
        opt_D = make_adam(D.parameters(), cfg.lr[0], cfg.beta2[0])
        opt_P = make_adam(proj.trainable_parameters(), cfg.lr[0], cfg.beta2[0])
        pix = PixelCritic(D, cfg.aug_policy, cfg.aug_p[0])
        slot_pix = CriticSlot("pix", pix, opt_D, g_weight=1.0, gamma=cfg.gamma_pix[0])
        slot_proj = CriticSlot("proj", proj, opt_P, g_weight=cfg.proj_weight, gamma=cfg.gamma_proj[0])
        trainer = AdversarialTrainer([slot_pix, slot_proj], device, amp=cfg.amp, reg_every=cfg.reg_every,
                                     reg_fp32=cfg.reg_fp32)
        G_fast = maybe_compile(G, cfg.compile)
        pix.set_fast(maybe_compile(D, cfg.compile))
        compiled = G_fast is not None

        step, nimg, best_fid, elapsed = 0, 0, float("inf"), 0.0
        st = rd.load_checkpoint()
        if load_compatible(rd, {"G": G, "D": D, "G_ema": G_ema, "proj_heads": proj.heads, "opt_G": opt_G,
                                "opt_D": opt_D, "opt_P": opt_P}, st):
            step, nimg, best_fid, elapsed = st["step"], st["nimg"], st["best_fid"], st["elapsed"]
            set_rng_state(st["rng"])
            rd.print(f"resumed base GAN at step {step} ({elapsed / 3600:.2f} h)")

        gen = torch.Generator(device=device).manual_seed(7)
        fixed_z = torch.randn(32, cfg.G.z_dim, device=device, generator=gen)
        fixed_c = F.normalize(hold_emb[torch.arange(32, device=device) % len(hold_emb)], dim=-1)
        save_grid(bank.to_float(bank.images[:32]), rd.path / "samples" / "real.png", max_side=2048)

        feature_fn = make_feature_fn(cfg.metrics, device)
        evaluator = {}

        def get_ev():
            if "ev" not in evaluator:
                evaluator["ev"] = ImageEvaluator(feature_fn, (x for x, _ in bank.iterate(250)),
                                                 cache=rd.path / f"stats_{cfg.low_res}_{cfg.metrics}.npz")
            return evaluator["ev"]

        @torch.no_grad()
        def sample_fn(n):
            c = F.normalize(hold_emb[torch.randint(len(hold_emb), (n,), device=device)], dim=-1)
            with autocast(device, cfg.amp):
                return G_ema.model(torch.randn(n, cfg.G.z_dim, device=device), c).float()

        def checkpoint():
            rd.save_checkpoint({"G": G.state_dict(), "D": D.state_dict(), "G_ema": G_ema.state_dict(),
                                "proj_heads": proj.heads.state_dict(), "opt_G": opt_G.state_dict(),
                                "opt_D": opt_D.state_dict(), "opt_P": opt_P.state_dict(), "step": step, "nimg": nimg,
                                "best_fid": best_fid, "elapsed": clock.now(), "rng": rng_state()})

        def evaluate(tag):
            nonlocal best_fid
            t0 = time.time()
            m = get_ev().evaluate(sample_fn, n=cfg.eval_n, batch=250)
            m.update(step=step, kimg=nimg / 1000, hours=clock.now() / 3600, eval_seconds=time.time() - t0, tag=tag)
            rd.log_metrics(kind="eval", **m)
            rd.print(f"[base eval {tag}] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                                      for k, v in m.items()))
            if m["fid"] < best_fid:
                best_fid = m["fid"]
                torch.save({"G_ema": G_ema.state_dict(), "G_config": dataclasses.asdict(cfg.G), "metrics": m},
                           rd.path / "ckpt" / "best_G_ema.pt")
            return m

        def train_step(g_net):
            x_real, c = bank.sample(cfg.batch)
            c = perturb(c, cfg.cond_noise)
            z = torch.randn(cfg.batch, cfg.G.z_dim, device=device)
            with torch.no_grad(), autocast(device, cfg.amp):
                x_fake = g_net(z, c)
            d_stats = trainer.d_step({"img": x_real}, {"img": x_fake}, c, step)
            x_real, c = bank.sample(cfg.batch)
            c = perturb(c, cfg.cond_noise)
            z = torch.randn(cfg.batch, cfg.G.z_dim, device=device)
            with autocast(device, cfg.amp):
                x_fake = g_net(z, c)
            g_loss, g_stats = trainer.g_adversarial_loss({"img": x_real}, {"img": x_fake}, c)
            opt_G.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_G.step()
            return d_stats, g_stats

        budget = cfg.base_hours * 3600
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
            slot_pix.gamma = log_cosine_interp(*cfg.gamma_pix, progress)
            slot_proj.gamma = log_cosine_interp(*cfg.gamma_proj, progress)
            pix.p = proj.p = cosine_interp(*cfg.aug_p, progress)
            for opt in (opt_G, opt_D, opt_P):
                set_optimizer_hparams(opt, lr=lr, beta2=beta2)
            try:
                d_stats, g_stats = train_step(G_fast if G_fast is not None else G)
            except Exception as exc:  # noqa: BLE001
                if compiled and step < 5:
                    rd.print(f"torch.compile failed ({type(exc).__name__}: {str(exc)[:300]}); using eager mode")
                    torch._dynamo.reset()
                    G_fast, compiled = None, False
                    pix.set_fast(None)
                    continue
                raise
            nimg += cfg.batch
            step += 1
            G_ema.update(G, ema_beta(cfg.batch, cosine_interp(*cfg.ema_kimg, progress), nimg))
            n_log += 1
            if step % cfg.log_every == 0:
                dt = time.time() - t_log
                stats = {**to_floats(d_stats), **to_floats(g_stats), "lr": lr, "gamma_pix": slot_pix.gamma,
                         "gamma_proj": slot_proj.gamma, "aug_p": pix.p, "it_per_s": n_log / max(dt, 1e-9),
                         "img_per_s": n_log * cfg.batch / max(dt, 1e-9)}
                bad = bad + 1 if not all(math.isfinite(v) for v in stats.values()) else 0
                if bad >= 3:
                    raise FloatingPointError("base GAN diverged (non-finite losses); resume from last checkpoint")
                rd.log_metrics(kind="train", step=step, kimg=nimg / 1000, hours=clock.now() / 3600, **stats)
                rd.write_status(stage="base", step=step, kimg=nimg / 1000, hours=clock.now() / 3600,
                                budget_hours=cfg.base_hours, best_fid=best_fid, **stats)
                if step % (cfg.log_every * 10) == 0 or cfg.max_steps:
                    rd.print(f"base step {step} kimg {nimg / 1000:.1f} " +
                             " ".join(f"{k}={v:.4g}" for k, v in stats.items()))
                t_log, n_log = time.time(), 0
            if timers[0]() or (cfg.max_steps and step == cfg.max_steps):
                clock.pause()
                with torch.no_grad(), autocast(device, cfg.amp):
                    grid = G_ema.model(fixed_z, fixed_c).float()
                save_grid(grid, rd.path / "samples" / f"step{step:07d}.png", max_side=2048)
                save_grid(grid, rd.path / "samples" / "latest.png", max_side=2048)
                clock.start()
            if timers[1]():
                clock.pause()
                evaluate("periodic")
                clock.start()
            if timers[2]():
                clock.pause()
                checkpoint()
                clock.start()
        clock.pause()
        checkpoint()
        final = evaluate("final")
        torch.save({"G_ema": G_ema.state_dict(), "G_config": dataclasses.asdict(cfg.G), "metrics": final}, out_path)
        rd.mark_done({"final_256": final, "budget_hours": cfg.base_hours, "steps": step, "kimg": nimg / 1000})
        del bank, G, D, G_ema, proj, trainer
        gc.collect()
        torch.cuda.empty_cache()
        return out_path


# --------------------------------------------------------------------------------------
# stage: x4 super-resolution GAN
# --------------------------------------------------------------------------------------

def train_sr(cfg: Task4Config, hr: np.ndarray, device, run_dir: Path) -> Path:
    from cocogoat.losses.perceptual import PerceptualLoss

    rd = RunDir(run_dir, config=cfg)
    out_path = run_dir / "sr.pt"
    with rd.capture_errors():
        if resume_or_skip(rd, cfg.sr_hours, cfg.force) and out_path.exists():
            return out_path
        n_val = max(1, min(500, hr.shape[0] // 10))
        hr_bank = to_device_nchw(hr[:-n_val], device, chunk=256)
        val_bank = to_device_nchw(hr[-n_val:], device, chunk=256)
        rd.print(f"SR: {hr_bank.shape[0]} training / {n_val} validation HR images on GPU")
        scale = cfg.hr_res // cfg.low_res
        cfg.sr.scale = scale
        net = SRNet(cfg.sr).to(device)
        cfg.sr_D.resolution = cfg.sr_crop
        D = Discriminator(cfg.sr_D).to(device)
        lpips = PerceptualLoss(device, random_weights=cfg.metrics == "random")
        opt = torch.optim.Adam(net.parameters(), lr=cfg.sr_lr[0], betas=(0.9, 0.99))
        opt_D = make_adam(D.parameters(), cfg.sr_lr[0], 0.99)
        critic = PixelCritic(D, "", 0.0)
        slot = CriticSlot("srD", critic, opt_D, gamma=cfg.sr_gamma[0])
        trainer = AdversarialTrainer([slot], device, amp=cfg.amp, reg_every=cfg.reg_every, reg_fp32=cfg.reg_fp32)
        rd.print(f"SR net {count_params(net) / 1e6:.2f}M, SR D {count_params(D) / 1e6:.2f}M params")
        step, elapsed = 0, 0.0
        st = rd.load_checkpoint()
        if load_compatible(rd, {"net": net, "D": D, "opt": opt, "opt_D": opt_D}, st):
            step, elapsed = st["step"], st["elapsed"]
            set_rng_state(st["rng"])
            rd.print(f"resumed SR at step {step}")

        def crops(bank: torch.Tensor, b: int, size: int) -> torch.Tensor:
            idx = torch.randint(bank.shape[0], (b,))
            ys = torch.randint(0, bank.shape[2] - size + 1, (b,))
            xs = torch.randint(0, bank.shape[3] - size + 1, (b,))
            out = torch.stack([bank[int(i), :, int(y):int(y) + size, int(x):int(x) + size]
                               for i, y, x in zip(idx.tolist(), ys.tolist(), xs.tolist())])
            return out.float() / 127.5 - 1

        torch.manual_seed(123)
        val_hr = crops(val_bank, min(64, 8 * n_val), cfg.sr_crop)
        val_lr = degrade(val_hr, scale, blur_prob=0.0, noise_prob=0.0)

        def checkpoint():
            rd.save_checkpoint({"net": net.state_dict(), "D": D.state_dict(), "opt": opt.state_dict(),
                                "opt_D": opt_D.state_dict(), "step": step, "elapsed": clock.now(), "rng": rng_state()})

        @torch.no_grad()
        def validate(tag):
            with autocast(device, cfg.amp):
                sr = net(val_lr).float().clamp(-1, 1)
            bic = F.interpolate(val_lr, scale_factor=scale, mode="bicubic", align_corners=False).clamp(-1, 1)
            m = {"psnr_sr": float(psnr(sr, val_hr).mean()), "psnr_bicubic": float(psnr(bic, val_hr).mean()),
                 "lpips_sr": float(lpips(sr, val_hr)), "lpips_bicubic": float(lpips(bic, val_hr))}
            rd.log_metrics(kind="sr_val", step=step, tag=tag, **m)
            rd.print(f"[sr val {tag}] " + " ".join(f"{k}={v:.4g}" for k, v in m.items()))
            save_grid(torch.cat([bic[:4], sr[:4], val_hr[:4]]), rd.path / "samples" / f"sr_{tag}.png", nrow=4)
            return m

        budget = cfg.sr_hours * 3600
        clock = TrainClock(elapsed)
        timers = (Every(cfg.eval_minutes * 60 / 2), Every(cfg.ckpt_minutes * 60))
        t_log = time.time()
        clock.start()
        while True:
            progress = clock.now() / budget
            if (cfg.max_steps and step >= cfg.max_steps) or (not cfg.max_steps and progress >= 1):
                break
            lr = cosine_interp(*cfg.sr_lr, progress)
            set_optimizer_hparams(opt, lr=lr)
            set_optimizer_hparams(opt_D, lr=lr)
            slot.gamma = log_cosine_interp(*cfg.sr_gamma, progress)
            hr_c = crops(hr_bank, cfg.sr_batch, cfg.sr_crop)
            flip = torch.rand(cfg.sr_batch, device=device) < 0.5
            hr_c = torch.where(flip[:, None, None, None], hr_c.flip(3), hr_c)
            lr_c = degrade(hr_c, scale)
            with autocast(device, cfg.amp):
                sr = net(lr_c).float()
            l1 = (sr - hr_c).abs().mean()
            lp = lpips(sr, hr_c)
            loss = l1 + cfg.sr_lpips * lp
            stats = {"l1": l1.detach(), "lpips": lp.detach(), "lr": lr}
            adv_on = progress >= cfg.sr_adv_start
            if adv_on:
                adv, g_stats = trainer.g_adversarial_loss({"img": hr_c}, {"img": sr}, None)
                loss = loss + cfg.sr_adv * adv
                stats.update(g_stats)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            if adv_on:
                stats.update(trainer.d_step({"img": hr_c}, {"img": sr.detach()}, None, step))
            step += 1
            if step % cfg.log_every == 0:
                fl = to_floats(stats)
                fl["it_per_s"] = cfg.log_every / max(time.time() - t_log, 1e-9)
                t_log = time.time()
                rd.log_metrics(kind="sr_train", step=step, hours=clock.now() / 3600, **fl)
                rd.write_status(stage="sr", step=step, hours=clock.now() / 3600, budget_hours=cfg.sr_hours, **fl)
                if step % (cfg.log_every * 10) == 0 or cfg.max_steps:
                    rd.print(f"sr step {step} " + " ".join(f"{k}={v:.4g}" for k, v in fl.items()))
            if timers[0]():
                clock.pause()
                validate(f"step{step:06d}")
                clock.start()
            if timers[1]():
                clock.pause()
                checkpoint()
                clock.start()
        clock.pause()
        checkpoint()
        final = validate("final")
        torch.save({"net": net.state_dict(), "sr_config": dataclasses.asdict(cfg.sr), "val": final}, out_path)
        rd.mark_done({"val": final, "budget_hours": cfg.sr_hours, "steps": step})
        del hr_bank, val_bank, net, D, trainer
        gc.collect()
        torch.cuda.empty_cache()
        return out_path


# --------------------------------------------------------------------------------------
# stage: 1024 evaluation
# --------------------------------------------------------------------------------------

@torch.no_grad()
def final_eval(cfg: Task4Config, low, hr, emb, gap, base_path: Path, sr_path: Path, device, run_dir: Path) -> dict:
    rd = RunDir(run_dir, config=cfg)
    with rd.capture_errors():
        gst = torch.load(base_path, map_location="cpu", weights_only=False)
        G = Generator(GeneratorConfig(**gst["G_config"])).to(device).eval()
        G.load_state_dict(gst["G_ema"])
        sst = torch.load(sr_path, map_location="cpu", weights_only=False)
        net = SRNet(SRConfig(**sst["sr_config"])).to(device).eval()
        net.load_state_dict(sst["net"])
        n_train = low.shape[0] - cfg.holdout
        hold_emb = F.normalize(torch.from_numpy(np.asarray(emb[n_train:]).astype(np.float32)).to(device), dim=-1)
        feat = make_feature_fn(cfg.metrics, device)
        crop = 299 if cfg.hr_res >= 299 else cfg.hr_res // 2

        def generate(c: torch.Tensor, z: torch.Tensor | None = None):
            z = torch.randn(len(c), G.cfg.z_dim, device=device) if z is None else z
            with autocast(device, cfg.amp):
                base = G(z, c).float().clamp(-1, 1)
                big = net(base).float().clamp(-1, 1)
            return base, big

        def rand_crops(x: torch.Tensor, k: int = 1) -> torch.Tensor:
            outs = []
            for _ in range(k):
                y = int(torch.randint(0, x.shape[2] - crop + 1, ()))
                xx = int(torch.randint(0, x.shape[3] - crop + 1, ()))
                outs.append(x[:, :, y:y + crop, xx:xx + crop])
            return torch.cat(outs)

        # real statistics at hr resolution (full image -> 299 antialiased, and native patches)
        real_full, real_patch = [], []
        torch.manual_seed(0)
        n_real = min(hr.shape[0], cfg.final_n)
        for i in range(0, n_real, 32):
            x = torch.from_numpy(np.ascontiguousarray(hr[i:i + 32])).to(device).permute(0, 3, 1, 2).float() / 127.5 - 1
            real_full.append(feat(x).cpu().numpy())
            real_patch.append(feat(rand_crops(x)).cpu().numpy())
        real_full, real_patch = np.concatenate(real_full), np.concatenate(real_patch)
        fake_full, fake_patch, fake_base = [], [], []
        t0 = time.time()
        for i in range(0, cfg.final_n, 32):
            b = min(32, cfg.final_n - i)
            c = hold_emb[torch.randint(len(hold_emb), (b,), device=device)]
            base, big = generate(c)
            fake_full.append(feat(big).cpu().numpy())
            fake_patch.append(feat(rand_crops(big)).cpu().numpy())
            fake_base.append(feat(base).cpu().numpy())
        gen_seconds = time.time() - t0
        fake_full, fake_patch, fake_base = map(np.concatenate, (fake_full, fake_patch, fake_base))
        sub = min(1000, len(real_full), len(fake_full))
        m = {}
        mu_r, s_r = gaussian_stats(real_full)
        m["fid_1024"] = frechet_distance(mu_r, s_r, *gaussian_stats(fake_full))
        m["kid_1024_x1000"], m["kid_1024_std_x1000"] = [v * 1000 for v in kid(real_full, fake_full, subset_size=sub)]
        m["fid_base256_vs_hr"] = frechet_distance(mu_r, s_r, *gaussian_stats(fake_base))
        mu_p, s_p = gaussian_stats(real_patch)
        m["patch_fid"] = frechet_distance(mu_p, s_p, *gaussian_stats(fake_patch))
        m["patch_kid_x1000"], m["patch_kid_std_x1000"] = [v * 1000 for v in kid(real_patch, fake_patch, subset_size=sub)]
        p, r = precision_recall(real_full[:sub * 5], fake_full[:sub * 5])
        m["precision"], m["recall"] = p, r
        m["n_fake"], m["n_real"] = int(len(fake_full)), int(len(real_full))
        m["sec_per_1024_image"] = gen_seconds / max(1, cfg.final_n)

        # prompts: CLIP score with a *different* CLIP (B/32) than the conditioning one (L/14)
        prompts = EVAL_PROMPTS
        if not cfg.smoke:
            from cocogoat.models.pretrained import CLIPEncoder

            c_txt = text_conditions(prompts, gap, cfg.clip_cond, device)
            judge = CLIPEncoder(cfg.clip_eval, device)
            t_eval = judge.encode_text(prompts)
        else:
            c_txt = F.normalize(torch.randn(len(prompts), 768, device=device), dim=-1)
            judge, t_eval = None, None
        per = 8
        grid_base, scores_match, scores_other = [], [], []
        (rd.path / "samples" / "full_res").mkdir(parents=True, exist_ok=True)
        for j, prompt in enumerate(prompts):
            base, big = generate(c_txt[j:j + 1].expand(per, -1))
            grid_base.append(F.interpolate(big, size=(256, 256), mode="bicubic", antialias=True))
            save_images(big[:2], rd.path / "samples" / "full_res", prefix=f"prompt{j}")
            if judge is not None:
                e = judge.encode_images(big)
                sims = e @ t_eval.T
                scores_match.append(float(sims[:, j].mean()))
                scores_other.append(float(torch.cat([sims[:, :j], sims[:, j + 1:]], 1).mean()))
        save_grid(torch.cat(grid_base), rd.path / "samples" / "prompts_grid.png", nrow=per)
        if scores_match:
            m["clip_score_matched"] = float(np.mean(scores_match))
            m["clip_score_mismatched"] = float(np.mean(scores_other))
        # unconditional-style sample sheet at 1024 (held-out conditions)
        c = hold_emb[:16]
        _, big = generate(c)
        save_grid(big, rd.path / "samples" / "final_1024_grid.png", nrow=4, max_side=4096)
        save_images(big[:4], rd.path / "samples" / "full_res", prefix="sample")
        rd.write_json("final_metrics.json", m)
        (rd.path / "report.md").write_text(summarize_metrics_md(
            "Task 4 - 1024x1024 conditional GAN (base 256 GAN + x4 SR GAN)", m,
            ["fid/kid_1024: full images, antialiased to 299 px for Inception; patch_*: native-resolution 299 px "
             "crops (high-frequency realism); fid_base256_vs_hr: the base GAN alone (256 px) vs the same real set.",
             f"Prompts ({len(prompts)}): " + "; ".join(prompts),
             "clip_score_matched vs _mismatched: CLIP B/32 similarity of images to their own prompt vs to the "
             "other prompts (a gap > 0 means the prompt conditioning works).",
             "Samples: samples/prompts_grid.png (rows = prompts), samples/final_1024_grid.png, samples/full_res/"]))
        rd.mark_done(m)
        rd.print("final 1024 evaluation: " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                                     for k, v in m.items()))
        return m


def run(cfg: Task4Config) -> dict:
    device = setup_torch(cfg.device)
    seed_everything(cfg.seed)
    root = Path(cfg.run_dir)
    for sub in ("", "base", "sr", "eval"):
        (root / sub).mkdir(parents=True, exist_ok=True)
        rotate_old_error(root / sub)
    rd = RunDir(root, config=cfg)
    with rd.capture_errors():
        rd.print(f"env: {describe_env()}")
        if cfg.smoke and not cfg.local_shards:
            cfg.local_shards = str(make_fake_shards(LOCAL_CACHE / "fake_ffhq", 2, 6, cfg.hr_res)[0].parent)
        low, hr, emb, gap = prepare_data(cfg, rd, device)
        if cfg.stage == "data":
            return {}
        base_path = root / "base" / "G_ema_base.pt"
        sr_path = root / "sr" / "sr.pt"
        if cfg.stage in ("base", "all"):
            base_path = train_base(cfg, low, emb, device, root / "base")
        if cfg.stage in ("sr", "all"):
            sr_path = train_sr(cfg, hr, device, root / "sr")
        if cfg.stage in ("eval", "all"):
            return final_eval(cfg, low, hr, emb, gap, base_path, sr_path, device, root / "eval")
        return {}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default=None)
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--base_hours", type=float, default=None)
    ap.add_argument("--sr_hours", type=float, default=None)
    ap.add_argument("--max_shards", type=int, default=None)
    ap.add_argument("--local_shards", default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--override", default=None)
    args = ap.parse_args(argv)
    cfg = Task4Config()
    if args.smoke:
        cfg = smoke_config(cfg)
    for k in ["stage", "run_dir", "base_hours", "sr_hours", "max_shards", "local_shards", "max_steps", "device"]:
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)
    cfg.force = args.force
    apply_overrides(cfg, parse_override(args.override))
    return run(cfg)


if __name__ == "__main__":
    main()
