"""Task 3 - class-conditional latent GAN with the data flow  z -> G -> z' -> Decoder -> fake -> D.

Stage ``vae``: beta-VAE on the Task 1 image dataset (same trainer as Task 2), then frozen.
Stage ``gan``: latent generator + two critics, all with RpGAN + R1 + R2:
  * image critic on decoded images: fakes Dec(z') vs *decoded reals* Dec(z_real).  Comparing
    decoded with decoded means the critic cannot win by spotting VAE reconstruction blur,
    which the generator could never fix - a likely cause of the old latent-GAN collapses.
  * latent critic on z' vs posterior samples of real images: keeps z' on the decoder's training
    manifold (the old failure mode: off-manifold latents that decode to look-alike images).
  * optional OT penalty lambda * ||z' - z||^2 on the generator.

Metrics: FID/KID/precision/recall of Dec(G(z)) against the *real* images (directly comparable with
Task 1), plus against decoded reals, plus the VAE's rFID (the floor any latent model inherits).

Usage (Colab):
    python -m cocogoat.tasks.task3_latent_gan --dataset genshin
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from cocogoat.data import diff_augment
from cocogoat.data.video import ClipBank
from cocogoat.metrics import ImageEvaluator
from cocogoat.models.gan2d import Discriminator, DiscriminatorConfig
from cocogoat.models.latent_gan import LatentGenerator, LatentGeneratorConfig
from cocogoat.models.vae import VAEConfig, load_vae
from cocogoat.nn import count_params
from cocogoat.tasks.common import (
    DRIVE_ROOT,
    USER_DATA,
    apply_overrides,
    make_feature_fn,
    parse_override,
    rotate_old_error,
    sample_labels,
    summarize_metrics_md,
)
from cocogoat.tasks.task1_gan64 import PixelCritic, Task1Config, load_data
from cocogoat.tasks.task2_world import Task2Config, train_vae
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
from cocogoat.train.adversarial import AdversarialTrainer, Critic, CriticSlot, make_adam, maybe_compile, to_floats


def _vae_stage_defaults() -> Task2Config:
    c = Task2Config()
    c.vae_hours = 1.0
    c.vae = VAEConfig()
    return c


@dataclass
class Task3Config:
    run_dir: str = str(DRIVE_ROOT / "runs" / "task3")
    dataset: str = "genshin"
    data_path: str = str(USER_DATA / "Complete_dataset.npy")
    labels: str = "auto"
    num_clusters: int = 100
    resolution: int = 64
    holdout: int = 2000
    stage: str = "all"                       # vae | gan | all
    vae_stage: Task2Config = field(default_factory=_vae_stage_defaults)
    G: LatentGeneratorConfig = field(default_factory=LatentGeneratorConfig)
    D_img: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    D_lat: DiscriminatorConfig = field(default_factory=lambda: DiscriminatorConfig(
        resolution=8, in_channels=16, channels={8: 512, 4: 512}, blocks=3, attn_res=(8,), heads=8))
    batch: int = 256
    hours: float = 3.0
    max_steps: int = 0
    lr: tuple = (2e-4, 5e-5)
    beta2: tuple = (0.9, 0.99)
    gamma_img: tuple = (2.0, 0.2)
    gamma_lat: tuple = (1.0, 0.1)
    lat_weight: float = 1.0                  # generator weight of the latent critic
    ot_weight: tuple = (0.1, 0.01)           # OT penalty schedule (0 disables)
    ema_kimg: tuple = (10.0, 300.0)
    aug_p: tuple = (0.2, 0.6)
    aug_policy: str = "color,translation,cutout"
    lat_aug_policy: str = "translation"
    reg_every: int = 4
    reg_fp32: bool = False
    amp: bool = True
    compile: bool = True
    log_every: int = 100
    sample_minutes: float = 15.0
    eval_minutes: float = 45.0
    ckpt_minutes: float = 20.0
    eval_n: int = 10000
    final_eval_n: int = 50000
    metrics: str = "inception"
    seed: int = 0
    device: str | None = None
    smoke: bool = False
    force: bool = False


def smoke_config(cfg: Task3Config) -> Task3Config:
    from cocogoat.tasks import task2_world

    cfg.smoke = True
    cfg.vae_stage = task2_world.smoke_config(Task2Config())
    cfg.vae_stage.max_steps = 4
    cfg.G = LatentGeneratorConfig(z_global_dim=8, w_dim=16, cond_dim=8, width=32, depth=2, heads=2, expansion=2.0)
    cfg.D_img = DiscriminatorConfig(channels={64: 16, 32: 16, 16: 24, 8: 32, 4: 32}, blocks=1, expansion=2.0, heads=2)
    cfg.D_lat = DiscriminatorConfig(resolution=8, in_channels=16, channels={8: 32, 4: 32}, blocks=1, attn_res=(8,),
                                    heads=2, expansion=2.0)
    cfg.batch, cfg.max_steps, cfg.holdout = 8, 6, 16
    cfg.log_every = 2
    cfg.sample_minutes = cfg.eval_minutes = cfg.ckpt_minutes = 0.0
    cfg.eval_n = cfg.final_eval_n = 64
    cfg.metrics = "random"
    cfg.amp = False
    cfg.compile = False
    return cfg


class LatentCritic(Critic):
    keys = ("lat",)

    def __init__(self, D: Discriminator, policy: str, p: float):
        super().__init__()
        self.D = D
        self.policy = policy
        self.p = p
        object.__setattr__(self, "D_fast", None)

    def set_fast(self, module) -> None:
        object.__setattr__(self, "D_fast", module)

    def encode(self, sample):
        return {"lat": diff_augment(sample["lat"], self.policy, self.p)}

    def head(self, enc, cond):
        net = self.D_fast if (self.fast and self.D_fast is not None) else self.D
        return net(enc["lat"], cond)


def run(cfg: Task3Config) -> dict:
    device = setup_torch(cfg.device)
    seed_everything(cfg.seed)
    root = Path(cfg.run_dir)
    for sub in ("", "vae", "gan"):
        (root / sub).mkdir(parents=True, exist_ok=True)
        rotate_old_error(root / sub)
    rd_root = RunDir(root, config=cfg)
    with rd_root.capture_errors():
        rd_root.print(f"env: {describe_env()}")
        # ---- data (same images / labels as Task 1) ----
        t1 = Task1Config(dataset=cfg.dataset, data_path=cfg.data_path, labels=cfg.labels,
                         num_clusters=cfg.num_clusters, resolution=cfg.resolution, metrics=cfg.metrics)
        images, labels, num_classes, label_source = load_data(t1, rd_root, device)
        n = len(images)
        perm = np.random.default_rng(0).permutation(n)
        hold = perm[:cfg.holdout]
        train_idx = perm[cfg.holdout:]
        train_bank = ClipBank(images[train_idx], np.arange(len(train_idx)), device)
        test_bank = ClipBank(images[hold], np.arange(len(hold)), device)
        rd_root.print(f"data {images.shape}, classes={num_classes} ({label_source}); VAE holdout {len(hold)}")

        # ---- stage A: VAE ----
        vcfg = cfg.vae_stage
        vcfg.resolution, vcfg.metrics, vcfg.amp, vcfg.force = cfg.resolution, cfg.metrics, cfg.amp, cfg.force
        vae_path = root / "vae" / "vae.pt"
        if cfg.stage in ("vae", "all"):
            vae_path = train_vae(vcfg, train_bank, test_bank, device, root / "vae")
        if cfg.stage == "vae":
            return {}
        out = train_latent_gan(cfg, vae_path, images, labels, num_classes, label_source, device, root / "gan")
        rd_root.print("task 3 finished")
        return out


def train_latent_gan(cfg: Task3Config, vae_path: Path, images: np.ndarray, labels: np.ndarray | None,
                     num_classes: int, label_source: str, device: torch.device, run_dir: Path) -> dict:
    rd = RunDir(run_dir, config=cfg)
    with rd.capture_errors():
        if rd.done and not cfg.force:
            rd.print("latent GAN stage already DONE")
            return {}
        vae = load_vae(str(vae_path), device).requires_grad_(False).eval()
        lc, ls = vae.cfg.latent_channels, cfg.resolution // vae.cfg.downsample

        # ---- encode the whole dataset once: posterior mean + logvar (normalised space) ----
        imgs_t = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2)  # uint8 on CPU
        means, logstds = [], []
        with torch.no_grad():
            for i in range(0, len(imgs_t), 1024):
                x = imgs_t[i:i + 1024].to(device).float() / 127.5 - 1
                with autocast(device, cfg.amp):
                    m, lv = vae.encode(x)
                means.append(vae.normalize(m.float()).half())
                logstds.append((0.5 * lv.float() - vae.latent_std.log().view(1, -1, 1, 1)).half())
        lat_mean = torch.cat(means)
        lat_logstd = torch.cat(logstds)
        y_all = torch.from_numpy(labels).to(device) if labels is not None else None
        rd.print(f"encoded {len(lat_mean)} images -> latents {tuple(lat_mean.shape[1:])}; "
                 f"mean posterior std (normalised units) {lat_logstd.float().exp().mean():.3f}")
        img_bank = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).contiguous().to(device)

        def sample_real(b: int):
            idx = torch.randint(len(lat_mean), (b,), device=device)
            z = lat_mean[idx].float() + torch.randn(b, lc, ls, ls, device=device) * lat_logstd[idx].float().exp()
            flip = torch.rand(b, device=device) < 0.5   # x-flip in latent space (decoder is ~flip-equivariant)
            z = torch.where(flip[:, None, None, None], z.flip(3), z)
            return z, (y_all[idx] if y_all is not None else None)

        cfg.G.latent_channels, cfg.G.latent_size, cfg.G.num_classes = lc, ls, num_classes
        cfg.D_img.resolution, cfg.D_img.num_classes = cfg.resolution, num_classes
        cfg.D_lat.resolution, cfg.D_lat.in_channels, cfg.D_lat.num_classes = ls, lc, num_classes
        G = LatentGenerator(cfg.G).to(device)
        D_img = Discriminator(cfg.D_img).to(device)
        D_lat = Discriminator(cfg.D_lat).to(device)
        G_ema = EMA(G)
        rd.print(f"G {count_params(G) / 1e6:.2f}M, D_img {count_params(D_img) / 1e6:.2f}M, "
                 f"D_lat {count_params(D_lat) / 1e6:.2f}M params")
        opt_G = make_adam(G.parameters(), cfg.lr[0], cfg.beta2[0])
        opt_Di = make_adam(D_img.parameters(), cfg.lr[0], cfg.beta2[0])
        opt_Dl = make_adam(D_lat.parameters(), cfg.lr[0], cfg.beta2[0])
        crit_img = PixelCritic(D_img, cfg.aug_policy, cfg.aug_p[0])
        crit_lat = LatentCritic(D_lat, cfg.lat_aug_policy, cfg.aug_p[0])
        slot_img = CriticSlot("img", crit_img, opt_Di, g_weight=1.0, gamma=cfg.gamma_img[0])
        slot_lat = CriticSlot("lat", crit_lat, opt_Dl, g_weight=cfg.lat_weight, gamma=cfg.gamma_lat[0])
        trainer = AdversarialTrainer([slot_img, slot_lat], device, amp=cfg.amp, reg_every=cfg.reg_every,
                                     reg_fp32=cfg.reg_fp32)
        G_fast = maybe_compile(G, cfg.compile)
        crit_img.set_fast(maybe_compile(D_img, cfg.compile))
        crit_lat.set_fast(maybe_compile(D_lat, cfg.compile))
        compiled = G_fast is not None

        step, nimg, best_fid, elapsed = 0, 0, float("inf"), 0.0
        st = rd.load_checkpoint()
        if st is not None:
            G.load_state_dict(st["G"])
            D_img.load_state_dict(st["D_img"])
            D_lat.load_state_dict(st["D_lat"])
            G_ema.load_state_dict(st["G_ema"])
            opt_G.load_state_dict(st["opt_G"])
            opt_Di.load_state_dict(st["opt_Di"])
            opt_Dl.load_state_dict(st["opt_Dl"])
            step, nimg, best_fid, elapsed = st["step"], st["nimg"], st["best_fid"], st["elapsed"]
            set_rng_state(st["rng"])
            rd.print(f"resumed latent GAN at step {step}")

        prior = None
        if y_all is not None:
            prior = torch.bincount(y_all, minlength=num_classes).float()
            prior = prior / prior.sum()
        gen = torch.Generator(device=device).manual_seed(1234)
        fixed_z = torch.randn(64, lc, ls, ls, device=device, generator=gen)
        fixed_zg = torch.randn(64, cfg.G.z_global_dim, device=device, generator=gen)
        if num_classes:
            top = torch.argsort(torch.bincount(y_all, minlength=num_classes), descending=True)[:8]
            fixed_c = top[torch.arange(8, device=device) % len(top)].repeat_interleave(8)
        else:
            fixed_c = None

        def decode(z):
            return vae.decode_normalized(z)

        def generate(model, b, c=None, z=None, zg=None):
            z = torch.randn(b, lc, ls, ls, device=device) if z is None else z
            zg = torch.randn(b, cfg.G.z_global_dim, device=device) if zg is None else zg
            return model(z, zg, c), z

        feature_fn = make_feature_fn(cfg.metrics, device)
        evaluator = {}

        def get_evaluator(kind: str):
            if kind not in evaluator:
                if kind == "real":
                    batches = (img_bank[i:i + 500].float() / 127.5 - 1 for i in range(0, len(img_bank), 500))
                else:  # decoded reals (posterior means)
                    def _dec():
                        for i in range(0, len(lat_mean), 500):
                            with torch.no_grad(), autocast(device, cfg.amp):
                                yield decode(lat_mean[i:i + 500].float()).float().clamp(-1, 1)
                    batches = _dec()
                evaluator[kind] = ImageEvaluator(feature_fn, batches, cache=rd.path / f"stats_{kind}_{cfg.metrics}.npz")
            return evaluator[kind]

        @torch.no_grad()
        def sample_fn(b: int) -> torch.Tensor:
            c = sample_labels(prior, b, device)
            with autocast(device, cfg.amp):
                z_fake, _ = generate(G_ema.model, b, c)
                return decode(z_fake).float().clamp(-1, 1)

        def evaluate(n: int, tag: str) -> dict:
            nonlocal best_fid
            t0 = time.time()
            ev = get_evaluator("real")
            feats = ev.features_of(sample_fn, n)
            m = ev.score_features(feats)
            m_dec = get_evaluator("decoded").score_features(feats)
            m.update({f"vs_decoded_{k}": v for k, v in m_dec.items() if k in ("fid", "kid_x1000", "precision", "recall")})
            m.update(step=step, kimg=nimg / 1000, hours=clock.now() / 3600, eval_seconds=time.time() - t0, tag=tag)
            rd.log_metrics(kind="eval", **m)
            rd.print(f"[eval {tag}] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                                 for k, v in m.items()))
            if m["fid"] < best_fid:
                best_fid = m["fid"]
                torch.save({"G_ema": G_ema.state_dict(), "config": dataclasses.asdict(cfg), "metrics": m},
                           rd.path / "ckpt" / "best_G_ema.pt")
            return m

        def checkpoint():
            rd.save_checkpoint({"G": G.state_dict(), "D_img": D_img.state_dict(), "D_lat": D_lat.state_dict(),
                                "G_ema": G_ema.state_dict(), "opt_G": opt_G.state_dict(), "opt_Di": opt_Di.state_dict(),
                                "opt_Dl": opt_Dl.state_dict(), "step": step, "nimg": nimg, "best_fid": best_fid,
                                "elapsed": clock.now(), "rng": rng_state()})

        with torch.no_grad(), autocast(device, cfg.amp):
            zr, _ = sample_real(64)
            save_grid(decode(zr).float().clamp(-1, 1), rd.path / "samples" / "decoded_real.png")
        save_grid(img_bank[:64].float() / 127.5 - 1, rd.path / "samples" / "real.png")

        def train_step(g_net, ot_w):
            # ---- critics ----
            z_real, c = sample_real(cfg.batch)
            with torch.no_grad(), autocast(device, cfg.amp):
                x_real = decode(z_real).float()
                z_fake, _ = generate(g_net, cfg.batch, c)
                x_fake = decode(z_fake).float()
            d_stats = trainer.d_step({"img": x_real, "lat": z_real}, {"img": x_fake, "lat": z_fake.float()}, c, step)
            # ---- generator ----
            z_real, c = sample_real(cfg.batch)
            with torch.no_grad(), autocast(device, cfg.amp):
                x_real = decode(z_real).float()
            with autocast(device, cfg.amp):
                z_fake, z_in = generate(g_net, cfg.batch, c)
                x_fake = decode(z_fake)
            adv, g_stats = trainer.g_adversarial_loss({"img": x_real, "lat": z_real},
                                                      {"img": x_fake.float(), "lat": z_fake.float()}, c)
            ot = (z_fake.float() - z_in).square().mean()
            loss = adv + ot_w * ot
            opt_G.zero_grad(set_to_none=True)
            loss.backward()
            opt_G.step()
            return d_stats, g_stats, z_fake.detach(), z_real, ot.detach()

        budget = cfg.hours * 3600
        clock = TrainClock(elapsed)
        sample_timer, eval_timer, ckpt_timer = Every(cfg.sample_minutes * 60), Every(cfg.eval_minutes * 60), \
            Every(cfg.ckpt_minutes * 60)
        t_log, n_log = time.time(), 0
        clock.start()
        while True:
            progress = clock.now() / budget
            if (cfg.max_steps and step >= cfg.max_steps) or (not cfg.max_steps and progress >= 1.0):
                break
            lr = cosine_interp(*cfg.lr, progress)
            beta2 = cosine_interp(*cfg.beta2, progress)
            slot_img.gamma = log_cosine_interp(*cfg.gamma_img, progress)
            slot_lat.gamma = log_cosine_interp(*cfg.gamma_lat, progress)
            crit_img.p = crit_lat.p = cosine_interp(*cfg.aug_p, progress)
            ot_w = log_cosine_interp(*cfg.ot_weight, progress) if max(cfg.ot_weight) > 0 else 0.0
            for opt in (opt_G, opt_Di, opt_Dl):
                set_optimizer_hparams(opt, lr=lr, beta2=beta2)

            try:
                d_stats, g_stats, z_fake, z_real, ot = train_step(G_fast if G_fast is not None else G, ot_w)
            except Exception as exc:  # noqa: BLE001
                if compiled and step < 5:
                    rd.print(f"torch.compile failed ({type(exc).__name__}: {str(exc)[:300]}); using eager mode")
                    torch._dynamo.reset()
                    G_fast, compiled = None, False
                    crit_img.set_fast(None)
                    crit_lat.set_fast(None)
                    continue
                raise
            nimg += cfg.batch
            step += 1
            G_ema.update(G, ema_beta(cfg.batch, cosine_interp(*cfg.ema_kimg, progress), nimg))
            n_log += 1

            if step % cfg.log_every == 0:
                dt = time.time() - t_log
                with torch.no_grad():
                    zf = z_fake.detach().float()
                    lat_stats = {"fake_lat_mean": float(zf.mean()), "fake_lat_std": float(zf.std()),
                                 "real_lat_std": float(z_real.std()),
                                 "fake_batch_div": float(torch.pdist(zf.flatten(1)[:64]).mean()),
                                 "real_batch_div": float(torch.pdist(z_real.flatten(1)[:64]).mean())}
                stats = {**to_floats(d_stats), **to_floats(g_stats), **lat_stats, "ot": float(ot), "ot_w": ot_w, "lr": lr,
                         "gamma_img": slot_img.gamma, "gamma_lat": slot_lat.gamma, "aug_p": crit_img.p,
                         "it_per_s": n_log / max(dt, 1e-9)}
                rd.log_metrics(kind="train", step=step, kimg=nimg / 1000, hours=clock.now() / 3600, **stats)
                rd.write_status(step=step, kimg=nimg / 1000, hours=clock.now() / 3600, budget_hours=cfg.hours,
                                progress=progress, best_fid=best_fid, **stats)
                if step % (cfg.log_every * 10) == 0 or cfg.max_steps:
                    rd.print(f"step {step} " + " ".join(f"{k}={v:.4g}" for k, v in stats.items()))
                t_log, n_log = time.time(), 0
            if sample_timer() or (cfg.max_steps and step == cfg.max_steps):
                clock.pause()
                with torch.no_grad(), autocast(device, cfg.amp):
                    grid = decode(G_ema.model(fixed_z, fixed_zg, fixed_c)).float().clamp(-1, 1)
                save_grid(grid, rd.path / "samples" / f"step{step:07d}.png")
                save_grid(grid, rd.path / "samples" / "latest.png")
                clock.start()
            if eval_timer():
                clock.pause()
                evaluate(cfg.eval_n, "periodic")
                clock.start()
            if ckpt_timer():
                clock.pause()
                checkpoint()
                clock.start()
        clock.pause()
        checkpoint()
        final = evaluate(cfg.final_eval_n, "final")
        with torch.no_grad(), autocast(device, cfg.amp):
            save_grid(decode(G_ema.model(fixed_z, fixed_zg, fixed_c)).float().clamp(-1, 1),
                      rd.path / "samples" / "final.png")
            if num_classes:
                cls = torch.randperm(num_classes, device=device)[:16].repeat_interleave(8)
                zz, _ = generate(G_ema.model, len(cls), cls)
                save_grid(decode(zz).float().clamp(-1, 1), rd.path / "samples" / "final_per_class.png", nrow=8)
        torch.save({"G_ema": G_ema.state_dict(), "config": dataclasses.asdict(cfg), "num_classes": num_classes,
                    "label_source": label_source, "vae_path": str(vae_path)}, rd.path / "G_ema_final.pt")
        vae_metrics = {}
        vm = run_dir.parent / "vae" / "vae_metrics.json"
        if vm.exists():
            import json

            vae_metrics = json.loads(vm.read_text())
        summary = {"final": final, "vae": {k: vae_metrics.get(k) for k in ("psnr", "ssim", "lpips", "rfid")},
                   "steps": step, "kimg": nimg / 1000, "hours": clock.now() / 3600, "num_classes": num_classes,
                   "label_source": label_source}
        rd.write_json("final_metrics.json", summary)
        (rd.path / "report.md").write_text(summarize_metrics_md(
            "Task 3 - latent GAN final metrics",
            {**{k: v for k, v in final.items() if not isinstance(v, str)},
             **{f"vae_{k}": v for k, v in summary["vae"].items() if v is not None}},
            ["fid/kid/precision/recall: Dec(G(z)) vs real images; vs_decoded_*: vs VAE reconstructions; "
             "vae_rfid: reconstruction floor.", "Samples: samples/final.png, samples/final_per_class.png"]))
        rd.mark_done(summary)
        rd.print("DONE")
        return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--data_path", default=None)
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--stage", default=None)
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--vae_hours", type=float, default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--override", default=None)
    args = ap.parse_args(argv)
    cfg = Task3Config()
    if args.smoke:
        cfg = smoke_config(cfg)
    for k in ["dataset", "data_path", "run_dir", "stage", "hours", "labels", "max_steps", "device"]:
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)
    if args.vae_hours is not None:
        cfg.vae_stage.vae_hours = args.vae_hours
    cfg.force = args.force
    apply_overrides(cfg, parse_override(args.override))
    return run(cfg)


if __name__ == "__main__":
    main()
