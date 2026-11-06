"""Task 1 - a modern pixel-space GAN at 64x64 with minimal mode collapse.

Recipe (see the design spec for references):
  * R3GAN objective: RpGAN + R1 + R2 (lazy, every ``reg_every`` steps), cosine
    schedules for lr / beta2 / gamma / EMA half-life / augmentation probability.
  * Normalization-free depthwise-separable inverted-bottleneck blocks, Fix-up init,
    multi-head self-attention with QK-norm at 16x16 (G) and 16x16 + 8x8 (D).
  * Conditioning: class labels, or - for unlabelled data such as the Genshin fan-art
    set - CLIP k-means pseudo-classes (cluster-conditioned GAN) as an explicit
    anti-mode-collapse mechanism.  Projection discriminator.
  * DiffAugment on both real and fake images; x-flip dataset augmentation; EMA G.
  * Mode coverage is *measured*: FID, KID, precision and recall (k=3) on 10k samples
    during training and 50k (or |dataset|) at the end.

Usage (Colab):
    python -m cocogoat.tasks.task1_gan64 --dataset genshin --hours 3.5
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

from cocogoat.data import GPUImageBank, diff_augment, load_array_images, load_tinyimagenet, resize_uint8
from cocogoat.metrics import ImageEvaluator
from cocogoat.models.gan2d import Discriminator, DiscriminatorConfig, Generator, GeneratorConfig
from cocogoat.nn import count_params
from cocogoat.tasks.common import (
    DRIVE_ROOT,
    fresh_smoke_run,
    LOCAL_CACHE,
    USER_DATA,
    apply_overrides,
    make_feature_fn,
    parse_override,
    load_compatible,
    resume_or_skip,
    rotate_old_error,
    sample_labels,
    summarize_metrics_md,
)
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


# expansion ratio per resolution: the inverted bottlenecks at 64/32 px dominate memory traffic
LIGHT_HIGH_RES = {64: 2.0, 32: 2.0, 16: 3.0, 8: 3.0, 4: 3.0}


@dataclass
class Task1Config:
    run_dir: str = str(DRIVE_ROOT / "runs" / "task1")
    dataset: str = "genshin"            # genshin | tinyimagenet | array
    data_path: str = str(USER_DATA / "Complete_dataset.npy")
    drive_cache: str = str(DRIVE_ROOT / "cache")
    resolution: int = 64
    labels: str = "auto"                # auto | none | clip_kmeans | dataset
    num_clusters: int = 100
    clip_model: str = "openai/clip-vit-base-patch32"
    batch: int = 256
    hours: float = 3.5
    max_steps: int = 0                  # >0: stop after this many steps (tests)
    lr: tuple = (2e-4, 5e-5)
    beta2: tuple = (0.9, 0.99)
    gamma: tuple = (2.0, 1.0)           # R1/R2 weight; the 3.5 h run diverged once gamma fell below ~0.3
    gap_limit: float = 3.0              # smoothed D gap (real - fake logit) above this => back off gamma
    gamma_backoff: float = 1.5          # ... by this factor (capped at gamma[0]); the raised floor is kept
    ema_kimg: tuple = (10.0, 300.0)
    aug_p: tuple = (0.2, 0.6)
    aug_policy: str = "color,translation,cutout"
    reg_every: int = 8
    reg_fp32: bool = False              # R1/R2 passes in bf16 (fp32 is ~2x slower)
    xflip: bool = True
    amp: bool = True
    channels_last: bool = True
    compile: bool = True                # torch.compile G and D for the non-regularisation passes
    G: GeneratorConfig = field(default_factory=lambda: GeneratorConfig(expansion=dict(LIGHT_HIGH_RES)))
    D: DiscriminatorConfig = field(default_factory=lambda: DiscriminatorConfig(expansion=dict(LIGHT_HIGH_RES)))
    log_every: int = 100
    sample_minutes: float = 15.0
    eval_minutes: float = 20.0
    ckpt_minutes: float = 20.0
    eval_n: int = 10000
    final_eval_n: int = 50000
    metrics: str = "inception"          # inception | random (CPU tests only)
    seed: int = 0
    device: str | None = None
    force: bool = False
    eval_best: bool = False             # only evaluate ckpt/best_G_ema.pt of an existing run (no training)


def smoke_config(cfg: Task1Config) -> Task1Config:
    """Tiny models and cadences that exercise every code path in seconds (CPU ok)."""
    cfg.G = GeneratorConfig(resolution=cfg.resolution, z_dim=16, w_dim=32, cond_dim=16, blocks=1,
                            channels={4: 32, 8: 32, 16: 24, 32: 16, 64: 16}, expansion=2.0)
    cfg.D = DiscriminatorConfig(resolution=cfg.resolution, blocks=1, channels={64: 16, 32: 16, 16: 24, 8: 32, 4: 32},
                                expansion=2.0)
    cfg.batch = 8
    cfg.max_steps = cfg.max_steps or 9
    cfg.log_every = 2
    cfg.sample_minutes = 0.0
    cfg.eval_minutes = 0.0
    cfg.ckpt_minutes = 0.0
    cfg.eval_n = 64
    cfg.final_eval_n = 64
    cfg.metrics = "random"
    cfg.amp = False
    cfg.channels_last = False
    cfg.compile = False
    return cfg


class PixelCritic(Critic):
    keys = ("img",)

    def __init__(self, D: Discriminator, policy: str, p: float):
        super().__init__()
        self.D = D
        self.policy = policy
        self.p = p
        object.__setattr__(self, "D_fast", None)  # compiled view (not registered as a submodule)

    def set_fast(self, module) -> None:
        object.__setattr__(self, "D_fast", module)

    def encode(self, sample):
        return {"img": diff_augment(sample["img"], self.policy, self.p)}

    def head(self, enc, cond):
        net = self.D_fast if (self.fast and self.D_fast is not None) else self.D
        return net(enc["img"], cond)


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------

def load_data(cfg: Task1Config, rd: RunDir, device: torch.device):
    """Returns (uint8 NHWC images at cfg.resolution, int64 labels or None, num_classes, label_source)."""
    labels, source = None, "none"
    drive_cache = Path(cfg.drive_cache) if Path(cfg.drive_cache).parent.exists() else None
    if cfg.dataset == "tinyimagenet":
        imgs, labels = load_tinyimagenet(LOCAL_CACHE)
        imgs = resize_uint8(imgs, cfg.resolution, device=device)
        source = "dataset"
        if cfg.labels in ("none",):
            labels, source = None, "none"
        full_res = imgs
    elif cfg.dataset in ("genshin", "array"):
        if cfg.dataset == "array" and not Path(cfg.data_path).exists():
            raise FileNotFoundError(cfg.data_path)
        if not Path(cfg.data_path).exists():
            raise FileNotFoundError(
                f"{cfg.data_path} not found. Copy Complete_dataset.npy there or run with --dataset tinyimagenet.")
        full_res = load_array_images(cfg.data_path, cache_dir=LOCAL_CACHE, drive_cache=drive_cache)
        rd.print(f"loaded {cfg.data_path}: {full_res.shape} uint8")
        imgs = resize_uint8(full_res, cfg.resolution, device=device)
    else:
        raise ValueError(cfg.dataset)

    want = cfg.labels
    if want == "auto":
        want = "dataset" if labels is not None else "clip_kmeans"
    if want == "clip_kmeans" and cfg.metrics != "random":
        from cocogoat.data.labels import clip_pseudo_labels

        cache = (drive_cache or LOCAL_CACHE) / f"{cfg.dataset}_clip_k{cfg.num_clusters}.npz"
        labels, _ = clip_pseudo_labels(np.asarray(full_res), cfg.num_clusters, cache=cache,
                                       clip_name=cfg.clip_model, device=str(device))
        source = f"clip_kmeans_k{cfg.num_clusters}"
    elif want == "clip_kmeans":  # smoke test: fake clusters
        labels = np.random.default_rng(0).integers(0, cfg.num_clusters, len(imgs))
        source = "random_test_labels"
    elif want == "none":
        labels, source = None, "none"
    num_classes = int(labels.max()) + 1 if labels is not None else 0
    return np.asarray(imgs), (None if labels is None else np.asarray(labels, dtype=np.int64)), num_classes, source


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------

def run(cfg: Task1Config) -> dict:
    device = setup_torch(cfg.device)
    seed_everything(cfg.seed)
    run_path = Path(cfg.run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    rotate_old_error(run_path)
    rd = RunDir(run_path, config=cfg)
    with rd.capture_errors():
        if not cfg.eval_best and resume_or_skip(rd, cfg.hours, cfg.force):
            return {}
        rd.print(f"env: {describe_env()}")
        images, labels, num_classes, label_source = load_data(cfg, rd, device)
        rd.print(f"data: {images.shape}, classes={num_classes} ({label_source})")
        bank = GPUImageBank(images, labels, device=device, xflip=cfg.xflip)
        prior = bank.label_prior()

        cfg.G.resolution = cfg.D.resolution = cfg.resolution
        cfg.G.num_classes = cfg.D.num_classes = num_classes
        G = Generator(cfg.G).to(device)
        D = Discriminator(cfg.D).to(device)
        if cfg.channels_last and device.type == "cuda":
            G = G.to(memory_format=torch.channels_last)
            D = D.to(memory_format=torch.channels_last)
        G_ema = EMA(G)
        rd.print(f"G params {count_params(G) / 1e6:.2f}M, D params {count_params(D) / 1e6:.2f}M")
        opt_G = make_adam(G.parameters(), cfg.lr[0], cfg.beta2[0])
        opt_D = make_adam(D.parameters(), cfg.lr[0], cfg.beta2[0])
        critic = PixelCritic(D, cfg.aug_policy, cfg.aug_p[0])
        slot = CriticSlot("img", critic, opt_D, g_weight=1.0, gamma=cfg.gamma[0])
        trainer = AdversarialTrainer([slot], device, amp=cfg.amp, reg_every=cfg.reg_every, reg_fp32=cfg.reg_fp32)
        rd.print(f"R1/R2 depthwise implementation: {'shift' if trainer.shift_for_reg else 'native'} "
                 f"(benchmark ms: {trainer.shift_timings})")
        G_fast = maybe_compile(G, cfg.compile)
        critic.set_fast(maybe_compile(D, cfg.compile))
        compiled = G_fast is not None
        if compiled:
            rd.print("torch.compile enabled for G and D (first steps include compilation)")

        step, nimg, best_fid, elapsed = 0, 0, float("inf"), 0.0
        gamma_hold, gap_ema = 0.0, None
        state = rd.load_checkpoint()
        if load_compatible(rd, {"G": G, "D": D, "G_ema": G_ema, "opt_G": opt_G, "opt_D": opt_D}, state):
            step, nimg, best_fid, elapsed = state["step"], state["nimg"], state["best_fid"], state["elapsed"]
            gamma_hold = state.get("gamma_hold", 0.0)
            set_rng_state(state["rng"])
            rd.print(f"resumed at step {step}, {nimg / 1e6:.2f}M images, {elapsed / 3600:.2f} h")

        # fixed samples for grids: 8 rows (classes) x 8 columns
        gen = torch.Generator(device=device).manual_seed(1234)
        fixed_z = torch.randn(64, cfg.G.z_dim, device=device, generator=gen)
        if num_classes:
            top = torch.argsort(torch.bincount(bank.labels, minlength=num_classes), descending=True)[:8]
            fixed_c = top[torch.arange(8, device=device) % len(top)].repeat_interleave(8)
        else:
            fixed_c = None
        real_x, _ = bank.sample(64)
        save_grid(real_x, rd.path / "samples" / "real.png")

        feature_fn = make_feature_fn(cfg.metrics, device)
        evaluator = None

        def get_evaluator():
            nonlocal evaluator
            if evaluator is None:
                evaluator = ImageEvaluator(feature_fn, (x for x, _ in bank.iterate(500)),
                                           cache=rd.path / f"real_stats_{cfg.metrics}.npz")
            return evaluator

        @torch.no_grad()
        def sample_fn(n: int) -> torch.Tensor:
            z = torch.randn(n, cfg.G.z_dim, device=device)
            c = sample_labels(prior, n, device)
            with autocast(device, cfg.amp):
                return G_ema.model(z, c).float()

        def checkpoint():
            rd.save_checkpoint({"G": G.state_dict(), "D": D.state_dict(), "G_ema": G_ema.state_dict(),
                                "opt_G": opt_G.state_dict(), "opt_D": opt_D.state_dict(), "step": step,
                                "nimg": nimg, "best_fid": best_fid, "elapsed": clock.now(), "rng": rng_state(),
                                "gamma_hold": gamma_hold,
                                "num_classes": num_classes, "label_source": label_source,
                                "config": dataclasses.asdict(cfg)})

        def evaluate(n: int, tag: str, track_best: bool = True) -> dict:
            nonlocal best_fid
            G_ema.model.eval()
            t0 = time.time()
            m = get_evaluator().evaluate(sample_fn, n=n)
            m.update(step=step, kimg=nimg / 1000, hours=clock.now() / 3600, eval_seconds=time.time() - t0, tag=tag)
            rd.log_metrics(kind="eval", **m)
            rd.print(f"[eval {tag}] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                                 for k, v in m.items()))
            if track_best and m["fid"] < best_fid:
                best_fid = m["fid"]
                torch.save({"G_ema": G_ema.state_dict(), "config": dataclasses.asdict(cfg), "metrics": m,
                            "num_classes": num_classes}, rd.path / "ckpt" / "best_G_ema.pt")
            return m

        def train_step(g_net):
            # ---- discriminator step ----
            x_real, c = bank.sample(cfg.batch)
            z = torch.randn(cfg.batch, cfg.G.z_dim, device=device)
            with torch.no_grad(), autocast(device, cfg.amp):
                x_fake = g_net(z, c)
            d_stats = trainer.d_step({"img": x_real}, {"img": x_fake}, c, step)
            # ---- generator step ----
            x_real, c = bank.sample(cfg.batch)
            z = torch.randn(cfg.batch, cfg.G.z_dim, device=device)
            with autocast(device, cfg.amp):
                x_fake = g_net(z, c)
            g_loss, g_stats = trainer.g_adversarial_loss({"img": x_real}, {"img": x_fake}, c)
            opt_G.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_G.step()
            return d_stats, g_stats

        def finalize_best() -> dict | None:
            """Full evaluation of the best periodic checkpoint; exports G_ema_best.pt and its sample grids."""
            best_path = rd.path / "ckpt" / "best_G_ema.pt"
            if not best_path.exists():
                return None
            best = torch.load(best_path, map_location="cpu", weights_only=False)
            current = {k: v.clone() for k, v in G_ema.state_dict().items()}
            G_ema.load_state_dict(best["G_ema"])
            m = evaluate(cfg.final_eval_n, "best", track_best=False)
            m["periodic_fid"] = best["metrics"].get("fid")
            m.update({k: best["metrics"][k] for k in ("step", "kimg", "hours") if k in best["metrics"]})
            torch.save({"G_ema": best["G_ema"], "config": dataclasses.asdict(cfg), "num_classes": num_classes,
                        "label_source": label_source, "metrics": m}, rd.path / "G_ema_best.pt")
            with torch.no_grad(), autocast(device, cfg.amp):
                save_grid(G_ema.model(fixed_z, fixed_c).float(), rd.path / "samples" / "best.png")
                if num_classes:
                    cls = torch.randperm(num_classes, device=device)[:16].repeat_interleave(8)
                    z = torch.randn(len(cls), cfg.G.z_dim, device=device)
                    save_grid(G_ema.model(z, cls).float(), rd.path / "samples" / "best_per_class.png", nrow=8)
            G_ema.load_state_dict(current)
            return m

        if cfg.eval_best:
            clock = TrainClock(elapsed)
            m = finalize_best()
            if m is None:
                raise FileNotFoundError(f"no ckpt/best_G_ema.pt in {rd.path}")
            rd.write_json("best_metrics.json", m)
            rd.print("best checkpoint evaluated: G_ema_best.pt, samples/best.png, best_metrics.json")
            return m

        budget = cfg.hours * 3600
        clock = TrainClock(elapsed)
        sample_timer, eval_timer, ckpt_timer = Every(cfg.sample_minutes * 60), Every(cfg.eval_minutes * 60), \
            Every(cfg.ckpt_minutes * 60)
        t_log, n_log = time.time(), 0
        nonfinite_streak = 0
        rd.print(f"training for {cfg.hours} h budget (already {elapsed / 3600:.2f} h)")
        clock.start()
        while True:
            progress = clock.now() / budget
            if (cfg.max_steps and step >= cfg.max_steps) or (not cfg.max_steps and progress >= 1.0):
                break
            lr = cosine_interp(*cfg.lr, progress)
            beta2 = cosine_interp(*cfg.beta2, progress)
            slot.gamma = max(log_cosine_interp(*cfg.gamma, progress), gamma_hold)
            critic.p = cosine_interp(*cfg.aug_p, progress)
            for opt in (opt_G, opt_D):
                set_optimizer_hparams(opt, lr=lr, beta2=beta2)

            try:
                d_stats, g_stats = train_step(G_fast if G_fast is not None else G)
            except Exception as exc:  # noqa: BLE001
                if compiled and step < 5:
                    rd.print(f"torch.compile failed ({type(exc).__name__}: {str(exc)[:300]}); using eager mode")
                    torch._dynamo.reset()
                    G_fast, compiled = None, False
                    critic.set_fast(None)
                    continue
                raise

            nimg += cfg.batch
            step += 1
            G_ema.update(G, ema_beta(cfg.batch, cosine_interp(*cfg.ema_kimg, progress), nimg))
            n_log += 1

            if step % cfg.log_every == 0:
                dt = time.time() - t_log
                stats = {**to_floats(d_stats), **to_floats(g_stats), "lr": lr, "beta2": beta2, "gamma": slot.gamma,
                         "aug_p": critic.p, "it_per_s": n_log / max(dt, 1e-9),
                         "img_per_s": n_log * cfg.batch / max(dt, 1e-9)}
                if not all(math.isfinite(v) for v in stats.values()):
                    nonfinite_streak += 1
                    rd.print(f"non-finite training statistics at step {step}: {stats}")
                    if nonfinite_streak >= 3:
                        raise FloatingPointError("training diverged (non-finite losses); resume from last checkpoint")
                else:
                    nonfinite_streak = 0
                # divergence guard: D running away from G (gap exploding) => raise the R1/R2 floor
                gap = stats.get("img/gap")
                if gap is not None and math.isfinite(gap):
                    gap_ema = gap if gap_ema is None else 0.7 * gap_ema + 0.3 * gap
                    if gap_ema > cfg.gap_limit and gamma_hold < cfg.gamma[0]:
                        gamma_hold = min(cfg.gamma[0], slot.gamma * cfg.gamma_backoff)
                        rd.print(f"step {step}: D gap {gap_ema:.2f} > {cfg.gap_limit} -> gamma floor {gamma_hold:.3g}")
                        gap_ema = None
                stats["gamma_hold"] = gamma_hold
                rd.log_metrics(kind="train", step=step, kimg=nimg / 1000, hours=clock.now() / 3600, **stats)
                rd.write_status(step=step, kimg=nimg / 1000, hours=clock.now() / 3600, budget_hours=cfg.hours,
                                progress=progress, best_fid=best_fid, **stats)
                if step % (cfg.log_every * 10) == 0 or cfg.max_steps:
                    rd.print(f"step {step} kimg {nimg / 1000:.1f} " + " ".join(f"{k}={v:.4g}" for k, v in stats.items()))
                t_log, n_log = time.time(), 0

            if sample_timer() or (cfg.max_steps and step == cfg.max_steps):
                clock.pause()
                with torch.no_grad(), autocast(device, cfg.amp):
                    grid = G_ema.model(fixed_z, fixed_c).float()
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

        # ---- final evaluation ----
        n_final = cfg.final_eval_n
        final = evaluate(n_final, "final")
        best_m = finalize_best()
        torch.save({"G_ema": G_ema.state_dict(), "config": dataclasses.asdict(cfg), "num_classes": num_classes,
                    "label_source": label_source}, rd.path / "G_ema_final.pt")
        with torch.no_grad(), autocast(device, cfg.amp):
            grid = G_ema.model(fixed_z, fixed_c).float()
        save_grid(grid, rd.path / "samples" / "final.png")
        if num_classes:
            # one row per class for 16 random classes - shows class conditioning / diversity
            cls = torch.randperm(num_classes, device=device)[:16].repeat_interleave(8)
            z = torch.randn(len(cls), cfg.G.z_dim, device=device)
            with torch.no_grad(), autocast(device, cfg.amp):
                save_grid(G_ema.model(z, cls).float(), rd.path / "samples" / "final_per_class.png", nrow=8)
        summary = {"final": final, "best": best_m, "steps": step,
                   "budget_hours": cfg.hours,
                   "kimg": nimg / 1000, "hours": clock.now() / 3600, "num_classes": num_classes,
                   "label_source": label_source, "dataset": cfg.dataset, "G_params_M": count_params(G) / 1e6,
                   "D_params_M": count_params(D) / 1e6}
        rd.write_json("final_metrics.json", summary)
        (rd.path / "report.md").write_text(summarize_metrics_md(
            "Task 1 - GAN 64x64 final metrics", {**{k: v for k, v in final.items() if not isinstance(v, str)},
                                                 "steps": step, "kimg": nimg / 1000, "dataset": cfg.dataset,
                                                 "labels": label_source,
                                                 **({f"best_{k}": v for k, v in best_m.items()
                                                     if not isinstance(v, str)} if best_m else {})},
            ["Final model: samples/final.png, samples/final_per_class.png; weights: G_ema_final.pt",
             "Best periodic checkpoint (re-evaluated on the final sample count): samples/best.png, "
             "samples/best_per_class.png; weights: G_ema_best.pt"]))
        rd.mark_done(summary)
        rd.print("DONE")
        return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--data_path", default=None)
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--num_clusters", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--metrics", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny models, few steps (code-path test)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--eval_best", action="store_true",
                    help="no training: evaluate ckpt/best_G_ema.pt of RUN_DIR on the final sample count")
    ap.add_argument("--override", default=None, help='JSON, e.g. \'{"G": {"expansion": 4}, "gamma": [1, 0.1]}\'')
    args = ap.parse_args(argv)
    cfg = Task1Config()
    if args.smoke:
        cfg = smoke_config(cfg)
    for k in ["dataset", "data_path", "run_dir", "hours", "batch", "labels", "num_clusters", "max_steps", "metrics",
              "device"]:
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)
    cfg.force = args.force
    cfg.eval_best = args.eval_best
    apply_overrides(cfg, parse_override(args.override))
    if args.smoke:  # smoke tests always start from scratch
        fresh_smoke_run(cfg.run_dir)
    return run(cfg)


if __name__ == "__main__":
    main()
