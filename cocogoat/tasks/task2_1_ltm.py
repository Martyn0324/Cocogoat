"""Task 2.1 - does a surprise-gated long-term memory help the latent world model?  (A/B experiment)

Two arms, identical except for the memory bank:
  base : Task 2 world model (space-time transformer + multi-scale EMA memory), trained and rolled out in
         chunks with a 16-frame sliding attention window and rotary (relative) time positions.
  ltm  : the same + a bank of K past frame latents chosen by prediction error (surprise), read by
         cross-attention (``cocogoat/models/ltm.py``; the ViVAE design notes' long-term memory).
Both arms: same frozen VAE (Task 2's, if it exists), same data, same seed, same budget, 64-frame
training clips, evaluation with 64 context + 64 generated frames.

Datasets: ``mmnist`` (smoke test: barely needs long-term memory) and ``shelter`` (main experiment).
Shelter averages ~21 frames per shot, so 64-frame clips cannot stay inside one shot: Task 2.1 trains
and evaluates across scene cuts (``cross_cuts``), which is also where recalling earlier shots matters.

Output: ``runs/task2_1/<dataset>/{base,ltm}/`` (metrics, rollout PNG/GIF) and ``compare.md`` /
``compare.json`` / ``compare.png`` in ``runs/task2_1/<dataset>/``.

Usage (Colab):
    python -m cocogoat.tasks.task2_1_ltm --dataset mmnist
    python -m cocogoat.tasks.task2_1_ltm --dataset shelter
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cocogoat.models.ltm import ChunkedWorldModel, LTMConfig
from cocogoat.models.vae import VAEConfig, load_vae
from cocogoat.models.world import WorldModel, WorldModelConfig
from cocogoat.nn import count_params
from cocogoat.tasks.common import DRIVE_ROOT, apply_overrides, fresh_smoke_run, parse_override, rotate_old_error
from cocogoat.tasks.task2_world import (
    Task2Config,
    encode_clips,
    evaluate_world_model,
    fit_eval_length,
    load_banks,
    smoke_config as task2_smoke_config,
    train_vae,
    write_wm_report,
)
from cocogoat.train import (
    Every,
    RunDir,
    TrainClock,
    autocast,
    cosine_interp,
    describe_env,
    rng_state,
    seed_everything,
    set_optimizer_hparams,
    set_rng_state,
    setup_torch,
)


@dataclass
class Task21Config(Task2Config):
    run_root: str = str(DRIVE_ROOT / "runs" / "task2_1")
    vae_from: str | None = None              # default: the Task 2 VAE of the same dataset (trained here if absent)
    arms: tuple = ("base", "ltm")
    clip_len: int = 64
    window: int = 16                         # chunk length: each prediction attends to the last W+1..2W frames
    ltm_slots: int = 8
    ltm_every: int = 2                       # memory cross-attention in every 2nd block
    ltm_decay: float = 0.99                  # priority decay per frame of age
    cross_cuts: bool = True                  # Shelter: clips may cross scene cuts (shots average ~21 frames)
    wm_batch: int = 32
    eval_context: int = 64
    eval_horizon: int = 64
    eval_clips: int = 128
    mmnist_train_seqs: int = 2000            # 128-frame sequences (enough for 64-frame clips and 128-frame evals)
    mmnist_test_seqs: int = 256
    mmnist_frames: int = 128


def smoke_config(cfg: Task21Config) -> Task21Config:
    cfg = task2_smoke_config(cfg)
    cfg.mmnist_frames = 16
    cfg.clip_len, cfg.window = 8, 4
    cfg.eval_context, cfg.eval_horizon, cfg.eval_clips = 6, 6, 4
    cfg.ltm_slots, cfg.ltm_every = 2, 1
    cfg.rollout_k = 2
    cfg.gif_clips = 2
    return cfg


def default_vae_path(cfg: Task21Config) -> Path:
    if cfg.vae_from:
        return Path(cfg.vae_from)
    return DRIVE_ROOT / "runs" / ("task2_smoke" if cfg.smoke else "task2") / cfg.dataset / "vae" / "vae.pt"


def arm_wm_config(cfg: Task21Config, arm: str, vae) -> WorldModelConfig:
    wm = dataclasses.replace(cfg.wm)
    wm.latent_channels = vae.cfg.latent_channels
    wm.latent_size = cfg.resolution // vae.cfg.downsample
    wm.rope_time, wm.time_window = True, 0     # chunk-aligned context (see ChunkedWorldModel)
    wm.max_frames = 2 * cfg.window
    wm.ltm_every = cfg.ltm_every if arm == "ltm" else 0
    return wm


def train_arm(cfg: Task21Config, arm: str, vae_path: Path, train, test, device, run_dir: Path) -> dict:
    rd = RunDir(run_dir, config=cfg)
    with rd.capture_errors():
        if rd.done and not cfg.force:
            rd.print(f"arm '{arm}' already DONE")
            return json.loads((run_dir / "wm_metrics.json").read_text())
        vae = load_vae(str(vae_path), device).requires_grad_(False)
        seed_everything(cfg.seed)
        wm = WorldModel(arm_wm_config(cfg, arm, vae)).to(device)
        runner = ChunkedWorldModel(wm, LTMConfig(slots=cfg.ltm_slots if arm == "ltm" else 0, window=cfg.window,
                                                 decay=cfg.ltm_decay))
        seed_everything(cfg.seed + 1)  # same data order and noise in both arms, whatever the extra parameters
        rd.print(f"arm '{arm}': world model params {count_params(wm) / 1e6:.2f}M; clip {cfg.clip_len}, window "
                 f"{cfg.window}, memory slots {runner.cfg.slots}")
        opt = torch.optim.AdamW(wm.parameters(), lr=cfg.wm_lr[0], betas=(0.9, 0.95), weight_decay=0.01)
        step, elapsed, best = 0, 0.0, float("inf")
        st = rd.load_checkpoint()
        if st is not None:
            wm.load_state_dict(st["wm"])
            opt.load_state_dict(st["opt"])
            step, elapsed, best = st["step"], st["elapsed"], st.get("best", best)
            set_rng_state(st["rng"])
            rd.print(f"resumed at step {step}")
        fit_eval_length(cfg, test, rd)
        eval_len = cfg.eval_context + cfg.eval_horizon
        n_avail = len(test.valid_starts(eval_len, cfg.clip_stride))
        n_eval = max(1, min(cfg.eval_clips, n_avail, (16384 * 64 * 64) // (eval_len * cfg.resolution ** 2)))
        eval_clips = test.fixed_clips(n_eval, eval_len, cfg.clip_stride, seed=0)
        rd.print(f"evaluation: {n_eval} held-out clips x ({cfg.eval_context} context + {cfg.eval_horizon} generated)")
        clock = TrainClock(elapsed)
        budget = cfg.wm_hours * 3600
        ckpt_timer, eval_timer = Every(cfg.ckpt_minutes * 60), Every(cfg.eval_minutes * 60)

        def checkpoint():
            rd.save_checkpoint({"wm": wm.state_dict(), "opt": opt.state_dict(), "step": step, "elapsed": clock.now(),
                                "best": best, "rng": rng_state(), "wm_config": dataclasses.asdict(wm.cfg)})

        def evaluate(tag: str) -> dict:
            nonlocal best
            wm.eval()
            m = evaluate_world_model(cfg, vae, runner, eval_clips, device, rd, tag)
            m.update({f"eval_{k}": v for k, v in runner.last_stats.items()})
            wm.train()
            rd.log_metrics(kind="wm_eval", step=step, hours=clock.now() / 3600, tag=tag, **m)
            score = float(np.mean(m["model_latent_mse"]))
            if score < best:
                best = score
                torch.save({"wm": wm.state_dict(), "wm_config": dataclasses.asdict(wm.cfg)}, rd.path / "wm_best.pt")
            rd.print(f"[{arm} eval {tag}] latent_mse model={np.mean(m['model_latent_mse']):.4f} "
                     f"copy={np.mean(m['copy_latent_mse']):.4f} psnr model={np.mean(m['model_psnr']):.2f} "
                     f"copy={np.mean(m['copy_psnr']):.2f} " + " ".join(f"{k}={v:.3g}" for k, v in
                                                                       runner.last_stats.items()))
            return m

        t_log = time.time()
        T, k = cfg.clip_len, cfg.rollout_k
        clock.start()
        while True:
            progress = clock.now() / budget
            if (cfg.max_steps and step >= cfg.max_steps) or (not cfg.max_steps and progress >= 1):
                break
            warm = min(1.0, (step + 1) / cfg.wm_warmup_steps)
            lr = cosine_interp(*cfg.wm_lr, progress) * warm
            set_optimizer_hparams(opt, lr=lr)
            clips = train.sample_clips(cfg.wm_batch, T, cfg.clip_stride)
            z = encode_clips(vae, clips, cfg.amp, device)
            sigma = torch.rand(z.shape[0], 1, 1, 1, 1, device=device) * cfg.ctx_noise
            z_in = z + sigma * torch.randn_like(z)
            opt.zero_grad(set_to_none=True)
            with autocast(device, cfg.amp):
                tf_mse, banks = runner.teacher_forced(z_in, z, backward_fn=lambda loss: loss.backward())
            stats = {"tf_mse": tf_mse, "lr": lr, **runner.last_stats}
            if k > 0 and cfg.rollout_weight > 0:
                br = max(1, int(cfg.wm_batch * cfg.rollout_fraction))
                s0 = cfg.window * ((T - k - 1) // cfg.window)  # chunk of the first rollout step: its bank
                with autocast(device, cfg.amp):
                    preds = runner.generate(z_in[:br, :T - k], k, banks[s0].select(slice(0, br))).float()
                loss_ro = F.mse_loss(preds, z[:br, T - k:])
                (cfg.rollout_weight * loss_ro).backward()
                stats["rollout_mse"] = float(loss_ro)
            stats["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0))
            opt.step()
            step += 1
            if step % cfg.log_every == 0:
                stats["it_per_s"] = cfg.log_every / max(time.time() - t_log, 1e-9)
                t_log = time.time()
                rd.log_metrics(kind="wm_train", step=step, hours=clock.now() / 3600, **stats)
                rd.write_status(stage=f"wm_{arm}", step=step, hours=clock.now() / 3600, budget_hours=cfg.wm_hours,
                                **stats)
                if step % (cfg.log_every * 10) == 0 or cfg.max_steps:
                    rd.print(f"{arm} step {step} " + " ".join(f"{k_}={v:.4g}" for k_, v in stats.items()))
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
        torch.save({"wm": wm.state_dict(), "wm_config": dataclasses.asdict(wm.cfg)}, rd.path / "wm_final.pt")
        rd.write_json("wm_metrics.json", final)
        write_wm_report(rd, final)
        rd.mark_done({"final": final})
        return final


def _span_mean(values: list, lo: int, hi: int) -> float | None:
    part = values[lo:hi]
    return float(np.mean(part)) if part else None


def compare_arms(cfg: Task21Config, results: dict, root: Path) -> dict:
    """Side-by-side table (short horizon <= window vs beyond the window) + PSNR/LPIPS-vs-horizon chart."""
    W = cfg.window
    out = {"window": W, "arms": {}}
    rows = []
    for arm, m in results.items():
        h = m["horizon"]
        spans = {"1-%d" % min(W, h): (0, W), "%d-%d" % (W + 1, h): (W, h), "all": (0, h)}
        entry = {}
        for name, (lo, hi) in spans.items():
            entry[name] = {key: _span_mean(m[key], lo, hi) for key in
                           ("model_psnr", "model_lpips", "model_latent_mse", "model_ssim", "copy_psnr", "copy_lpips")}
        entry["memory"] = {k: v for k, v in m.items() if k.startswith("eval_ltm")}
        out["arms"][arm] = entry
        for name, v in entry.items():
            if name == "memory" or v["model_psnr"] is None:
                continue
            rows.append(f"| {arm} | {name} | {v['model_psnr']:.2f} | {v['copy_psnr']:.2f} | {v['model_ssim']:.3f} | "
                        f"{v['model_lpips']:.3f} | {v['copy_lpips']:.3f} | {v['model_latent_mse']:.4f} |")
    lines = [f"# Task 2.1 - long-term memory A/B ({cfg.dataset})", "",
             f"Context {cfg.eval_context} real frames, then {cfg.eval_horizon} generated. Attention window {W} "
             f"frames; 'base' = Task 2 model (EMA memory), 'ltm' = + surprise-gated memory "
             f"({cfg.ltm_slots} slots). Horizon spans: inside the attention window vs beyond it.", "",
             "| arm | horizon | PSNR model | PSNR copy | SSIM model | LPIPS model | LPIPS copy | latent MSE |",
             "|---|---|---|---|---|---|---|---|", *rows]
    if "base" in out["arms"] and "ltm" in out["arms"]:
        lines += ["", "Difference ltm - base (PSNR: higher is better; LPIPS / latent MSE: lower is better):", ""]
        for name in out["arms"]["base"]:
            if name == "memory":
                continue
            b, l_ = out["arms"]["base"][name], out["arms"]["ltm"][name]
            if b["model_psnr"] is None or l_["model_psnr"] is None:
                continue
            lines.append(f"- {name}: PSNR {l_['model_psnr'] - b['model_psnr']:+.2f} dB, LPIPS "
                         f"{l_['model_lpips'] - b['model_lpips']:+.3f}, latent MSE "
                         f"{l_['model_latent_mse'] - b['model_latent_mse']:+.4f}")
        mem = out["arms"]["ltm"]["memory"]
        if mem:
            lines += ["", "Memory bank at the end of the evaluation rollouts: " +
                      ", ".join(f"{k.replace('eval_', '')}={v:.3g}" for k, v in mem.items())]
    lines += ["", "One evaluation run per arm (fixed seed): small differences can be noise; compare with the "
              "per-step curves in compare.png and the rollout GIFs in <arm>/samples/."]
    (root / "compare.md").write_text("\n".join(lines) + "\n")
    (root / "compare.json").write_text(json.dumps(out, indent=1))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for arm, m in results.items():
            ks = np.arange(1, m["horizon"] + 1)
            axes[0].plot(ks, m["model_psnr"], label=arm)
            axes[1].plot(ks, m["model_lpips"], label=arm)
        first = next(iter(results.values()))
        ks = np.arange(1, first["horizon"] + 1)
        axes[0].plot(ks, first["copy_psnr"], "k--", label="copy last frame")
        axes[1].plot(ks, first["copy_lpips"], "k--", label="copy last frame")
        for ax, name in zip(axes, ["PSNR (dB, higher is better)", "LPIPS (lower is better)"]):
            ax.axvline(W, color="grey", lw=0.8, ls=":")
            ax.set_xlabel("generated frame t+k")
            ax.set_title(name)
            ax.legend()
        fig.tight_layout()
        fig.savefig(root / "compare.png", dpi=120)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001 - the chart is optional
        print(f"compare.png not written: {exc}")
    return out


def run(cfg: Task21Config) -> dict:
    device = setup_torch(cfg.device)
    seed_everything(cfg.seed)
    root = Path(cfg.run_root) / cfg.dataset
    root.mkdir(parents=True, exist_ok=True)
    for sub in ("vae", *cfg.arms):
        (root / sub).mkdir(exist_ok=True)
        rotate_old_error(root / sub)
    rd = RunDir(root, config=cfg)
    with rd.capture_errors():
        rd.print(f"env: {describe_env()}")
        train, test, info = load_banks(cfg, device, rd)
        if cfg.dataset != "mmnist" and cfg.cross_cuts:
            for bank in (train, test):
                bank.seg = torch.zeros_like(bank.seg)
                bank._starts = {}
            rd.print("clips may cross scene cuts (cross_cuts=True)")
        rd.write_json("data_info.json", info)
        vae_path = default_vae_path(cfg)
        if vae_path.exists():
            rd.print(f"using the frozen VAE {vae_path}")
        else:
            rd.print(f"no VAE at {vae_path}: training one for this experiment")
            vae_path = train_vae(cfg, train, test, device, root / "vae")
        results = {arm: train_arm(cfg, arm, vae_path, train, test, device, root / arm) for arm in cfg.arms}
        out = compare_arms(cfg, results, root)
        rd.print((root / "compare.md").read_text())
        return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--data_path", default=None)
    ap.add_argument("--run_root", default=None)
    ap.add_argument("--vae_from", default=None, help="frozen VAE checkpoint (default: Task 2's for this dataset)")
    ap.add_argument("--wm_hours", type=float, default=None, help="training budget per arm")
    ap.add_argument("--arms", default=None, help="comma-separated subset of base,ltm")
    ap.add_argument("--ltm_slots", type=int, default=None)
    ap.add_argument("--eval_context", type=int, default=None)
    ap.add_argument("--eval_horizon", type=int, default=None)
    ap.add_argument("--start_frame", type=int, default=None)
    ap.add_argument("--gif_clips", type=int, default=None)
    ap.add_argument("--gif_fps", type=float, default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--override", default=None)
    args = ap.parse_args(argv)
    cfg = Task21Config()
    if args.smoke:
        cfg = smoke_config(cfg)
    for k in ["dataset", "data_path", "run_root", "vae_from", "wm_hours", "ltm_slots", "eval_context", "eval_horizon",
              "start_frame", "gif_clips", "gif_fps", "max_steps", "device"]:
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)
    if args.arms:
        cfg.arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    cfg.smoke = args.smoke
    cfg.force = args.force
    apply_overrides(cfg, parse_override(args.override))
    if args.smoke:
        fresh_smoke_run(Path(cfg.run_root) / cfg.dataset, root=cfg.run_root)
    return run(cfg)


if __name__ == "__main__":
    main()
