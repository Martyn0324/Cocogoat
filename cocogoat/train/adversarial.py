"""Generic RpGAN + R1/R2 training step shared by every GAN in the project.

``cond`` may be a single tensor or a dict {critic name: condition tensor} when critics see
batches of different composition.

A *critic* splits the discriminator into
    encode(sample) -> dict of tensors   (augmentation and/or frozen feature networks)
    head(encoded, cond) -> logits       (the trainable part)
R1 / R2 are computed with respect to the head inputs.  For a pixel discriminator,
``encode`` is just DiffAugment, so R1/R2 act on the (augmented) image; for a
projected discriminator (frozen DINOv2 features), the penalty acts on the frozen
features and no second-order gradient has to flow through the backbone.

Several critics (e.g. image + latent discriminators of the latent GAN) are trained
jointly, each with its own optimizer, gamma and generator-loss weight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn

from cocogoat.losses import gradient_penalty, rpgan_d_loss, rpgan_g_loss
from cocogoat.train.utils import autocast


class Critic(nn.Module):
    """Base class. Subclasses implement ``head`` and optionally ``encode``.

    ``fast`` is toggled by the trainer: True for ordinary passes (a torch.compile'd copy of
    the discriminator may be used), False for R1/R2 passes, which need double backward and
    therefore run the eager module.
    """

    keys: tuple[str, ...] = ("img",)
    fast: bool = False

    def encode(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: sample[k] for k in self.keys}

    def head(self, enc: dict[str, torch.Tensor], cond: torch.Tensor | None) -> torch.Tensor:
        raise NotImplementedError

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


@dataclass
class CriticSlot:
    name: str
    critic: Critic
    opt: torch.optim.Optimizer
    g_weight: float = 1.0
    gamma: float = 1.0           # current R1/R2 weight (updated by the schedule)
    r2: bool = True              # R3GAN uses both R1 and R2
    params: list = field(default_factory=list)


def _detach_dict(d: dict[str, torch.Tensor], requires_grad: bool) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in d.items():
        v = v.detach()
        if requires_grad:
            v = v.float().requires_grad_(True)
        out[k] = v
    return out


class AdversarialTrainer:
    """Stats are returned as detached 0-dim tensors (no GPU sync); convert them at log time."""

    def __init__(self, critics: list[CriticSlot], device: torch.device, amp: bool = True, reg_every: int = 4,
                 reg_fp32: bool = False):
        self.critics = critics
        self.device = device
        self.amp = amp
        self.reg_every = max(1, reg_every)
        self.reg_fp32 = reg_fp32
        for slot in critics:
            slot.params = slot.critic.trainable_parameters()

    # ----------------------------------------------------------------------------------
    def d_step(self, real: dict[str, torch.Tensor], fake: dict[str, torch.Tensor], cond: torch.Tensor | None,
               step: int) -> dict[str, float]:
        do_reg = step % self.reg_every == 0
        stats: dict[str, torch.Tensor] = {}
        for slot in self.critics:
            critic = slot.critic
            critic.fast = not do_reg
            for p in slot.params:
                p.requires_grad_(True)
            with torch.no_grad(), autocast(self.device, self.amp):
                enc_r = critic.encode(real)
                enc_f = critic.encode(fake)
            enc_r = _detach_dict(enc_r, do_reg)
            enc_f = _detach_dict(enc_f, do_reg and slot.r2)
            use_amp = self.amp and not (do_reg and self.reg_fp32)
            cc = cond.get(slot.name) if isinstance(cond, dict) else cond
            with autocast(self.device, use_amp):
                lr_ = critic.head(enc_r, cc)
                lf_ = critic.head(enc_f, cc)
            loss = rpgan_d_loss(lr_, lf_)
            stats[f"{slot.name}/d_loss"] = loss.detach()
            stats[f"{slot.name}/gap"] = (lr_.detach().float() - lf_.detach().float()).mean()
            if do_reg:
                r1 = gradient_penalty(lr_, list(enc_r.values())).mean()
                reg = r1
                stats[f"{slot.name}/r1"] = r1.detach()
                if slot.r2:
                    r2 = gradient_penalty(lf_, list(enc_f.values())).mean()
                    reg = reg + r2
                    stats[f"{slot.name}/r2"] = r2.detach()
                loss = loss + (slot.gamma / 2.0) * self.reg_every * reg
            slot.opt.zero_grad(set_to_none=True)
            loss.backward()
            slot.opt.step()
            critic.fast = False
        return stats

    # ----------------------------------------------------------------------------------
    def g_adversarial_loss(self, real: dict[str, torch.Tensor], fake: dict[str, torch.Tensor],
                           cond: torch.Tensor | None) -> tuple[torch.Tensor, dict[str, float]]:
        """Sum of weighted RpGAN generator losses over critics (critic params frozen)."""
        total = torch.zeros((), device=self.device)
        stats: dict[str, torch.Tensor] = {}
        for slot in self.critics:
            critic = slot.critic
            critic.fast = True
            for p in slot.params:
                p.requires_grad_(False)
            cc = cond.get(slot.name) if isinstance(cond, dict) else cond
            with autocast(self.device, self.amp):
                with torch.no_grad():
                    lr_ = critic.head(critic.encode(real), cc)
                lf_ = critic.head(critic.encode(fake), cc)
            critic.fast = False
            lg = rpgan_g_loss(lr_, lf_)
            stats[f"{slot.name}/g_loss"] = lg.detach()
            total = total + slot.g_weight * lg
        return total, stats


def make_adam(params, lr: float, beta2: float, beta1: float = 0.0, eps: float = 1e-8) -> torch.optim.Optimizer:
    return torch.optim.Adam(params, lr=lr, betas=(beta1, beta2), eps=eps)


def to_floats(stats: dict) -> dict:
    """Convert a stats dict of 0-dim tensors / numbers to Python floats (one sync)."""
    out = {}
    for k, v in stats.items():
        out[k] = float(v.detach().float()) if torch.is_tensor(v) else v
    return out


def maybe_compile(module: nn.Module, enabled: bool) -> nn.Module | None:
    """torch.compile'd view of ``module`` (sharing its parameters), or None."""
    if not enabled or not torch.cuda.is_available():
        return None
    try:
        return torch.compile(module, dynamic=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] torch.compile unavailable: {exc}")
        return None


ScheduleFn = Callable[[float], float]
