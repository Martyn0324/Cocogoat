"""Generic RpGAN + R1/R2 training step shared by every GAN in the project.

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
    """Base class. Subclasses implement ``head`` and optionally ``encode``."""

    keys: tuple[str, ...] = ("img",)

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


def _finite(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all())


class AdversarialTrainer:
    def __init__(self, critics: list[CriticSlot], device: torch.device, amp: bool = True, reg_every: int = 4,
                 reg_fp32: bool = True):
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
        stats: dict[str, float] = {}
        for slot in self.critics:
            critic = slot.critic
            for p in slot.params:
                p.requires_grad_(True)
            with torch.no_grad(), autocast(self.device, self.amp):
                enc_r = critic.encode(real)
                enc_f = critic.encode(fake)
            enc_r = _detach_dict(enc_r, do_reg)
            enc_f = _detach_dict(enc_f, do_reg and slot.r2)
            use_amp = self.amp and not (do_reg and self.reg_fp32)
            with autocast(self.device, use_amp):
                lr_ = critic.head(enc_r, cond)
                lf_ = critic.head(enc_f, cond)
            loss = rpgan_d_loss(lr_, lf_)
            stats[f"{slot.name}/d_loss"] = float(loss.detach())
            stats[f"{slot.name}/gap"] = float((lr_.detach().float() - lf_.detach().float()).mean())
            if do_reg:
                r1 = gradient_penalty(lr_, list(enc_r.values())).mean()
                reg = r1
                stats[f"{slot.name}/r1"] = float(r1.detach())
                if slot.r2:
                    r2 = gradient_penalty(lf_, list(enc_f.values())).mean()
                    reg = reg + r2
                    stats[f"{slot.name}/r2"] = float(r2.detach())
                loss = loss + (slot.gamma / 2.0) * self.reg_every * reg
            if not _finite(loss):
                stats[f"{slot.name}/nonfinite"] = 1.0
                slot.opt.zero_grad(set_to_none=True)
                continue
            slot.opt.zero_grad(set_to_none=True)
            loss.backward()
            slot.opt.step()
        return stats

    # ----------------------------------------------------------------------------------
    def g_adversarial_loss(self, real: dict[str, torch.Tensor], fake: dict[str, torch.Tensor],
                           cond: torch.Tensor | None) -> tuple[torch.Tensor, dict[str, float]]:
        """Sum of weighted RpGAN generator losses over critics (critic params frozen)."""
        total = torch.zeros((), device=self.device)
        stats: dict[str, float] = {}
        for slot in self.critics:
            critic = slot.critic
            for p in slot.params:
                p.requires_grad_(False)
            with autocast(self.device, self.amp):
                with torch.no_grad():
                    lr_ = critic.head(critic.encode(real), cond)
                lf_ = critic.head(critic.encode(fake), cond)
            lg = rpgan_g_loss(lr_, lf_)
            stats[f"{slot.name}/g_loss"] = float(lg.detach())
            total = total + slot.g_weight * lg
        return total, stats


def make_adam(params, lr: float, beta2: float, beta1: float = 0.0, eps: float = 1e-8) -> torch.optim.Optimizer:
    return torch.optim.Adam(params, lr=lr, betas=(beta1, beta2), eps=eps)


ScheduleFn = Callable[[float], float]
