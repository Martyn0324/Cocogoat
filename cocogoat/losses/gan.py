"""Adversarial objectives.

RpGAN (relativistic pairing GAN, Jolicoeur-Martineau 2018; analysed by Sun et al.
2020 and used by R3GAN, Huang et al. NeurIPS 2024):

    L_D = E[ softplus( D(fake) - D(real) ) ]
    L_G = E[ softplus( D(real) - D(fake) ) ]

Each fake is compared with a paired real sample (same batch index, and the same
class when conditional), which removes the mode-dropping local minima of the
classic (non-relativistic) loss.  Zero-centred gradient penalties on real data (R1)
and on fake data (R2) give local convergence (R3GAN, Thm. 3.2), i.e.
"R3" = RpGAN + R1 + R2.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rpgan_d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return F.softplus(fake_logits.float() - real_logits.float()).mean()


def rpgan_g_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return F.softplus(real_logits.float() - fake_logits.float()).mean()


def hinge_d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return F.relu(1 - real_logits.float()).mean() + F.relu(1 + fake_logits.float()).mean()


def hinge_g_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.float().mean()


def gradient_penalty(logits: torch.Tensor, inputs: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    """Per-sample squared gradient norm ||d logits / d inputs||^2 (zero-centred GP).

    ``inputs`` must have ``requires_grad=True`` and have been used to compute
    ``logits``.  The returned tensor keeps the graph (create_graph=True) so it can be
    added to the discriminator loss.  Several input tensors (e.g. image + latent) are
    supported; their squared norms are summed.
    """
    single = torch.is_tensor(inputs)
    inputs = [inputs] if single else list(inputs)
    grads = torch.autograd.grad(outputs=logits.float().sum(), inputs=inputs, create_graph=True)
    total = 0.0
    for g in grads:
        total = total + g.float().square().flatten(1).sum(dim=1)
    return total


def relativistic_logit_gap(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> float:
    """Mean D(real) - D(fake): a health monitor (large & growing = D winning)."""
    return float((real_logits.float() - fake_logits.float()).mean().item())
