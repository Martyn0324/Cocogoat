"""Wrappers around the *allowed* pretrained networks (embedding / classification models).

* CLIP (OpenAI, via ``transformers``): image/text embeddings for pseudo-labels,
  prompt conditioning (language-free LAFITE-style training) and CLIP-score.
* DINOv2 (via ``transformers``): self-supervised features for the projected
  ("vision-aided") discriminator used by the high-resolution GAN.
* InceptionV3 (FID weights, via ``pytorch-fid``) lives in ``cocogoat.metrics``.

No pretrained *generative* model (VAE, diffusion, GAN, upsampler) is used anywhere.
All inputs are NCHW float tensors in [-1, 1].
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _normalize(x_m11: torch.Tensor, size: int, mean, std) -> torch.Tensor:
    x = (x_m11.float().clamp(-1, 1) + 1) / 2
    if x.shape[-1] != size or x.shape[-2] != size:
        x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False, antialias=x.shape[-1] > size)
    m = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    return (x - m) / s


def _as_tensor(out):
    if torch.is_tensor(out):
        return out
    for attr in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
        if hasattr(out, attr) and getattr(out, attr) is not None:
            return getattr(out, attr)
    raise TypeError(f"cannot extract tensor from {type(out)}")


class CLIPEncoder(nn.Module):
    """Frozen CLIP image/text encoder returning L2-normalised embeddings."""

    def __init__(self, name: str = "openai/clip-vit-base-patch32", device: str | torch.device = "cuda"):
        super().__init__()
        from transformers import CLIPModel, CLIPTokenizer

        self.model = CLIPModel.from_pretrained(name).to(device).eval().requires_grad_(False)
        self.tokenizer = CLIPTokenizer.from_pretrained(name)
        self.size = self.model.config.vision_config.image_size
        self.dim = self.model.config.projection_dim
        self.device = torch.device(device)

    @torch.no_grad()
    def encode_images(self, x_m11: torch.Tensor, batch: int = 256) -> torch.Tensor:
        outs = []
        for i in range(0, x_m11.shape[0], batch):
            px = _normalize(x_m11[i:i + batch].to(self.device), self.size, CLIP_MEAN, CLIP_STD)
            outs.append(F.normalize(_as_tensor(self.model.get_image_features(pixel_values=px)).float(), dim=-1))
        return torch.cat(outs)

    @torch.no_grad()
    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        tok = self.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        return F.normalize(_as_tensor(self.model.get_text_features(**tok)).float(), dim=-1)


class DINOv2Features(nn.Module):
    """Frozen DINOv2 backbone returning token features from a few intermediate blocks.

    Gradients flow *through* it (to the generator); its weights stay frozen.
    """

    def __init__(self, name: str = "facebook/dinov2-small", layers: tuple[int, ...] = (3, 6, 9, 12),
                 size: int = 224, device: str | torch.device = "cuda"):
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(name).to(device).eval().requires_grad_(False)
        self.layers = layers
        self.size = size
        self.dim = self.model.config.hidden_size
        self.patch = self.model.config.patch_size

    def forward(self, x_m11: torch.Tensor) -> list[torch.Tensor]:
        px = _normalize(x_m11, self.size, IMAGENET_MEAN, IMAGENET_STD)
        out = self.model(pixel_values=px, output_hidden_states=True)
        hs = out.hidden_states
        g = self.size // self.patch
        feats = []
        for li in self.layers:
            t = hs[li][:, 1:1 + g * g]  # drop CLS (+ register tokens if any)
            feats.append(t.transpose(1, 2).reshape(t.shape[0], t.shape[2], g, g))
        return feats
