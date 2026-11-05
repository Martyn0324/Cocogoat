"""Cocogoat 2026: modern GANs, latent GANs and latent world models.

Subpackages
-----------
nn       building blocks (depthwise inverted bottlenecks, attention, modulation)
losses   adversarial (RpGAN + R1/R2), VAE and perceptual losses
data     dataset loaders, differentiable augmentation, pseudo-labels
metrics  FID / KID / precision-recall and video prediction metrics
train    EMA, schedules, checkpoint/resume, logging
models   GANs, VAE, world model, latent GAN, super-resolution
tasks    one entry point per task of the revival roadmap (and the Colab probe)
"""

__version__ = "2026.11.0"
