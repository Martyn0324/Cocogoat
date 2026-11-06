# Cocogoat notebooks

Each notebook runs one task on Google Colab. Open it from this branch, pick the G4 runtime, and choose *Run all*. Each task has a short paper-style write-up next to its notebook. The write-up covers what the task is, a diagram of the architecture and training, and what is measured. Diagrams are Mermaid, which GitHub renders.

| Notebook | Write-up | Task |
|---|---|---|
| `00_probe.ipynb` | - | Checks the GPU, datasets and pretrained downloads |
| `01_task1_gan64.ipynb` | [01_task1_gan64.md](01_task1_gan64.md) | Pixel GAN at 64x64 with minimal mode collapse |
| `02_task2_vae_world_model.ipynb` | [02_task2_vae_world_model.md](02_task2_vae_world_model.md) | Per-frame VAE + latent world model (video prediction) |
| `02b_task2_1_ltm.ipynb` | [02b_task2_1_ltm.md](02b_task2_1_ltm.md) | Task 2.1: A/B test of a surprise-gated long-term memory |
| `03_task3_latent_gan.ipynb` | [03_task3_latent_gan.md](03_task3_latent_gan.md) | Latent GAN: `z -> G -> z' -> Decoder -> D` |
| `04_task4_hires_1024.ipynb` | [04_task4_hires_1024.md](04_task4_hires_1024.md) | Prompt-conditional 1024x1024 GAN in ~10 h |
| `05_task5_world_gan_1024.ipynb` | [05_task5_world_gan_1024.md](05_task5_world_gan_1024.md) | Latent world-GAN: images and videos at 1024 |

Shared pieces in every task:
- The R3GAN objective: relativistic pairing loss (RpGAN) + R1 + R2.
- Normalization-free depthwise inverted-bottleneck blocks.
- Attention with QK-norm.
- DiffAugment.
- An EMA generator.
- Schedules driven by the time budget.

The design document with the references is `docs/superpowers/specs/2026-11-05-cocogoat-revival-design.md`.

## Whose ideas are in here

This table separates the ideas that came from the author's own notes from those taken from the literature. The author's notes are the ViVAE design notes and the 2022 Cocogoat README. "Used" means the idea is in the code. It does **not** mean the idea was shown to improve the results: no ablations have been run yet, so no single component's contribution has been measured.

| Author's idea (source) | Status | Where |
|---|---|---|
| VAE strictly 2D / per-frame; all dynamics in a separate transition model (ViVAE) | **Used as stated** | Tasks 2, 3, 5 (`cocogoat/models/vae.py`) |
| PixelUnshuffle/PixelShuffle + 1x1 pointwise + depthwise convs instead of plain CNN (ViVAE) | **Used** (DC-AE residual shortcuts added) | `ResidualDown/Up`, `ConvNeXtBlock` |
| Self-attention only at the low-resolution bottleneck (ViVAE) | **Used** | VAE bottleneck at 8x8 |
| Transition model over a window of latent grids with axial (space / time) attention, Genie-style (ViVAE) | **Used** | `WorldModel` (spatial + causal temporal attention) |
| **Short-term EMA memory** (ViVAE) | **Used**: three decays 0.5 / 0.9 / 0.98, i.e. memories of about 2, 10 and 50 frames | `ema_memory` in `cocogoat/models/world.py` |
| **Long-term memory buffer gated by prediction error (surprise / salience)** (ViVAE) | **Implemented in Task 2.1** as an A/B experiment; not yet run | `cocogoat/models/ltm.py` |
| Frozen KL-regularised encoder means the transition loss needs no extra anti-collapse regulariser: pure MSE (ViVAE) | **Used as stated** | Task 2 world-model loss |
| Variance-gated backup reconstruction term (ViVAE) | Not implemented (no collapse was observed, so the trigger never applied) | - |
| Curriculum starting with Moving MNIST: train VAE, freeze, train transition model (ViVAE) | **Used** | Task 2 `mmnist` before `shelter` |
| PSNR/SSIM of decoded predictions and MSE over timesteps (ViVAE) | **Used** (plus LPIPS and a copy-last-frame baseline) | Task 2 metrics |
| FVD with VideoMAE-v2 features instead of I3D (ViVAE) | Not implemented | - |
| Latent-action conditioning, BAIR/KTH/CoppeliaSim stages (ViVAE / Arachne) | Out of scope for these five tasks | - |
| Relativistic discriminator (2022 README: "Use Relativistic Discriminator") | **Used** (its modern form, RpGAN, is the core loss of every task) | `cocogoat/losses/gan.py` |
| Several discriminators ("the more discriminators, the better") | **Used** | Task 3 (2 critics), Task 4 (2), Task 5 (3) |
| Labelling the dataset with a neural network instead of by hand (2022 `data_selector`) | **Used in spirit**: CLIP embeddings + k-means pseudo-classes, unsupervised | Tasks 1, 3 |
| BatchNorm adds noise in G; pixel-norm hurt (2022 notes) | Consistent: no BatchNorm or pixel-norm anywhere | all |
| Residual blocks (2022 notes, from SRGAN/ESRGAN) | **Used** | all |
| Multi-scale outputs from G with one D per scale (2022 "Optimized Cocogoat") | Not used (R3GAN stabilises a single-scale GAN without it) | - |
| Dropout in D (2022 notes) | Not used (DiffAugment + R1/R2 are used instead) | - |

On the memory system: the ViVAE notes describe **two** memories: a short-term EMA and a long-term buffer gated by prediction error. A third, very-short-term memory is not in the notes this project received. If it is in another document, it was not seen. In the code, the role of a very-short-term memory is played by the causal temporal attention window: the world model attends exactly to the last `max_frames` latents (16 during training), on top of the EMA channels. The long-term surprise-gated buffer is now tested in Task 2.1 against an otherwise identical model.
