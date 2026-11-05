# Cocogoat 2026 revival - design

Status: living document (updated as runs come back). Date: 2026-11-05.

## 1. Goal and constraints

Five rungs, each building on the previous one:

1. Functional 64x64 GAN with minimal mode collapse, using current mechanisms (depthwise convs, attention, ...).
2. VAE + world model: an auxiliary model predicts the latent of the next video frame (64x64).
3. Latent GAN with the exact data flow `z -> Generator -> z' -> Decoder -> fake image -> Discriminator`
   (optionally a model on `z'`), class-conditional, 64x64.
4. Conditional (class or prompt) GAN producing 1024x1024 images after ~8-12 h of training on one GPU, with reasonable
   KID, recognizable objects/textures and minimal mode collapse. Auxiliary SR/VAE allowed; main generative process = GAN.
5. Latent GAN + world model + >=1024x1024 frames, one model for single images and videos (world-model toggle).

Rules honoured: only pretrained **embedding / text / classification** networks (CLIP, DINOv2, InceptionV3, VGG for
LPIPS) - never a pretrained generative model (no SD-VAE, no diffusion, no Real-ESRGAN weights), no distillation.
No human evaluation mid-training: every run scores itself (FID/KID/precision/recall/PSNR/LPIPS) and writes reports.
Compute: one Colab G4 (RTX PRO 6000 Blackwell, 96 GB), ~8.7 compute units/h, 688 units => ~79 GPU-hours total.

## 2. Execution model

* Code lives in this repo (branch `claude/cocogoat-revival-glfnmz`). Notebooks in `notebooks/` are opened in Colab
  straight from GitHub; each cell clones/updates the branch and runs `python -m cocogoat.tasks.<task>`.
* Every run writes to `MyDrive/Cocogoat/runs/<task>/`: `config.json`, `log.txt`, `metrics.jsonl`, `status.json`,
  `ckpt/latest.pt` (resume), `samples/*.png`, `report.md`, `final_metrics.json`, `error.txt` on crashes, `DONE`.
  Training is **time-budgeted** (all schedules are functions of elapsed/budget), resumes after disconnects.
* The author (Claude) reads those files through the Google Drive connector, decides whether a stage is concluded
  or needs changes, and pushes fixes. The user only presses *Run all*.
* `notebooks/00_probe.ipynb` first measures the GPU, inspects the user's datasets, lists candidate Hugging Face
  datasets and checks every pretrained-weight download, so budgets/batch sizes are set from facts.

## 3. Shared mechanisms (all tasks)

| Mechanism | Where | Why |
|---|---|---|
| RpGAN + R1 + R2 ("R3GAN", Huang et al. NeurIPS 2024) | every adversarial loss | relativistic pairing removes mode-dropping minima; R1+R2 give local convergence |
| Cosine schedules of lr, Adam beta2, gamma, EMA half-life, augmentation p over the time budget | every GAN | R3GAN training schedule |
| Normalization-free inverted bottleneck: 1x1 expand -> depthwise kxk -> 1x1 project, Fix-up (zero-init last layer) | G, D, latent nets | efficient convs; no batch statistics interfering with gradient penalties |
| Multi-head self-attention with pre-RMSNorm + QK-norm at low resolution | G, D, VAE bottleneck, world model | global context (SAGAN idea) with a stable logit scale; explicit softmax in D so R1/R2 double-backward works |
| PixelUnshuffle / PixelShuffle with DC-AE style residual space<->channel shortcuts | VAE, D downsampling | lossless resampling, easy optimisation at high compression |
| FiLM / adaLN conditioning (zero-init) and projection discriminator | conditional models | strong, stable conditioning |
| Cluster conditioning (CLIP k-means pseudo-classes) | unlabelled datasets | every mode is requested from G in proportion to the data, so dropping one is visible to D |
| DiffAugment (colour/translation/cutout) on real+fake | D inputs | prevents D over-fitting on 10k-100k image datasets |
| EMA generator, bf16 autocast (fp32 for regularisation passes), TF32 | all | standard quality/speed |
| Metrics: FID, KID, precision/recall (k=3) on Inception pool3; PSNR/SSIM/LPIPS per horizon for video | all | recall = direct mode-coverage measure |

## 4. Per-task design

### Task 1 - GAN 64x64 (`cocogoat/tasks/task1_gan64.py`)
* Data: user's `Complete_dataset.npy` (45,935 Genshin fan-arts, 100x100 float64 -> uint8 cache -> 64x64).
  Labels: 100 CLIP (ViT-B/32) k-means clusters. Fallback: Tiny-ImageNet (200 real classes).
* G: z(128) + class emb -> 2-layer mapping -> w(512); 4x4 -> 64x64 with channels 512/512/384/256/128, 2 blocks per
  stage, attention at 16x16; FiLM(w) in every block. D mirrors it, attention at 16 and 8, minibatch-std, projection.
* Schedules over 3.5 h: lr 2e-4->5e-5, beta2 0.9->0.99, gamma 2->0.2 (log-cosine), EMA half-life 10->300 kimg,
  augmentation p 0.2->0.6. Batch 256, lazy R1/R2 every 4 steps.
* Done when: final FID/KID computed on 50k samples, recall reported, sample grids saved, no collapse
  (recall not trending to 0; per-cluster grids diverse).

### Task 2 - VAE + world model (64x64 video)
* Data: (a) Colored Moving MNIST (synthetic, the first rung of the ViVAE curriculum, deterministic dynamics) and
  (b) the user's `Shelterdataset.npz` (anime music video frames).
* VAE (per-frame, 2D, as in the ViVAE notes): ConvNeXt-style depthwise blocks, PixelUnshuffle downsampling with
  residual space-to-channel shortcuts, MHSA at the 8x8 bottleneck; f=8, 16 latent channels (8x8x16 latents).
  Loss: L1 + LPIPS + beta*KL (warm-up) + EQ-VAE scale-equivariance term + RpGAN patch adversarial term after warm-up.
* World model: factorised space-time transformer over the 8x8 latent tokens (spatial attention within a frame, causal
  temporal attention across frames), multi-scale EMA memory channels (decays 0.5/0.9/0.98 - the "short-term EMA
  memory" of the ViVAE design), residual next-latent prediction `z_{t+1} = z_t + f(z_<=t)`, MSE loss on normalised
  latents (encoder frozen => no extra anti-collapse regulariser needed), context-noise augmentation and a short
  self-rollout loss against exposure bias.
* Metrics: VAE rPSNR/rSSIM/LPIPS/rFID; world model per-horizon latent MSE and decoded PSNR/SSIM/LPIPS for
  autoregressive rollouts, always compared with the copy-last-frame baseline.

### Task 3 - Latent GAN 64x64 (`z -> G -> z' -> Decoder -> D`)
* VAE as in Task 2, trained on the Task 1 image dataset, then **frozen**; latents normalised per channel.
* G works *in latent space*: input noise has the latent's shape (8x8x16); isotropic stack of depthwise blocks +
  attention with adaLN(class); residual output `z' = z + G(z, c)`.
* Critics: (i) image D on decoded fakes vs **decoded reals** `Dec(E(x))` - the D cannot win by spotting VAE blur,
  which the generator cannot fix (a likely cause of the old latent-GAN collapses); (ii) latent D on `z'` vs posterior
  latents, keeping `z'` on the decoder's training manifold (the old failure: off-manifold latents decoding to look-alike
  images). Both RpGAN + R1 + R2.
* Optional OT regulariser `lambda * ||z' - z||^2` (Adversarial Flow Models, Lin et al. 2025/ICML 2026): biases G to the
  unique optimal-transport map, which the paper reports stabilises adversarial training.
* Compared directly with Task 1 on the same data (FID/KID/precision/recall), plus rFID of the VAE as the floor.

### Task 4 - conditional 1024x1024 in 8-12 h
* Base GAN at 256x256 (Task 1 architecture scaled) + super-resolution GAN x4 (trained from scratch on real 1024
  images; L1 + LPIPS + RpGAN). Main generative process = base GAN; SR only adds high-frequency detail.
* Discriminators: pixel D (R1/R2 on pixels) + projected D on frozen DINOv2-S features (R1/R2 on features) for fast
  convergence (Projected GAN / vision-aided GAN idea). Evaluated with Inception (different backbone, so the
  projected D cannot "game" the metric).
* Conditioning: prompt via CLIP. Training uses CLIP *image* embeddings of the training images (with noise
  perturbation), inference uses CLIP *text* embeddings (language-free training, LAFITE). Dataset chosen after the
  probe (FFHQ-1024 is the default candidate).
* Metrics: FID/KID at 299 px, patch-KID on native 1024 crops, CLIP score for prompts, precision/recall.

### Task 5 - latent world-GAN at >=1024
* Per-frame VAE at 256 (f=8, 32x32x16 latents) + latent GAN generator with a world-model toggle:
  `G(noise, cond, context)`; context = none (image mode) or previous latents + EMA memory (video mode); trained
  with image batches and clip batches in one run (context dropout). Critics: latent image D, latent temporal D
  (judges (context, next) pairs), decoded-frame D. Frames upscaled by the Task 4 SR model (or one trained on the
  Task 5 domain) to 1024. Dataset chosen after the probe.

## 5. Risks and mitigations
* Blind GPU tuning (no GPU here): probe first; every task has a `--smoke` mode run first in the notebook; errors
  land in `error.txt`, which I read through Drive.
* Gamma too high/low: logged per step (R1, R2, logit gap); log-cosine schedule; can be overridden with `--override`.
* Colab disconnects: atomic checkpoints to Drive every ~20 min; time budget counts training time only.
* Network: datasets are downloaded on Colab (this container cannot reach Hugging Face).
