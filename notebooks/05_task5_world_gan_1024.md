# Task 5: latent world-GAN, one model for images and videos at 1024 px

**Goal.** A single system that makes a still image with the world model OFF, or a video with the world model ON. It combines the latent GAN (Task 3), the world model (Task 2) and super-resolution (Task 4). Data: Inter4K clips at 15 fps; latents at 256 px, frames upscaled to 1024.

**Idea.** Task 2's world model is deterministic. Trained with MSE, it predicts the *average* future, which blurs and drifts. Here the world model becomes a **stochastic generator**:
- noise tokens are fed in at every frame;
- a **temporal critic** judges whether a predicted next frame is a plausible continuation;
- **self-forcing** trains it on its own rollouts, so it learns to recover from its own errors.

## Architecture

```mermaid
flowchart LR
  P["prompt / CLIP ViT-L/14 embedding"] -.-> GI & GW
  subgraph Img["image mode (toggle OFF)"]
    Z["noise"] --> GI["G_img<br/>latent generator (Task 3)"] --> Z0["latent z_0 (32x32x16)"]
  end
  subgraph Vid["video mode (toggle ON)"]
    Z0 --> GW["G_wm: space-time transformer (Task 2)<br/>+ EMA memory + per-frame noise tokens<br/>+ adaLN(prompt)"]
    GW --> ZT["z_1 ... z_T"]
    ZT -->|autoregressive| GW
  end
  Z0 & ZT --> DEC["frozen VAE decoder"] --> F256["256 px frames"] --> SR["SR x4"] --> OUT["1024 px image / mp4"]
```

## Critics and training

```mermaid
flowchart TB
  subgraph Critics["three RpGAN + R1/R2 critics"]
    DL["D_lat: single latent<br/>real vs G_img output and G_wm predictions"]
    DI["D_img: Dec(real latent) vs Dec(G_img output)<br/>decoded vs decoded"]
    DT["D_T temporal: window z(t-3..t) + z(t+1)<br/>real continuation vs predicted"]
  end
  CTX["4 real context frames"] --> RO["self-forcing rollout, 4 steps"] --> DT & DL
  RO --> MSE["MSE anchor to true future<br/>weight 1.0 to 0.05 (critics take over)"]
  OT["OT penalty on G_img"] --> GSUM
  DL & DI & DT & MSE --> GSUM["G_img + G_wm update, EMA"]
```

Image batches and clip batches are mixed in one run, so the same generator serves both modes.

## Measurement

- Image mode: FID, KID, precision and recall at 1024 px, plus CLIP score.
- Video mode: per-horizon PSNR, SSIM and LPIPS against held-out clips, with the copy-last-frame baseline.
- mp4 and GIF videos.
