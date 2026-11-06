# Task 4: prompt-conditional 1024x1024 GAN in about 10 hours

**Goal.** This task generates 1024x1024 images (FFHQ faces) from text prompts after about 10 GPU-hours, trained from scratch. The only pretrained networks allowed are embedding and classification models, never generative ones.

**Idea.**
- **The base GAN at 256 px is the generative model.** A x4 super-resolution GAN, trained from scratch on real 1024 images, only adds high-frequency detail.
- **Prompting without captions** follows LAFITE. Training conditions on the CLIP *image* embedding, perturbed with noise. At inference the CLIP *text* embedding takes its place, shifted by the modality gap.
- **Fast convergence** comes from a second critic, which works on frozen DINOv2 features (Projected GAN idea). The metrics use Inception, a different network, so the critic cannot simply game the score.

## Architecture

```mermaid
flowchart LR
  subgraph Cond["Conditioning (frozen CLIP ViT-L/14)"]
    IMG[training image] --> CI["image embedding + noise (xi=0.25)"]
    TXT[prompt at inference] --> CT["text embedding + gap shift"]
  end
  CI -. train .-> G
  CT -. inference .-> G
  Z["z"] --> G["Base G (Task 1 design scaled to 256)"] --> F256[256 px fake]
  F256 --> DP["pixel critic<br/>R1/R2 on pixels"]
  F256 --> DIN["frozen DINOv2-S, layers 3/6/9/12"] --> PH["projected heads<br/>R1/R2 on features"]
  F256 --> SR["SR x4 GAN<br/>(from scratch)"] --> F1024[1024 px image]
```

## Optimisation

```mermaid
flowchart TB
  subgraph Base["Stage base (~7 h)"]
    A["RpGAN on pixel D + projected D<br/>gamma_pix 30 to 3, gamma_proj 1 to 0.1<br/>DiffAugment, EMA G, cosine schedules"]
  end
  subgraph SRs["Stage sr (~2.5 h)"]
    B["pairs: degraded 256 crop to real 1024 crop<br/>L1 + LPIPS + 0.05 RpGAN (+R1/R2)"]
  end
  Base --> SRs --> E["Stage eval"]
```

## Measurement

- FID and KID at 299 px.
- **Patch-KID** on native-resolution crops: does the 1024 detail look real?
- Precision and recall.
- CLIP score for text prompts, using CLIP B/32 (a different model from the conditioning one).
- Prompt grids and full-resolution PNGs.
