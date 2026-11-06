# Task 3: latent GAN, `z -> G -> z' -> Decoder -> fake image -> D`

**Goal.** This task builds a class-conditional GAN whose generator works in the latent space of a frozen VAE. The data is the same Genshin set and the same CLIP pseudo-classes as Task 1, so its numbers are directly comparable.

**Idea.** Two failure modes are typical of latent GANs, and both are designed out:
1. **The discriminator wins on VAE blur.** If D compares decoded fakes with *raw* reals, it can always spot the decoder's blur, which G cannot fix. Here the image critic compares `Dec(z')` with *decoded reals* `Dec(E(x))`.
2. **G drifts off the latent manifold.** Off-manifold latents decode to look-alike images. A second, latent critic judges `z'` against real posterior latents.

## Architecture

```mermaid
flowchart LR
  subgraph Pre["Stage 1 (frozen after training)"]
    X[real image] --> ENC[VAE encoder, as in Task 2] --> ZR["z_real 8x8x16 (normalised)"]
  end
  subgraph Gen["Latent generator"]
    N["spatial noise z 8x8x16"] --> BLK["isotropic stack at 8x8:<br/>depthwise inverted bottleneck<br/>+ self-attention, FiLM(w)"]
    G0["global noise + class c to mapping MLP to w"] -.-> BLK
    BLK --> RES["z' = z + G(z, c)"]
  end
  RES --> DL["latent critic D_lat<br/>z' vs z_real"]
  ZR --> DL
  RES --> DEC[frozen decoder] --> FI["fake = Dec(z')"]
  ZR --> DEC2[frozen decoder] --> RI["Dec(z_real)"]
  FI & RI --> DA[DiffAugment] --> DI["image critic D_img<br/>(Task 1 D, projection on c)"]
```

## Optimisation

```mermaid
flowchart TB
  DI["D_img: RpGAN + R1/R2 (gamma 2 to 0.2)"] --> LG
  DL["D_lat: RpGAN + R1/R2 (gamma 1 to 0.1)"] --> LG
  OT["OT penalty lambda * ||z' - z||^2<br/>(lambda 0.1 to 0.01)"] --> LG
  LG["G loss = adv_img + 1.0 * adv_lat + OT"] --> EMA[EMA G]
```

The OT penalty biases G towards the shortest map from noise to data. Its source is Adversarial Flow Models (Lin et al.). The idea is that this makes the adversarial game better posed.

## Measurement

FID, KID, precision and recall of `Dec(G(z))` are measured against:
- the real images, which is comparable with Task 1;
- the decoded reals;
- the VAE's own rFID, the floor that any latent model inherits.
