# Task 1: a modern 64x64 GAN with minimal mode collapse

**Goal.** This task trains an image GAN from scratch on the Genshin fan-art set (45,935 images, 9 characters, strongly imbalanced) at 64x64. It does so without the mode collapse of the 2022 DCGAN/ProGAN attempts.

**Idea in one paragraph.** The data has no labels. CLIP ViT-B/32 embeds every image and k-means splits it into 100 pseudo-classes. The GAN is conditioned on these clusters, so the generator is asked for every cluster in proportion to the data, and a dropped mode is directly visible to the projection discriminator. The objective is R3GAN: the relativistic pairing loss plus R1 and R2 gradient penalties. It is known to remove the mode-dropping minima of the classic losses. Both networks are normalization-free stacks of depthwise inverted bottlenecks, with attention at low resolution.

## Architecture

```mermaid
flowchart LR
  subgraph Labels["Pseudo-labels (once, frozen CLIP)"]
    X0[Real images] --> CLIP[CLIP ViT-B/32] --> KM[k-means, k=100] --> Y[cluster id c]
  end
  subgraph G["Generator"]
    Z["z ~ N(0,I) 128"] --> MAP["Mapping MLP (2 layers)"]
    Y --> EMB[class embedding] --> MAP
    MAP --> W["w (512)"]
    C4["learned 4x4 const"] --> B4["Blocks 4x4 to 64x64<br/>channels 512/512/384/256/128<br/>depthwise inverted bottleneck + FiLM(w)<br/>self-attention at 16x16"] --> RGB[64x64 RGB]
    W -. FiLM .-> B4
  end
  subgraph D["Discriminator"]
    AUG["DiffAugment<br/>color / translation / cutout"] --> DB["Blocks 64 to 4<br/>attention at 16 and 8<br/>minibatch-std"] --> OUT["logit = psi(h) + &lt;emb(c), h&gt;<br/>(projection)"]
  end
  RGB --> AUG
  X0 --> AUG
```

## Optimisation

```mermaid
flowchart TB
  R[real x, label c] --> DR["D(x,c)"]
  F["fake G(z,c)"] --> DF["D(G(z,c),c)"]
  DR & DF --> L["RpGAN: softplus(D(fake) - D(real))"]
  L --> UD["update D (+ lazy R1 on reals, R2 on fakes, every 8 steps, weight gamma)"]
  L --> UG["update G (reversed pair)"]
  UG --> EMA["EMA copy of G, used for samples and metrics"]
  S["Cosine schedules over the time budget:<br/>lr 2e-4 to 5e-5, beta2 0.9 to 0.99, gamma 2 to 0.2,<br/>EMA half-life 10 to 300 kimg, augment p 0.2 to 0.6"] -.-> UD & UG & EMA
```

## Measurement

- FID and KID on Inception pool3 features: 10k samples during training, 50k at the end.
- Precision and recall (k=3). Recall is the direct mode-coverage number: falling recall means collapse.
- Per-cluster sample grids, so a missing or repeated cluster is visible.

## Results so far

The first full run took 3.5 h and 30k steps, reaching **FID 111.2**. The samples are recognisable, with some tendency towards collapse. About 50% of the images show two of the nine characters. Details are in `runs/task1/report.md` on Drive.
