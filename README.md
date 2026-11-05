# Cocogoat

## 2026 revival: modern GANs, latent GANs and latent world models

Four years after the first Cocogoat (a progressive DCGAN), this branch rebuilds the project with what GAN research
has learned since: the R3GAN objective (relativistic pairing loss + R1 + R2), normalization-free depthwise-separable
blocks, attention with QK-norm, latent-space generation, projected discriminators, and adversarial latent world
models. Everything is trained from scratch; the only pretrained networks used are *embedding / classification*
models (CLIP, DINOv2, InceptionV3 for metrics, VGG for LPIPS). No pretrained generative model is used anywhere.

| # | Task | Data | Highlights | Notebook |
|---|------|------|------------|----------|
| 1 | GAN 64x64 with minimal mode collapse | Genshin fan-art (45,935 images) | RpGAN+R1+R2, depthwise inverted bottlenecks, attention, CLIP k-means pseudo-classes (cluster-conditioned GAN), DiffAugment, FID/KID/precision/recall | `notebooks/01_task1_gan64.ipynb` |
| 2 | VAE + world model (next-latent prediction) | Colored Moving MNIST, Shelter music video (64x64) | per-frame beta-VAE (PixelUnshuffle + DC-AE shortcuts, EQ-VAE term, adversarial fine-tuning), space-time transformer with EMA memory, rollout metrics vs copy-last baseline | `notebooks/02_task2_vae_world_model.ipynb` |
| 3 | Latent GAN `z -> G -> z' -> Decoder -> D` | Genshin fan-art (64x64), CLIP pseudo-classes | generator in latent space with OT penalty, image critic on *decoded* reals vs decoded fakes, latent critic on `z'` | `notebooks/03_task3_latent_gan.ipynb` |
| 4 | Prompt-conditional 1024x1024 in ~10 h | FFHQ-1024 | CLIP-conditioned base GAN at 256 (language-free LAFITE training), pixel + DINOv2-projected critics, x4 SR GAN from scratch, KID + patch-KID + CLIP score | `notebooks/04_task4_hires_1024.ipynb` |
| 5 | Latent world-GAN at 1024 (image + video) | Inter4K (1,000 4K clips) | latent image GAN + stochastic adversarial world model (self-forced rollouts, temporal critic), world-model toggle, SR to 1024, mp4 videos | `notebooks/05_task5_world_gan_1024.ipynb` |

**How to run (Google Colab, G4 / RTX PRO 6000):** open a notebook from this branch on Colab, choose the G4 runtime
and *Run all*. Each notebook clones the branch, runs a short GPU smoke test, then trains with the time budgets
(hours) and output folder set in its settings cell (cell 5); Tasks 4 and 5 can also use a data subset there for
quick tests. Checkpoints, metrics (`metrics.jsonl`, `final_metrics.json`, `report.md`), sample grids and logs go to
`MyDrive/Cocogoat/runs/<task>/`; after a disconnect, *Run all* resumes from the last checkpoint. Run
`notebooks/00_probe.ipynb` first on a new machine to check the GPU, datasets and pretrained downloads.

**Code layout:** `cocogoat/nn` (blocks), `cocogoat/losses` (RpGAN, R1/R2, LPIPS), `cocogoat/data` (datasets,
DiffAugment, pseudo-labels), `cocogoat/metrics` (FID/KID/PR, PSNR/SSIM/LPIPS), `cocogoat/models` (GAN, VAE, world
model, latent GAN, projected critic, SR), `cocogoat/train` (adversarial trainer, EMA, schedules, checkpoints),
`cocogoat/tasks` (one entry point per task), `tests/` (CPU unit tests: `python -m pytest tests`). The design
document is `docs/superpowers/specs/2026-11-05-cocogoat-revival-design.md`.

Results are added below as runs finish.

---

# Cocogoat - A Progressive DCGAN (2022)
Progressive DCGAN created on Pytorch, based on many different codes. The only originality here is the name and the DatasetCreator class and the prototype file...perhaps the AI to label the dataset, too, but I'm pretty sure that, though I didn't see anyone using this tactic to label a big dataset, I'm not the first one using it.

The name Cocogoat it's because I've been testing this code initially using a dataset of 10.000 images of Ganyu from Genshin Impact.

## UPDATE: The Behavior of the DCGAN architecture -> Clinical Trials - TODO

* Vanilla Configurations:
	* Non-Conditional
	* Generator and Discriminator with similar layers and parameters
	* Initialization with low value parameters -> Normal (0, 0.02)
	* Discriminator 1-D Classifier (Sigmoid)
	* DropOut (0.25) in Discriminator --> **IMPORTANT: Applies Regularization**
	* BatchNormalization Layers
	* Learning rate = 2e-4 for both Gen and Discriminator
		* Beta1 = 0.5 ; Beta2 = 0.99
	* Loss = Binary Cross Entropy
* Observations: Decent, stable training.
	* **NOTE:** The first optimization steps *catapults* each model weights towards a direction of values. If this step is too aggressive, it causes the models to diverge. **CONSIDER EXPERIMENTS WITH ROBUST GANs TO CHECK THIS BEHAVIOR**
       * This also justifies why it's mandatory to initialize weights with such low values. Also why not to zero the beta1 from Adam (which would cause the optimization to be too aggressive)

<img width="1180" height="471" alt="DCGAN Explosive Start" src="https://github.com/user-attachments/assets/cb752324-995c-4f33-ad65-aa2184cdc625" />


## Creating dataset

The file `DataAugmentator.ipynb` has a model and some functions to provide Data Augmentation. The model is used to basically select Ganyu's face on each image. In the DataAugmentation function, the same model is used in evaluation mode to extract Ganyu's face and resize that box so it has the same size as the other images in the dataset. Other functions are also included.

The files data_selector.py(or Pytorch version) consists of a Neural Network used to filter the images in your dataset(originally 100x100 Ganyu fanarts downloaded using [gallery-dl](https://github.com/mikf/gallery-dl)) and the creation of a multi-class neural network to classify images according to its quality and if they're not Ganyu fanarts. 3000 images are passed to a train dataset and manually labeled accordingly.

After training for some time(the performance stabilizes at around 100 epochs with learning_rate = 1e-3), the neural network is used to label the remaining 9000 images. All undesired images can then be eliminated.

## **UPDATE:** Optimized Cocogoat, Diffusion Model and Minimum Entropy

The original idea for Cocogoat wasn't that effective at all, as it takes too much time to get ok 16x16 images, and the model tends to collapse at 32x32 images when the dataset is too small.

It seems that the current state-of-the-art GAN, StyleGAN, uses the Progressive Grow idea in a much smarter way: the Generator must create 64x64 images, but it outputs all images generated on the way. It means that the generator actually produces images 4x4, 8x8, 16x16, 32x32 and 64x64. This makes it possible to take advantage of the progressive growing process while also having to spend less time training...at a lesser risk os mode collapse.
Unfortunately, you'll also have to use multiple discriminators, which means more computational power. Because of that, I wasn't able to produce images bigger than 32x32. However, using multiple Discriminators also increase the GAN stability, which ends up being helpful.

The code can be seem in the Optimized Cocogoat notebook.

I've also managed to finally make a functional Diffusion Model, which is a Denoising Diffusion Probabilistic Model, and it managed to generate quite interesting outputs. They have nothing to do with the dataset, but hey, that's some beautiful colors. The code can be found in the Diffusion notebook.

It's interesting to note that Failure modes of Diffusion models are quite similar to GAN's. In Diffusion Models, some images can't be properly produced in the sampling mode due to insufficient training, thus, the model generates black or white squares, without any image. It's also possible to note some "mode collapse" in some cases. Using higher resolution images also tends to make the model more unstable and it'll require more epochs to produce proper results *consistently*.

I've also figured out that using a big dataset will provide better results, so I'm working on models to filter a big dataset composed of Genshin characters fanarts. In order to avoid having to label dozens of thousands of images, I'm making some experiments on pretraining a model based on [Minimum Entropy](https://github.com/xidianai/MinEnt/tree/main/MinEnt), with supervised fine-tuning plus self-learning. It's been taking a while to learn how to properly apply it, but the Minimum Entropy has been going fine. Now, the problem is just with the fine-tuning, as it compromises severely the self-learning process.
The model also might have some elements of few-shot, as the complete dataset has 48,000 images and the fine-tuning process consists of 1,000 images.

## ~~The Cocogoat Model -- **Work in Progress**~~

The ProGAN, in a nutshell, begins to train with low resolution images(4x4), and then progressively trains on higher resolution images(8x8 for level 1, 16x16 for level 2, 32x32 level 3 and so forth). This strategy makes it easier for the Neural Network to learn patterns as the weights are initially adjusted with simple data and only when the weights are properly calibrated they are adjusted with more complex data.

However, while testing this, I've noticed that, even with the transfer learning, there's a point where the model simply collapses, usually with images at around 32x32 and 64x64. **This did NOT happen when dealing with the Fashion MNIST dataset**, which might indicate that sufficiently large datasets(+- 100,000 images) might contribute to the GAN performance. This is where the Emboxer, the neural network for Data Augmentation, comes in.

I also want to try other alternatives. I've been making some tests using the DCGAN architecture(since it's the only architecture that really worked for me so far...though it's inappropriate even for a proper dataset like the Animeface dataset). Relativistic Discriminators seem a silly idea, but in my tests they appear to rarely get out of control.
The SRGAN also came with the idea of using Residual Blocks in GANs, and this appear to help with vanishing gradients and to have a better performance. ESRGAN discovered that BatchNormalization layers in the Generator causes some random noise in the generated images(though, in my experience, I've seen that they're essential to keep the discriminator under control).
In my tests, I also discovered that dropout layers might help with the Discriminator performance, but they also induce random noise in the Generator output.

Real-ESRGAN also uses a UNet discriminator, something that is also used by OpenAI's Guided Diffusion and provides a better feedback to the Generator, since it classifies both the image and the pixels.

Karras, in Progressive Grow, used a "pixel-wise normalization" technique. In my tests, this normalization technique contributed immensely to...collapse my model miserably. So I'll just stick to normal things.

## Update

![frankenstein-its-alive](https://user-images.githubusercontent.com/28028007/170381156-0e9b4e50-de03-411e-886f-2f40596ce097.gif)

Some tips when testing:

* The greater the batch size, the better.
* Avoid using zeros in your weights. I thought about that when my weights in dimension 1 had size 3 and I wanted to concatenate it until it had size 50(50 isn't a multiple of 3). However, this will probably cause great harm to your model performance. Prefer using multiples of 3 in your channels as your model levels up.
* When you get to level 3(generating images 16x16), beware of model collapse. My default model had both G and D LR = 0.001 and I was using 500.000 epochs for each level. Maybe this number wasn't necessary at all for level 1(4x4) and 2(8x8), but it didn't see to do any harm. However, in level 3, after 450.000 epochs I noticed what could be a possible beginning of model colapse. In level 4(32x32), the model collapsed after 200.000(innacurate number, I used checkpoints that would plot the output image after 50.000 epochs).
* The bigger your dataset, the better.
* **Use Relativistic Discriminator**
* **Use Duelling Discriminators...the more discriminators, the better**

### Examples of Model Collapse

![image](https://user-images.githubusercontent.com/28028007/170381860-bd62f402-abfd-47b3-8b34-c5adb519df2c.png)

-> Low diversity of outputs, images that are nothing more than some colored blur and has 9 evident squares(that are still present in normal/good outputs, but are way more subtle)

![image](https://user-images.githubusercontent.com/28028007/170382154-4ea72cf6-ceb9-406f-9686-8e7839247a33.png)

*Actually 200.000 + 80.000 epochs.*

In my experience, your model is probably doing fine if your **Discriminator loss is around 1~1.5** and your **Generator loss is around 1.2~1.8**

![image](https://user-images.githubusercontent.com/28028007/170382476-a04f6907-a7c6-486f-8a9c-815f7dc40399.png)
*Default model level 3 after 450.000 epochs, what may be its optimal point. In 500.000 epochs, its performance is compromised*

![image](https://user-images.githubusercontent.com/28028007/170382660-5180b995-0e2b-424e-b1e7-3ff77b4b15b3.png)


*PS: I still didn't test the Prototype Architecture... I didn't even reach level 6(128x128) with default Cocogoat. Maybe soon...who knows...*

## Update²

Unfortunately, my "original" idea for Cocogoat didn't work out at all. The output images didn't get better than those 2 images shown above, without any clear image, only blurred figures. There's also the 9 squares problem. Obviously, my generated images can't be compared to the images attached to the papers in the links below.

This might be a problem with my dataset, as I'm using just under 7000 images to train my model from scratch, while those papers actually use datasets like MNIST(100,000 images) and ImageNet(more than **14 MILLION** images).

I've actually tested Cocogoat with MNIST before actually testing it with Ganyu fanarts. This was the result:

![image](https://user-images.githubusercontent.com/28028007/184609232-a78b8599-4fb5-4945-8377-00ff7c27db1d.png)

*Cocogoat level 4(32x32) after 450,000 epochs*

![image](https://user-images.githubusercontent.com/28028007/184609482-acf26c75-1970-4f0c-a6ed-6685650abe23.png)

*Cocogoat level 5(64x64) after 50,000 epochs.*

![image](https://user-images.githubusercontent.com/28028007/184609652-ef51f9e1-4321-4a65-bf46-94d1844c795e.png)

*Cocogoat level 6(128x128) right after starting its training, after having the weights from level 5 applied. The training was stopped right after the print, as I got a bit excited and wanted to try on Ganyu fanarts as soon as possible.*


Meanwhile, when I tested Cocogoat on Ganyu fanarts, this was perhaps one of the few good results I got:

![image](https://user-images.githubusercontent.com/28028007/184610555-01d07919-4dbc-45fe-bd12-1dcf8c457ddc.png)

The quality of the generated images tend to fall greatly after level 4...and there's also those annoying squares.

However, it also seems that DCGAN architecture isn't quite good to generate images, according to what I've seen in [these](https://github.com/jayleicn/animeGAN) [projects](https://github.com/pavitrakumar78/Anime-Face-GAN-Keras)


## References:
**Nathan Inkawhich. Pytorch's DCGAN Tutorial:** https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

**Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford and Xi Chen. Improved Techniques for Training GANs:** https://arxiv.org/pdf/1606.03498.pdf

**Tero Karras, Timo Aila, Samuli Laine and Jaakko Lehtinen. PROGRESSIVE GROWING OF GANS FOR IMPROVED QUALITY, STABILITY, AND VARIATION:** https://arxiv.org/pdf/1710.10196.pdf

*Some classes from Didática Tech(PT-BR) about DCGAN: https://didatica.tech/*

**Florian Dedov(AKA Neural Nine):** https://www.youtube.com/watch?v=GFSiL6zEZF0 (Thanks for finally making me understand how these annoying, nitpicking LSTM layers work)

**sgrvinod**: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution/tree/8a2ecba423760bfed791a92748bdc58de7fed918

**Alexia Jolicoeur-Martineau. The relativistic discriminator: a key element missing
from standard GAN:** https://arxiv.org/pdf/1807.00734.pdf - Probably a so special element that seems to even discard the previous techniques.

**Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Chen Change Loy, Yu Qiao, Xiaoou Tang. ESRGAN: Enhanced Super-Resolution
Generative Adversarial Networks:** https://arxiv.org/pdf/1809.00219.pdf

**Xintao Wang, Liangbin Xie, Chao Dong, Ying Shan. Real-ESRGAN: Training Real-World Blind Super-Resolution
with Pure Synthetic Data:** https://arxiv.org/pdf/2107.10833.pdf

**Jiaheng Wei, Minghao Liu, Jiahao Luo, Andrew Zhu, James Davis, Yang Liu. DuelGAN: A Duel Between Two Discriminators Stabilizes the GAN Training:** https://arxiv.org/pdf/2101.07524.pdf

**Shuo Li, Fang Liu, Zehua Hao, Licheng Jiao, Xu Liu, Yuwei Guo. MinEnt: Minimum entropy for self-supervised representation learning:** https://www.sciencedirect.com/science/article/abs/pii/S0031320323000651

## Further Reading

**Andrew Brock, Jeff Donahue and Karen Simonyan. LARGE SCALE GAN TRAINING FOR HIGH FIDELITY NATURAL IMAGE SYNTHESIS:** https://arxiv.org/pdf/1809.11096v2.pdf - Also Known as BigGAN. Best GAN so far. The article summarize the story behind data generating algorithms and the improvements developed around them, especially around GANs. Eliminates the need for Progressive Grow by using new tricks.

**Prafulla Dhariwal and Alex Nichol. Diffusion Models Beat GANs on Image Synthesis:** https://arxiv.org/pdf/2105.05233.pdf - I didn't know about Diffusion Models, but it seems they weren't that interesting...until this paper. Maybe they'll be the best architeture for generating images. Will try to make one as soon as I get some time to read the paper and its code completely(thanks for using Pytorch instead of tensorflow, OpenAI...but I still hate you for neglecting gym and RL).

**Arash Vardat and Karsten Kreis. Improving Diffusion Models as an Alternative To GANs:** https://developer.nvidia.com/blog/improving-diffusion-models-as-an-alternative-to-gans-part-1/ ; https://developer.nvidia.com/blog/improving-diffusion-models-as-an-alternative-to-gans-part-2/

**Alammar, Jay. The Illustrated Stable Diffusion:** https://jalammar.github.io/illustrated-stable-diffusion/

**Singh, Vaibhav. An In-Depth Guide to Denoising Diffusion Probabilistic Models - From Theory to Implementation:** https://learnopencv.com/denoising-diffusion-probabilistic-models/#Mathematical-Details-Of-The-Forward-Diffusion-Process

**Weng, Lilian. (Jul 2021). What are diffusion models? Lil’Log:** https://lilianweng.github.io/posts/2021-07-11-diffusion-models/.

**Nick Huang, Aaron Gokaslan, Volodymyr Kuleshov, James Tompkin. The GAN is dead; Long Live the GAN!:** https://neurips.cc/virtual/2024/poster/95345
