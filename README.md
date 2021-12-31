# Cocogoat - A Progressive DCGAN
Progressive DCGAN created on Pytorch, based on many different codes. The only originality here is the name and the DatasetCreator class...perhaps the AI to label the dataset, too, but I'm pretty sure that, though I didn't see anyone using this tactic to label a big dataset, I'm not the first one using it.

The name Cocogoat it's because I've been testing this code initially using a dataset of 10.000 images of Ganyu from Genshin Impact.

*OBS: If you read the reference papers and check this code, you'll notice that some resources are missing. The explanation for this is...simply because I don't know how to do it. That's specially the case for layer fading from NVidia's paper.*

## How it works

The file data_selector.py consists of the creation of the dataset(originally 100x100 Ganyu fanarts downloaded using [gallery-dl](https://github.com/mikf/gallery-dl)) and the creation of a multi-class neural network to classify images according to its quality and if they're not Ganyu fanarts. 1000 images are passed to a train dataset and manually labeled accordingly.

After training for quite some time, the neural network is used to label the remaining 9000 images. All undesired images can then be eliminated. The remaining ones can be passed into the ProGAN to generate images.


## References:
**Nathan Inkawhich. Pytorch's DCGAN Tutorial:** https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

**Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford and Xi Chen. Improved Techniques for Training GANs:** https://arxiv.org/pdf/1606.03498.pdf

**Tero Karras, Timo Aila, Samuli Laine and Jaakko Lehtinen. PROGRESSIVE GROWING OF GANS FOR IMPROVED QUALITY, STABILITY, AND VARIATION:** https://arxiv.org/pdf/1710.10196.pdf
