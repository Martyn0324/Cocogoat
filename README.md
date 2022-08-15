# Cocogoat - A Progressive DCGAN
Progressive DCGAN created on Pytorch, based on many different codes. The only originality here is the name and the DatasetCreator class and the prototype file...perhaps the AI to label the dataset, too, but I'm pretty sure that, though I didn't see anyone using this tactic to label a big dataset, I'm not the first one using it.

The name Cocogoat it's because I've been testing this code initially using a dataset of 10.000 images of Ganyu from Genshin Impact.

*OBS: If you read the reference papers and check this code, you'll notice that some resources are missing. The explanation for this is...simply because I don't know how to do it. That's especially the case for layer fading from NVidia's paper.*

## How it works

The file data_selector.py consists of the creation of the dataset(originally 100x100 Ganyu fanarts downloaded using [gallery-dl](https://github.com/mikf/gallery-dl)) and the creation of a multi-class neural network to classify images according to its quality and if they're not Ganyu fanarts. 1000 images are passed to a train dataset and manually labeled accordingly.

After training for a small time(the performance stabilizes at around 500), the neural network is used to label the remaining 9000 images. All undesired images can then be eliminated. The remaining ones are then resized so they can be passed into the ProGAN to generate images.

The ProGAN, in a nutshell, begins to train with low resolution images(4x4), and then progressively trains on higher resolution images(8x8 for level 1, 16x16 for level 2, 32x32 level 3 and so forth). This strategy makes it easier for the Neural Network to learn patterns as the weights are initially adjusted with simple data and only when the weights are properly calibrated they are adjusted with more complex data.


## Prototype

I've never seen any image generating GAN using LSTMs, so I decided to give it a chance. Perhaps LSTMs aren't used for images simply because it isn't worth it, but I'm still eager to give it a try and see it for myself.

Instead of passing sequences of data directly into the LSTM layer, I've decided to let the GAN try to generate and image from noise using the transposed 2D convs, and, at the end of the neural network, when the image has already been generated, pass it into a LSTM layer. The idea here is to make the Neural Network try to predict correctly what is the best pixel value to be added into an image based on what has been created so far.

Suppose that the final tranposed conv 2D layer returned certain image. We'll have an image with 3 channels(RGB) and height x width dimensions.
Each channel would be something like this:

![rubbish](https://user-images.githubusercontent.com/28028007/148013549-2ae06096-b728-4647-b757-0e4c4a9d5ac8.png)

That is, a grid, which could also be seen as a table, just like a DataFrame or Series when we work with Pandas, where we got X (height) and Y labels(width). Each value of X is assigned to a value of Y. That can be more easily visualized if you think about price prediction, for example. You got a table of values, with X being the open price of that asset that day, and Y being the close price. You want to use the open price to predict the close price. Then, you have a table with X values and Y values, convert them into a sequence dataset, pass into a LSTM and voilá.
With images, I'm simply considering that I have the same table, but, in a 8x8 image, I'll have a table with 8 columns and 8 values. Some of those columns can be considered my X to predict the Y, the remaining ones.

Considering that, we could pass an image like this into a LSTM

![illustration](https://user-images.githubusercontent.com/28028007/148013037-664707cf-75b9-45ca-8bd5-618d84139760.png)

So it could predict what is missing based on what it already has:

![scheme](https://user-images.githubusercontent.com/28028007/148014309-9f8b4bf2-864b-4238-a2fa-fb2a12905d63.png)

Unfortunately, though, I was stupid and didn't consider that creating sequences would make me get more data than an array (1,3,8,8) could support, so I have to pass the output of that LSTM to another tranposed conv 2D, in order to filter some data and if everything goes ok, I could get something like this:

![complete](https://user-images.githubusercontent.com/28028007/148014411-2ea06314-7300-4937-a63b-ad81d6b48a5d.png)

As the name suggests, this is just a prototype. Perhaps it would be more profitable to, instead of inserting the LSTM into the Generator, to create a second Generator with those LSTMs. I'll see how this goes with time.

*None of those images have been generated by this neural network. I'm just trying to explain my idea. Maybe someone could think about something better based on that*

That being said, I'm currently testing this hypothesis...and I'm not a researcher in this area, which means I might not dedicate that much time into this...


## Update

![frankenstein-its-alive](https://user-images.githubusercontent.com/28028007/170381156-0e9b4e50-de03-411e-886f-2f40596ce097.gif)

Some tips when testing:

* The greater the batch size, the better.
* Avoid using zeros in your weights. I thought about that when my weights in dimension 1 had size 3 and I wanted to concatenate it until it had size 50(50 isn't a multiple of 3). However, this will probably cause great harm to your model performance. Prefer using multiples of 3 in your channels as your model levels up.
* When you get to level 3(generating images 16x16), beware of model collapse. My default model had both G and D LR = 0.001 and I was using 500.000 epochs for each level. Maybe this number wasn't necessary at all for level 1(4x4) and 2(8x8), but it didn't see to do any harm. However, in level 3, after 450.000 epochs I noticed what could be a possible beginning of model colapse. In level 4(32x32), the model collapsed after 200.000(innacurate number, I used checkpoints that would plot the output image after 50.000 epochs).

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

## Update² + Prototype 2

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

So I wanted to make more tests and try to invent new things for Cocogoat, and I came with the idea of outputting the image generated in each level using a single generator, so I could analyze where the bigger problem is occurring.

Then it came to me the idea "if I'll be using a single discriminator for each level...why not use the backpropagation of those discriminators to teach the generator how it should modify its weights?"

Karras showed that a smaller image is easier to train and the *experience acquired* through that training can help generate bigger and more complex images. So, how about we implement this in a single architecture, with a some kind of feedback system?


![endocrigan](https://user-images.githubusercontent.com/28028007/184614025-1d6d300c-a7b7-48b0-9482-d212d004a045.png)

*Glorious Paint 3D design*

It kind of resembles the endocrine feedback system. The generator itself could be analogous to the pituitary gland, secreting its hormone, the output 1, which would be received by the discriminator 1, which would, then, secrete another hormone that would work as a feedback system to the pituitary gland, the generator. That feedback can also influence the hormone(output 2) that is being secreted to another gland(discriminator 2). All this network controls the way our methabolism express itself(the final output).

![image](https://user-images.githubusercontent.com/28028007/184615289-4e036bc7-4bb3-45c4-a8e4-eafa7231880b.png)

*The hormonal feedback system in the the female reproductive system(which is easier to see the mechanism, as it's a more complex system than the male's)*

However, this architecture indeed is quite heavy, as it'll be using many different networks(the discriminators) and with lower effectivity(fewer layers in order to avoid using too much memory), so it might be not effective at all...but I liked the idea anyway. I'll try testing it out but I won't be working too hard on it. Feel free to test it out...and perhaps try using a dataset that is bigger than 7,000 images...


*PS²: I still didn't test the LSTM structure. I still don't know exactly how to deal with those layers...perhaps after I study some NLP...*

## References:
**Nathan Inkawhich. Pytorch's DCGAN Tutorial:** https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

**Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford and Xi Chen. Improved Techniques for Training GANs:** https://arxiv.org/pdf/1606.03498.pdf

**Tero Karras, Timo Aila, Samuli Laine and Jaakko Lehtinen. PROGRESSIVE GROWING OF GANS FOR IMPROVED QUALITY, STABILITY, AND VARIATION:** https://arxiv.org/pdf/1710.10196.pdf

*Some classes from Didática Tech(PT-BR) about DCGAN: https://didatica.tech/*

**Florian Dedov(AKA Neural Nine):** https://www.youtube.com/watch?v=GFSiL6zEZF0 (Thanks for finally making me understand how these annoying, nitpicking LSTM layers work)


## Further Reading

**Andrew Brock, Jeff Donahue and Karen Simonyan. LARGE SCALE GAN TRAINING FOR HIGH FIDELITY NATURAL IMAGE SYNTHESIS:** https://arxiv.org/pdf/1809.11096v2.pdf - Also Known as BigGAN. Best GAN so far. The article summarize the story behind data generating algorithms and the improvements developed around them, especially around GANs. Eliminates the need for Progressive Grow by using new tricks.

**Prafulla Dhariwal and Alex Nichol. Diffusion Models Beat GANs on Image Synthesis:** https://arxiv.org/pdf/2105.05233.pdf - I didn't know about Diffusion Models, but it seems they weren't that interesting...until this paper. Maybe they'll be the best architeture for generating images. Will try to make one as soon as I get some time to read the paper and its code completely(thanks for using Pytorch instead of tensorflow, OpenAI...but I still hate you for neglecting gym and RL).

**Arash Vardat and Karsten Kreis. Improving Diffusion Models as an Alternative To GANs:** https://developer.nvidia.com/blog/improving-diffusion-models-as-an-alternative-to-gans-part-1/ ; https://developer.nvidia.com/blog/improving-diffusion-models-as-an-alternative-to-gans-part-2/
