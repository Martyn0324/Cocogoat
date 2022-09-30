# Even OpenAI's Guided Diffusion needs a SuperResolution Model

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cudnn.benchmark = True

class ConvBlock(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, stride=1):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding=kernel_size//2)
        self.batchnorm1 = nn.BatchNorm2d(output_channels)
        self.PRelu = nn.PReLU()
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size, stride, padding=kernel_size//2)
        self.batchnorm2 = nn.BatchNorm2d(output_channels)

    
    def forward(self, input):

        residual = input

        x = self.conv1(input)
        x = self.batchnorm1(x)
        x = self.PRelu(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)

        output = residual + x

        return output


class SubPixelConvBlock(nn.Module):

    def __init__(self, n_channels=64, kernel_size=3, scaling_factor=2):

        super(SubPixelConvBlock, self).__init__()

        self.conv = nn.Conv2d(n_channels, n_channels*(scaling_factor**2), kernel_size, 1, padding=kernel_size//2)
        self.pixel_shuffle = nn.PixelShuffle(scaling_factor)
        self.PRelu = nn.PReLU()

    def forward(self, input):

        x = self.conv(input)
        x = self.pixel_shuffle(x)
        output = self.PRelu(x)

        return output


class SRGAN(nn.Module):

    def __init__(self):
        super(SRGAN, self).__init__()

        self.convin = nn.Conv2d(3, 64, 9, 1, 9//2)
        self.PRelu = nn.PReLU()

        self.resblock1 = ConvBlock(64, 64, 3, 1)
        self.resblock2 = ConvBlock(64, 64, 3, 1)
        self.resblock3 = ConvBlock(64, 64, 3, 1)
        self.resblock4 = ConvBlock(64, 64, 3, 1)
        self.resblock5 = ConvBlock(64, 64, 3, 1)
        self.resblock6 = ConvBlock(64, 64, 3, 1)
        self.resblock7 = ConvBlock(64, 64, 3, 1)
        self.resblock8 = ConvBlock(64, 64, 3, 1)
        self.resblock9 = ConvBlock(64, 64, 3, 1)
        self.resblock10 = ConvBlock(64, 64, 3, 1)
        self.resblock11 = ConvBlock(64, 64, 3, 1)
        self.resblock12 = ConvBlock(64, 64, 3, 1)
        self.resblock13 = ConvBlock(64, 64, 3, 1)
        self.resblock14 = ConvBlock(64, 64, 3, 1)
        self.resblock15 = ConvBlock(64, 64, 3, 1)
        self.resblock16 = ConvBlock(64, 64, 3, 1)

        self.conv = nn.Conv2d(64, 64, 3, 1, 3//2)
        self.batchnorm = nn.BatchNorm2d(64)

        self.pixel_shuffle1 = SubPixelConvBlock(64, 3, 2)
        self.pixel_shuffle2 = SubPixelConvBlock(64, 3, 2)

        self.convout = nn.Conv2d(64, 3, 9, 1, 9//2)
        self.tanh = nn.Tanh()

    def forward(self, input):

        x = self.convin(input)
        x = self.PRelu(x)
        residual = x

        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.resblock6(x)
        x = self.resblock7(x)
        x = self.resblock8(x)
        x = self.resblock9(x)
        x = self.resblock10(x)
        x = self.resblock11(x)
        x = self.resblock12(x)
        x = self.resblock13(x)
        x = self.resblock14(x)
        x = self.resblock15(x)
        x = self.resblock16(x)

        x = self.conv(x)
        x = self.batchnorm(x)

        x = x + residual

        x = self.pixel_shuffle1(x)
        x = self.pixel_shuffle2(x)

        output = self.convout(x)

        return output


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)

        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, 1)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)
        self.batchnorm4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, 3, 1, 1)
        self.batchnorm5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 512, 3, 2, 1)
        self.batchnorm6 = nn.BatchNorm2d(512)
        self.conv7 = nn.Conv2d(512, 512, 3, 1, 1)
        self.batchnorm7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 1024, 3, 2, 1)
        self.batchnorm8 = nn.BatchNorm2d(1024)
        self.neuron9 = nn.Linear(1024, 1024)
        self.neuron10 = nn.Linear(1024, 1)

        self.LeakyRelu = nn.LeakyReLU(0.2)

    def forward(self, input):

        x = self.conv1(input)
        x = self.LeakyRelu(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.LeakyRelu(x)
        
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.LeakyRelu(x)

        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.LeakyRelu(x)

        x = self.conv5(x)
        x = self.batchnorm5(x)
        x = self.LeakyRelu(x)

        x = self.conv6(x)
        x = self.batchnorm6(x)
        x = self.LeakyRelu(x)

        x = self.conv7(x)
        x = self.batchnorm7(x)
        x = self.LeakyRelu(x)

        x = self.conv8(x)
        x = self.batchnorm8(x)
        x = self.LeakyRelu(x)

        x = self.neuron9(x.view(x.size(0), -1))
        x = self.LeakyRelu(x)
        output = self.neuron10(x)

        return output

class TruncatedVGG19(nn.Module):

    def __init__(self, i, j):

        super(TruncatedVGG19, self).__init__()

        self.imagenet_mean = torch.FloatTensor([0.485, 0.456, 0.406]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        self.imagenet_std_cuda = torch.FloatTensor([0.229, 0.224, 0.225]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)

        vgg19 = models.vgg19(pretrained=True)

        maxpool_counter = 0
        conv_counter = 0
        truncate_at = 0

        for layer in vgg19.features.children():

            truncate_at += 1

            if isinstance(layer, nn.Conv2d):
                conv_counter += 1

            if isinstance(layer, nn.MaxPool2d):
                maxpool_counter += 1
                conv_counter = 0


            if maxpool_counter == i - 1 and conv_counter == j:

                break

        assert maxpool_counter == i - 1 and conv_counter == j

        self.truncated_vgg19 = nn.Sequential(*list(vgg19.features.children())[:truncate_at + 1])

    def forward(self, input):

        # Since VGG19 was trained on ImageNet, it's necessary to normalize our data using ImageNet stats

        if input.ndimension() == 3:
            input = (input - self.imagenet_mean) / self.imagenet_std_cuda
        elif input.ndimensio() == 4:
            input = (input - self.imagenet_mean) / self.imagenet_std_cuda

        output = self.truncated_vgg19(input)

        return output


def train(data=None, batch_size=16, epochs=1000, checkpoint=100, lr=1e-4, beta=1e-3, grad_clip=None, save_path=None):

    Gen = SRGAN().to(device)
    Disc = Discriminator().to(device)

    if save_path is None:

        start_epoch = 0

        optimizerG = torch.optim.Adam(params=Gen.parameters(), lr=lr)

        optimizerD = torch.optim.Adam(params=Disc.parameters(), lr=lr)

    else:

        params = torch.load(f"{save_path}/SRGAN_Checkpoint.json")

        start_epoch = params['Epoch'] + 1
        Gen.load_state_dict(params['Generator_Params'])
        Disc.load_state_dict(params['Discriminator_Params'])
        optimizerG = params['OptimizerG']
        optimizerD = params['OptimizerD']

        print(f"\nLoaded checkpoint from epoch {start_epoch}")

    TrunkVGG = TruncatedVGG19(5, 4).to(device)
    TrunkVGG.eval()

    content_loss = nn.MSELoss()
    disc_loss = nn.BCEWithLogitsLoss()

    content_loss = content_loss.to(device)
    disc_loss = disc_loss.to(device)

    Gen.train()
    Disc.train()


    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, pin_memory=True)

    for epoch in range(start_epoch, epochs):

        for item, (lr_images, hr_images) in enumerate(dataloader):

            if epoch == int(epochs/2):

                for param_group in optimizerG:
                    param_group['lr'] = param_group['lr'] * 0.1
                for param_group in optimizerD:
                    param_group['lr'] = param_group['lr'] * 0.1

            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)

            
            # Generator Update

            sr_images = Gen(lr_images) # (Batch_size, Channels, H, W) [-1, 1]
            
            # Calculate VGG feature maps for the loss functions

            sr_images_vgg = TrunkVGG(sr_images)
            hr_images_vgg = TrunkVGG(hr_images).detach() # Detached since they're targets, constants

            # Discriminate Super Resolved Images

            disc_output = Disc(sr_images) # (Batch_size,)
            labels = torch.ones_like(disc_output)

            # Calculating losses

            content_cost = content_loss(sr_images_vgg, hr_images_vgg)
            adversarial_loss = disc_loss(disc_output, labels)
            perceptual_loss = content_cost + beta * adversarial_loss

            # Backpropagation

            Gen.zero_grad()
            perceptual_loss.backward()

            if grad_clip is not None:
                nn.utils.clip_grad.clip_grad_value_(optimizerG, grad_clip)

            optimizerG.step()


            # Discriminator Update

            disc_out_original = Disc(hr_images)
            disc_out_fake = Disc(sr_images.detach())

            labels = torch.ones_like(disc_out_original)
            error_original = disc_loss(disc_out_original, labels)

            labels = torch.zeros_like(disc_out_fake)
            error_fake = disc_loss(disc_out_fake, labels)

            disc_error = error_original + error_fake

            Disc.zero_grad()
            disc_error.backward()

            if grad_clip is not None:
                nn.utils.clip_grad.clip_grad_value_(optimizerD, grad_clip)

            optimizerD.step()

            best_disc_loss = float("inf")
            best_gen_loss = float("inf")

            if perceptual_loss.item() < best_gen_loss:
                best_gen_loss = perceptual_loss.item()
                best_generator_params = Gen.state_dict()

            if disc_error.item() < best_disc_loss:
                best_disc_loss = disc_error.item()
                best_disc_params = Disc.state_dict()

            if item % checkpoint == 0 or epoch == epochs-1:

                print(f"{epoch}|{epochs}\t Iteration: {item}\nBest Discriminator Loss: {best_disc_loss}\nBest Generator Loss: {best_gen_loss}")

                torch.save({
                    'Epoch': epoch,
                    'Generator_Params': best_generator_params,
                    'OptimizerG': optimizerG,
                    'Discriminator_Params': best_disc_params,
                    'OptimizerD': optimizerD
                }, f"{save_path}/SRGAN_Checkpoint.json")

                #torch.save(best_generator_params, "SRGAN_Generator.pth")
                #torch.save(best_disc_params, "SRGAN_Discriminator.pth")

            if epoch % checkpoint == 0 or epoch == epochs-1:

                with torch.no_grad():

                    saving_images = Gen(lr_images)

                saving_images = saving_images.view(saving_images.shape[0], saving_images.shape[2], saving_images.shape[3], saving_images.shape[1])
                saving_images = saving_images.cpu().numpy()
                saving_images = (saving_images+1.0)*0.5
                
                fig, ax = plt.subplots(3,9)
                for x in range(ax.shape[0]):
                    for y in range(ax.shape[1]):
                        ax[x,y].axis("off")
                
                for i in ax.shape[0]:
                    ax[i,0].imshow(lr_images[i])
                    ax[i,0].set_title("Low Resolution")
                    ax[i,1].imshow(saving_images[i])
                    ax[i,1].set_title("SRGAN")
                    ax[i,2].imshow(hr_images[i])
                    ax[i,2].set_title("High Resolution")
                
                plt.show()


            del lr_images, hr_images, hr_images_vgg, sr_images, sr_images_vgg, disc_error, disc_out_fake, disc_out_original, disc_output, labels



def evaluate(data=None, model_path=None):
    Gen = SRGAN().to(device)
    params = torch.load(f"{model_path}/SRGAN_Checkpoint.json")

    Gen.load_state_dict(params['Generator_Params'])

    Gen.eval()

    images = Gen(data)

    images = images.view(images.shape[0], images.shape[2], images.shape[3], images.shape[1])
    images = images.cpu().numpy()
    images = (images+1.0)*0.5

    return images
