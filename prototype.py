import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout2d
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
import os

data = np.load('D:/Python/Projects/GANs/ganyu_64.npy')

data = data.astype('float32')

data = data/127.5 - 1 # range [-1,1]

data = torch.from_numpy(data)

data = data.view(data.shape[0], data.shape[3], data.shape[1], data.shape[2]) #(n_samples, n_channels, width, height)

print(data.shape) # The shape is (2092, 3, 64, 64)

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
        
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # input is Z, going into a convolution
        self.transconv1 = nn.ConvTranspose2d( 100, 64 * 8, 4, 1, 0, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(64 * 8)
        self.ReLU = nn.ReLU(True)
        # state size. (64*8) x 4 x 4
        self.transconv2 = nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(64 * 4)
        # state size. (64*4) x 8 x 8
        self.transconv3 = nn.ConvTranspose2d( 64 * 4, 64 * 2, 4, 2, 1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(64 * 2)
        # state size. (64*2) x 16 x 16
        self.transconv4 = nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(64)
        # state size. (64) x 32 x 32
        self.transconv5 = nn.ConvTranspose2d( 64, 3, 4, 2, 1, bias=False)
        self.lstm5 = nn.LSTM(input_size=64,hidden_size=64,num_layers=10, bias=False, batch_first=True)
        self.transconvA = nn.ConvTranspose2d(1, 1, 1, 1, (32,80), bias=False)
        self.transconvB = nn.ConvTranspose2d(3, 3, 1, 1, (32,80), bias=False)
        self.tanh = nn.Tanh()
    
    # Note: Consider using Batchnorm with momentum = 0.8

    def forward(self, input):
        x = self.transconv1(input)
        x = self.batchnorm1(x)
        x = self.ReLU(x)
        x = self.transconv2(x)
        x = self.batchnorm2(x)
        x = self.ReLU(x)
        x = self.transconv3(x)
        x = self.batchnorm3(x)
        x = self.transconv4(x)
        x = self.batchnorm4(x)
        x = self.transconv5(x)
        R, G, B = self._preprocess(x)
        outR, _ = self.lstm5(R)
        outG, _ = self.lstm5(G)
        outB, _ = self.lstm5(B)
        out_lstm = torch.stack((outR, outG, outB), dim=1) # (64-8, 3, 8, 64)
        # It's necessary to convert the output to (1, 3, 64, 64)
        # Possible options:
        # *Pass each color array to a transconv until it's possible to get a 64x64 monochromatic image(R, G, B). Then, stack all 3 channels to get RGB image;
        # *Use rectangular images and pass them to transconv, adjusting parameters to get 64x64 at the end.
        
        option1R = torch.reshape(outR, (1, 1, 128, 224))
        option1G = torch.reshape(outG, (1, 1, 128, 224))
        option1B = torch.reshape(outB, (1, 1, 128, 224))
        option1R = self.transconvA(option1R) # Returns (1, 1, 64, 64)
        option1G = self.transconvA(option1G)
        option1B = self.transconvA(option1B)
        outputR = self.tanh(option1R)
        outputG = self.tanh(option1G)
        outputB = self.tanh(option1B)

        output1 = torch.cat((outputR, outputG, outputB), dim=1) # Returns (1, 3, 64, 64)

        option2 = torch.reshape(out_lstm, (1, 3, 128, 224))
        option2 = self.transconvB(option2)
        output2 = self.tanh(option2)

        return output1, output2

    def _preprocess(self, RGB):
        R = RGB[:,0,:,:] # [n_samples, width, height, RGB] ----> Returns (n_samples, width, height)
        G = RGB[:,1,:,:]
        B = RGB[:,2,:,:]
        #print(R.shape)
        # É preciso converter para numpy array

        R = R.detach().cpu().numpy()
        G = G.detach().cpu().numpy()
        B = B.detach().cpu().numpy()
        #print(R.shape)
        
        outputR = []
        outputG = []
        outputB = []

        for i in range(8, 64): # Using last 8 pixels to get the next one. It's better to use 1 sample at time, otherwise the process can be too expensive.
            outputR.append(R[0, i-8:i,:])
            outputG.append(G[0, i-8:i,:])
            outputB.append(B[0, i-8:i,:]) # For each sample, get 1 array with (n_samples, n_timesteps, n_features)
        
        outputR = np.array(outputR, dtype='float32')
        outputG = np.array(outputG, dtype='float32')
        outputB = np.array(outputB, dtype='float32')

        # Returns (64-8, 8, 64) ---> (n_samples, timesteps, features) ---> ready for LSTM
        return torch.from_numpy(outputR).to(device), torch.from_numpy(outputG).to(device), torch.from_numpy(outputB).to(device)
      
      
# Create the generator
netG = Generator().to(device)

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.02.
netG.apply(weights_init)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1, bias=False)
        self.LeakyRelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.4, inplace=False)
        self.conv2 = nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(64 * 2)
        self.conv3 = nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(64 * 4)
        self.conv4 = nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(64 * 8)
        self.conv5 = nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input):
        x = self.conv1(input)
        x = torch.randn(x.size()).to(device) + x
        x = self.LeakyRelu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = torch.randn(x.size()).to(device) + x
        x = self.batchnorm2(x)
        x = self.LeakyRelu(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = torch.randn(x.size()).to(device) + x
        x = self.batchnorm3(x)
        x = self.LeakyRelu(x)
        x = self.dropout(x)
        x = self.conv4(x)
        x = torch.randn(x.size()).to(device) + x
        x = self.batchnorm4(x)
        x = self.LeakyRelu(x)
        x = self.dropout(x)
        x = self.conv5(x)
        x = self.sigmoid(x)

        return x

# Create the Discriminator
netD = Discriminator().to(device)

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Establish convention for real and fake labels during training --> Using 0.9 and 0.1 instead of 1 and 0 ---> One-sided label smoothing
# https://arxiv.org/pdf/1606.03498.pdf
real_label = 0.9
fake_label = 0.1 

# Setup Adam optimizers for both G and D ---> NVidia Progressive Grow: Same optimizer parameters Adam lr = 0.001, b1 = 0, b2 = 0.99
optimizerD = optim.Adam(netD.parameters(), lr=0.001, betas=(0, 0.99)) # Consider changing learning rate or even the optimizer
optimizerG = optim.Adam(netG.parameters(), lr=0.001, betas=(0, 0.99)) # If learning rate is too aggressive --> might fail to converge or might collapse

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []

print("Starting Training Loop...")

def train(data=None, epochs=1000,loss=nn.BCELoss(), optimizerD=optimizerD, optimizerG=optimizerG, save_point=100, checkpoint=5000, model_name='default_Cocogoat'):
    if os.path.isfile(f'discriminator_{model_name}.pth'):
        try:
            netD.load_state_dict(torch.load(f'discriminator_{model_name}.pth')) # Checkpoint
        
        except RuntimeError:
            previous_discriminator = torch.load(f'discriminator_{model_name}.pth') # Previous model
            current_discriminator = netD.state_dict()
            desired_shape = current_discriminator['conv1.weight'].size()
            previous_shape = previous_discriminator['conv1.weight'].size()
            zeros = torch.zeros(desired_shape[0], desired_shape[1], previous_shape[2], previous_shape[3]).to(device)
            weights = torch.add(zeros, previous_discriminator['conv1.weight'])
            weights = upsampler(weights)
            current_discriminator['conv1.weight'] = weights
    
    if os.path.isfile(f'generator_{model_name}.pth'):
        try:
            netG.load_state_dict(torch.load(f'generator_{model_name}.pth'))
        
        except RuntimeError:
            previous_generator = torch.load(f'generator_{model_name}.pth') # Previous model
            current_generator = netG.state_dict()
            desired_shape = current_generator['transconv1.weight'].size()
            previous_shape = previous_generator['transconv1.weight'].size()
            zeros = torch.zeros(desired_shape[0], previous_shape[1], desired_shape[2], desired_shape[3]).to(device)
            weights = torch.add(zeros, previous_generator['transconv1.weight'])
            weights = torch.cat((weights, weights, weights, weights), dim=1)
            weights = torch.cat((weights, weights, weights, weights), dim=1)
            zeros = torch.zeros(desired_shape[0], 2, desired_shape[2], desired_shape[3]).to(device)
            weights = torch.cat((weights, zeros), dim=1)
            current_generator['transconv1.weight'] = weights

    print("Weights Updated!")

    for epoch in range(epochs):
        netD.zero_grad()
        # Format batch
        b_size = 1
        real_cpu = data[np.random.randint(0, data.shape[0], size=b_size), :, :, :].to(device)
        label = torch.full((real_cpu.shape[0],), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1) # Generates tensor with size (b_size,)
        # Calculate loss on all-real batch
        errD_real = loss(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, 100, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = loss(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = loss(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Getting and saving best parameters:

        best_gen_loss, best_disc_loss = float('inf'), float('inf')
        generator_loss, discriminator_loss = errG.item(), errD.item()

        if generator_loss < best_gen_loss:
            best_gen_loss = generator_loss
            best_generator_parameters = netG.state_dict()
                
        if discriminator_loss < best_disc_loss:
            best_disc_loss = discriminator_loss
            best_discriminator_parameters = netD.state_dict()

        # Output training stats
        if epoch % checkpoint == 0:
            print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, epochs,#len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            

            with torch.no_grad():
                test_image = netG(noise).detach().cpu()

            test_image = (test_image+1)*0.5
            img_list.append(vutils.make_grid(fake.detach().cpu(), padding=2, normalize=True))
            
            torch.save(best_generator_parameters, f'generator_{model_name}.pth')
            torch.save(best_discriminator_parameters, f'discriminator_{model_name}.pth')
            print("Models saved!")

        if epoch % save_point == 0:
            saving_image = netG(noise).detach().cpu()
            saving_image = np.transpose(saving_image, (0,2,3,1))
            np.save(f'fake_images/{model_name}_{epoch}.npy', saving_image[0], allow_pickle=True)
            print(f'Fake image saved!')

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        
def plot_images(fake=True, n_samples=16, noise=None, X_train=None, model='default_Cocogoat'):
    if os.path.isfile(f'discriminator_{model}.pth'):
        netD.load_state_dict(torch.load(f'discriminator_{model}.pth'))
    if os.path.isfile(f'generator_{model}.pth'):
        netG.load_state_dict(torch.load(f'discriminator_{model}.pth'))

    if fake: # To show fake images:
        if noise is None:
            noise = torch.randn(n_samples, 100, 1, 1, device=device)
        with torch.no_grad():
            images = netG(noise)
    else: # To show real images:
        i = np.random.randint(0, X_train.shape[0], n_samples)
        images = X_train[i, :, :, :]

    # Plotting images:
    plt.figure(figsize=(10,10))
    for i in range(images.shape[0]):
        plt.subplot(4, 4, i+1)
        imagem = images[i, :, :, :]
        imagem = (imagem+1)*0.5
        imagem = imagem.cpu()
        imagem = imagem.view(imagem.shape[1], imagem.shape[2], imagem.shape[0])
        plt.imshow(imagem.numpy())
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# TEST SESSION

train(data=data,epochs=50001, checkpoint=5000, save_point=5000, model_name='default_Cocogoat')

optimizerD = optim.Adam(netD.parameters(), lr=0.0005)
optimizerG = optim.Adam(netG.parameters(), lr=0.0003)

print("****BEGINNING MODEL1 TRAINING***")

train(data=data, epochs=50001, checkpoint=5000, save_point=5000, model_name='Cocogoat1')
