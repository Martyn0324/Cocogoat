import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from torchsummary import summary
import os
from PIL import Image
from datasetcreator import DatasetCreator # Custom class

# Beginning to work with 4x4 images data

ganyu4x4 = DatasetCreator.images('D:/Python/gallery-dl/deviantart/Popular/all-time/Ganyu', 4, 4)

print(ganyu4x4.shape)
print(ganyu4x4[0].shape)
plt.imshow(ganyu4x4[0])
plt.show()

DatasetCreator.save_dataset('D:/Python/Projects/GANs', ganyu4x4, 'ganyu4x4')

ganyu4x4 = DatasetCreator.load_dataset('D:/Python/Projects/GANs', 'ganyu4x4')

ganyu4x4 = ganyu4x4.astype('float32')

ganyu4x4 = ganyu4x4/127.5 - 1

ganyu4x4 = torch.from_numpy(ganyu4x4)

ganyu4x4 = ganyu4x4.view(ganyu4x4.shape[0], ganyu4x4.shape[3], ganyu4x4.shape[1], ganyu4x4.shape[2]) # Input_shape must be (n_samples, n_channels, width, height)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Utility function just to create layers more easily

def transconv2out(input, kernel, stride, padding):
    x = (input-1)*stride
    y = 2*padding
    z = 1*(kernel-1)

    output = x - y + z + 1
    return output

#print(transconv2out(1, 4, 1, 0))

# Generator Code

class Generator(nn.Module):
    def __init__(self, ):
        super(Generator, self).__init__()
        self.transconv1 = nn.ConvTranspose2d( 100, 3, 4, 1, 0, bias=False)
        '''self.batchnorm1 = nn.BatchNorm2d(ngf * 8)
        self.ReLU = nn.ReLU(True)
        self.transconv2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(ngf * 4)
        self.transconv3 = nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(ngf * 2)
        self.transconv4 = nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(ngf)
        self.transconv5 = nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False)'''
        self.tanh = nn.Tanh()
    
    # Note: Consider adding LSTMs
    # Note²: Consider using Batchnorm with momentum = 0.8
    # Note³: Consider using LeakyReLU 0.2 just like NVidia did

    def forward(self, input):
        # Level 1 ---> 1 transconv only
        x = self.transconv1(input)
        output = self.tanh(x)
        '''#Level 2
        x = self.batchnorm1(x)
        x = self.ReLU(x)
        x = self.transconv2(x)
        output = self.tanh(x)'''
        # And so on...
        return output


# For LSTMs: We would need sequence(n) input ----> sequence(n) output
# That is, we would need a sequence of images, or a sequence of feature maps, to get a sequence of images/feature maps as output.
# Sequence of images ---> Preprocessing the data
# Sequence of feature maps ---> Dealing with the output of the conv2d

netG = Generator().to(device)

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.02.
netG.apply(weights_init) # This probably will only be useful for the 1st level (4x4) of training

# Print the model
#print(netG)
#summary(netG, (100, 1,1))

# Another utility function

def conv2out(input, kernel, stride, padding):
    x = 2*padding
    y = 1*(kernel-1)
    z = (input + x - y - 1)/stride

    output = z + 1
    return output

#print(conv2out(4, 4, 1, 0))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(3, 1, 4, 1, 0, bias=False)
        '''self.LeakyRelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.4, inplace=False) # Perhaps consider using Dropout2D? Or even 3D?
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(ndf * 2)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(ndf * 4)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(ndf * 8)
        self.conv5 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)'''
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input):
        # Level 1 ---> Conv and sigmoid
        x = self.conv1(input)
        output = self.sigmoid(x)
        '''# Level 2
        x = torch.randn(x.size()).to(device) + x # Adding random noise (Improved Technique from OpenAI)
        x = self.LeakyRelu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.sigmoid(x)'''
        # And so on...
        return output

# Create the Discriminator
netD = Discriminator().to(device)

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init) # This probably will only be useful for the 1st level (4x4) of training

# Print the model
#print(netD)
#summary(netD, (nc, ndf,ndf))

# Establish convention for real and fake labels during training --> Using 0.9 and 0.1 instead of 1 and 0 ---> One-sided label smoothing
# https://arxiv.org/pdf/1606.03498.pdf
real_label = 0.9
fake_label = 0.1 

# Setup Adam optimizers for both G and D ---> NVidia Progressive Grow: Same optimizer parameters with Adam - lr = 0.001, b1 = 0, b2 = 0.99
optimizerD = optim.Adam(netD.parameters(), lr=0.001, betas=(0, 0.99)) # Consider changing learning rate or even the optimizer
optimizerG = optim.Adam(netG.parameters(), lr=0.001, betas=(0, 0.99)) # If learning rate is too aggressive --> might fail to converge or might collapse

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []

print("Starting Training Loop...")

def train(data=None, epochs=1000, batch_size=6,loss=nn.BCELoss(), optimizerD=optimizerD, optimizerG=optimizerG, save_point=100, checkpoint=5000, model_name='model'):
    if os.path.isfile(f'discriminator_{model_name}.pth'):
        netD.load_state_dict(torch.load(f'discriminator_{model_name}.pth'))
    
    if os.path.isfile(f'generator_{model_name}.pth'):
        netG.load_state_dict(torch.load(f'generator_{model_name}.pth'))


    for epoch in range(epochs):
        netD.zero_grad()
        # Format batch
        b_size = batch_size
        real_cpu = data[np.random.randint(0, data.shape[0], size=batch_size), :, :, :].to(device)
        label = torch.full((real_cpu.shape[0],), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1) # Gera um tensor com shape (batch_size,)
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
        # Update scheduler
        #schedulerD.step()

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
        # Update scheduler
        #schedulerG.step()

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
                    % (epoch, epochs,
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
            #print(f'Discriminator last LR: {schedulerD.get_last_lr()}')
            #print(f'Generator last LR: {schedulerG.get_last_lr()}')

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
            np.save(f'fake_images/{model_name}_{epoch}.npy', saving_image[6], allow_pickle=True)
            print(f'Fake image saved!')

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())


train(data=ganyu4x4,epochs=50001, checkpoint=5000, save_point=5000, model_name='default')


plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=3000, blit=True)

HTML(ani.to_jshtml())

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()

def plot_images(fake=True, n_samples=16, noise=None, X_train=None, model='default'):
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

#train(data=ganyu4x4,epochs=50001, checkpoint=5000, save_point=5000, model_name='default')

optimizerD = optim.Adam(netD.parameters(), lr=0.0005)
optimizerG = optim.Adam(netG.parameters(), lr=0.0003)

print("****BEGINNING MODEL1 TRAINING***")
train(data=ganyu4x4, epochs=25001, checkpoint=5000, save_point=5000, model_name='model1')

#plot_images(fake=True, X_train=dataset, model='model1')
