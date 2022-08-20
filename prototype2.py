import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
import os
from PIL import Image

PATH = 'filtered_images.npy'
data = np.load(PATH)

dataset = torch.from_numpy(dataset)
dataset = dataset.view(dataset.shape[0], dataset.shape[3], dataset.shape[1], dataset.shape[2])
print(dataset.size())

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SimpleGen(nn.Module):
    def __init__(self, ):
        super(SimpleGen, self).__init__()

        self.transconv1 = nn.ConvTranspose2d(100, 3, 4, 2, 0, bias=False) # 100 -> 75 -> 50 -> 40 -> 30 -> 25 -> 15 -> 10 -> 7
        #self.transconv1 = nn.ConvTranspose2d(100, 75, 4, 2, 0, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(3, momentum=0.8)
        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.transconv2 = nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(3, momentum=0.8)
        self.transconv3 = nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(3, momentum=0.8)
        self.transconv4 = nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(3, momentum=0.8)
        self.transconv5 = nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=False)
        self.tanh = nn.Tanh()
    

    def forward(self, input):
        # Level 1 ---> 1 transconv only
        x = self.transconv1(input)
        output1 = self.tanh(x) # 4x4
        x = self.batchnorm1(x)
        x = self.LeakyReLU(x)
        x = self.transconv2(x)
        output2 = self.tanh(x) # 8x8
        x = self.batchnorm2(x)
        x = self.LeakyReLU(x)
        x = self.transconv3(x)
        output3 = self.tanh(x) # 16x16
        x = self.batchnorm3(x)
        x = self.LeakyReLU(x)
        x = self.transconv4(x)
        output4 = self.tanh(x) # 32x32
        x = self.batchnorm4(x)
        x = self.LeakyReLU(x)
        x = self.transconv5(x)
        output = self.tanh(x) # 64x64
        return output1, output2, output3, output4, output
      
      
netG = SimpleGen().to(device)

netG.apply(weights_init)

summary(netG, (100, 1,1)) # just to make sure everything's ok

class Discriminator4x4(nn.Module):
    def __init__(self):
        super(Discriminator4x4, self).__init__()

        self.conv1 = nn.Conv2d(3, 1, 4, 2, 0, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input):
        x = self.conv1(input)
        output = self.sigmoid(x)

        return output
      
      
D4 = Discriminator4x4().to(device)

D4.apply(weights_init)

print(D4)
summary(D4, (3, 4,4))

class Discriminator8x8(nn.Module):
    def __init__(self):
        super(Discriminator8x8, self).__init__()

        self.conv1 = nn.Conv2d(3, 100, 4, 2, 0, bias=False)
        self.LeakyRelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.4, inplace=False) # Perhaps consider using Dropout2D? Or even 3D?
        self.conv2 = nn.Conv2d(100, 1, 4, 2, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input):
        x = self.conv1(input)
        x = self.LeakyRelu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        output = self.sigmoid(x)

        return output
      
D8 = Discriminator8x8().to(device)

D8.apply(weights_init)

print(D8)
summary(D8, (3, 8,8))

class Discriminator16x16(nn.Module):
    def __init__(self):
        super(Discriminator16x16, self).__init__()

        self.conv1 = nn.Conv2d(3, 100, 4, 2, 0, bias=False)
        self.LeakyRelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.4, inplace=False) # Perhaps consider using Dropout2D? Or even 3D?
        self.conv2 = nn.Conv2d(100, 50, 4, 2, 1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(50, momentum=0.8)
        self.conv3 = nn.Conv2d(50, 1, 4, 2, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input):
        x = self.conv1(input)
        x = self.LeakyRelu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.LeakyRelu(x)
        x = self.dropout(x)
        x = self.conv3(x)
        output = self.sigmoid(x)

        return output
      
D16 = Discriminator16x16().to(device)

D16.apply(weights_init)

print(D16)
summary(D16, (3, 16,16))

class Discriminator32x32(nn.Module):
    def __init__(self):
        super(Discriminator32x32, self).__init__()

        self.conv1 = nn.Conv2d(3, 100, 4, 2, 0, bias=False)
        self.LeakyRelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.4, inplace=False) # Perhaps consider using Dropout2D? Or even 3D?
        self.conv2 = nn.Conv2d(100, 75, 4, 2, 1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(75, momentum=0.8)
        self.conv3 = nn.Conv2d(75, 50, 4, 2, 1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(50, momentum=0.8)
        self.conv4 = nn.Conv2d(50, 1, 4, 2, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input):
        x = self.conv1(input)
        x = self.LeakyRelu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.LeakyRelu(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.LeakyRelu(x)
        x = self.dropout(x)
        x = self.conv4(x)
        output = self.sigmoid(x)

        return output
      
D32 = Discriminator32x32().to(device)

D32.apply(weights_init)

print(D32)
summary(D32, (3, 32,32))

class Discriminator64x64(nn.Module):
    def __init__(self):
        super(Discriminator64x64, self).__init__()

        self.conv1 = nn.Conv2d(3, 100, 4, 2, 0, bias=False)
        self.LeakyRelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.4, inplace=False) # Perhaps consider using Dropout2D? Or even 3D?
        self.conv2 = nn.Conv2d(100, 75, 4, 2, 1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(75, momentum=0.8)
        self.conv3 = nn.Conv2d(75, 50, 4, 2, 1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(50, momentum=0.8)
        self.conv4 = nn.Conv2d(50, 25, 4, 2, 1, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(25)
        self.conv5 = nn.Conv2d(25, 1, 4, 2, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input):
        x = self.conv1(input)
        x = self.LeakyRelu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.LeakyRelu(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.LeakyRelu(x)
        x = self.dropout(x)
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.LeakyRelu(x)
        x = self.dropout(x)
        x = self.conv5(x)
        output = self.sigmoid(x)

        return output
      
D64 = Discriminator64x64().to(device)

D64.apply(weights_init)

print(D64)
summary(D64, (3, 64,64))


real_label = 0.8
fake_label = 0.2

optimizerD4 = optim.Adam(D4.parameters(), lr=0.001, betas=(0, 0.99))
optimizerD8 = optim.Adam(D8.parameters(), lr=0.001, betas=(0, 0.99))
optimizerD16 = optim.Adam(D16.parameters(), lr=0.001, betas=(0, 0.99))
optimizerD32 = optim.Adam(D32.parameters(), lr=0.001, betas=(0, 0.99))
optimizerD64 = optim.Adam(D64.parameters(), lr=0.001, betas=(0, 0.99))

optimizerG = optim.AdamW(netG.parameters(), lr=0.001, betas=(0, 0.99), weight_decay=1e-5)



def train(
    data=None,
    netG=netG,
    epochs=1000,
    batch_size=256,
    loss=nn.BCELoss(),
    optimizerG=optimizerG,
    optimizerD4=optimizerD4,
    optimizerD8=optimizerD8,
    optimizerD16=optimizerD16,
    optimizerD32=optimizerD32,
    optimizerD64=optimizerD64,
    save_point=1000,
    checkpoint=5000,
    model_name='default_Cocogoat',
    keep_going="no"):
    
    print("Starting Training Loop...")

    if keep_going == "yes":
        D4.load_state_dict(torch.load(f'Saves/discriminator_{model_name}.pth'))
        D8.load_state_dict(torch.load(f'Saves/discriminator_{model_name}.pth'))
        D16.load_state_dict(torch.load(f'Saves/discriminator_{model_name}.pth'))
        D32.load_state_dict(torch.load(f'Saves/discriminator_{model_name}.pth'))
        D64.load_state_dict(torch.load(f'Saves/discriminator_{model_name}.pth'))
        netG.load_state_dict(torch.load(f'Saves/generator_{model_name}.pth'))
        print("Continuing from last save")
        
    for epoch in range(epochs):
        D4.zero_grad()
        # Format batch
        real_cpu = data[np.random.randint(0, data.shape[0], size=batch_size), :, :, :].to(device)
        label = torch.full((real_cpu.shape[0],), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = D4(real_cpu).view(-1) # Gera um tensor com shape (batch_size,)
        # Calculate loss on all-real batch
        errD4_real = loss(output, label) # Target, Input
        # Calculate gradients for D in backward pass
        errD4_real.backward()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(batch_size, 100, 1, 1, device=device)
        # Generate fake image batch with G
        output1, output2, output3, output4, output = netG(noise)
        label.fill_(fake_label)
        
        # REMEMBER:
        # output1 = 4x4
        # output2 = 8x8
        # output3 = 16x16
        # output4 = 32x32
        # output = 64x64
        
        # Classify all fake batch with D
        Dout1, Dout2 = D4(output1.detach()).view(-1), D8(output2.detach()).view(-1)
        Dout3, Dout4 = D16(output3.detach()).view(-1), D32(output4.detach()).view(-1)
        Dout = D64(output.detach()).view(-1)
        
        # Calculate D's loss on the all-fake batch
        errD4_fake, errD8_fake, errD16_fake = loss(Dout1, label), loss(Dout2, label), loss(Dout3, label)
        errD32_fake, errD64_fake = loss(Dout4, label), loss(Dout, label)
        
        # Calculate the gradients for this batch. D4 will be accumulated (summed) with previous gradients
        errD4_fake.backward()
        optimizerD4.step()
        
        D8.zero_grad()
        errD8_fake.backward()
        optimizerD8.step()
        
        D16.zero_grad()
        errD16_fake.backward()
        optimizerD16.step()
        
        D32.zero_grad()
        errD32_fake.backward()
        optimizerD32.step()
        
        D64.zero_grad()
        errD64_fake.backward()
        optimizerD64.step()
        
        # Compute error of D as sum over the fake and the real batches
        
        errD4 = errD4_real + errD4_fake
        errD8 = errD8_fake
        errD16 = errD16_fake
        errD32 = errD32_fake
        errD64 = errD64_fake
        
        # Update D ----- Moved to lines above to avoid things like "using D64's loss to backpropagate through D8's network"
        # UPDATE: After running some tests, I've discovered that this won't be happening. Each Neural Network acts independently from each other.
        #optimizerD.step()
        #optimizerD4.step()
        #optimizerD8.step()
        #optimizerD16.step()
        #optimizerD32.step()
        #optimizerD64.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output1, output2 = D4(output1).view(-1), D8(output2).view(-1)
        output3, output4 = D16(output3).view(-1), D32(output4).view(-1)
        output = D64(output).view(-1)
        
        # Calculate G's loss based on this output
        
        errG1 = loss(output1, label)
        errG2 = loss(output2, label)
        errG3 = loss(output3, label)
        errG4 = loss(output4, label)
        errG5 = loss(output, label)
        # Calculate gradients for G
        
        errG1.backward(retain_graph=True)
        errG2.backward(retain_graph=True)
        errG3.backward(retain_graph=True)
        errG4.backward(retain_graph=True)
        errG5.backward(retain_graph=True)
        
        # Update G
        optimizerG.step()

        # Getting and saving best parameters:

        best_gen_loss = float('inf')
        generator_loss = errG1.item() + errG2.item() + errG3.item() + errG4.item() + errG5.item()
        
        best_disc1_loss, best_disc2_loss, best_disc3_loss = float('inf'), float('inf'), float('inf')
        best_disc4_loss, best_disc5_loss = float('inf'), float('inf')
        
        discriminator1_loss, discriminator2_loss, discriminator3_loss = errD4.item(), errD8.item(), errD16.item()
        discriminator4_loss, discriminator5_loss = errD32.item(), errD64.item()

        if generator_loss < best_gen_loss:
            best_gen_loss = generator_loss
            best_generator_parameters = netG.state_dict()
                
        if discriminator1_loss < best_disc1_loss:
            best_disc1_loss = discriminator1_loss
            best_discriminator1_parameters = D4.state_dict()
        
        if discriminator2_loss < best_disc2_loss:
            best_disc2_loss = discriminator2_loss
            best_discriminator2_parameters = D8.state_dict()
        
        if discriminator3_loss < best_disc3_loss:
            best_disc3_loss = discriminator3_loss
            best_discriminator3_parameters = D16.state_dict()
        
        if discriminator4_loss < best_disc4_loss:
            best_disc4_loss = discriminator4_loss
            best_discriminator4_parameters = D32.state_dict()
        
        if discriminator5_loss < best_disc5_loss:
            best_disc5_loss = discriminator5_loss
            best_discriminator5_parameters = D64.state_dict()

        # Output training stats
        if epoch % checkpoint == 0:
            print('[%d/%d]\nLoss_D4: %.3f\nLoss_D8: %.3f\nLoss_D16: %.3f\nLoss_D32: %.3f\nLoss_D64: %.3f\tLoss_G: %.4f'
                    % (epoch, epochs,
                        errD4.item(), errD8.item(), errD16.item(), errD32.item(), errD64.item(),
                        generator_loss))

            with torch.no_grad():
                test_image1, test_image2, test_image3, test_image4, test_image5 = netG(noise)
                test_image1, test_image2 = test_image1.detach().cpu(), test_image2.detach().cpu()
                test_image3, test_image4 = test_image3.detach().cpu(), test_image4.detach().cpu()
                test_image5 = test_image5.detach().cpu()

            test_image1 = (test_image1+1.0)*0.5
            test_image1 = test_image1.view(test_image1.shape[0], test_image1.shape[2], test_image1.shape[3], test_image1.shape[1])
            
            test_image2 = (test_image2+1.0)*0.5
            test_image2 = test_image2.view(test_image2.shape[0], test_image2.shape[2], test_image2.shape[3], test_image2.shape[1])
            
            test_image3 = (test_image3+1.0)*0.5
            test_image3 = test_image3.view(test_image3.shape[0], test_image3.shape[2], test_image3.shape[3], test_image3.shape[1])
            
            test_image4 = (test_image4+1.0)*0.5
            test_image4 = test_image4.view(test_image4.shape[0], test_image4.shape[2], test_image4.shape[3], test_image4.shape[1])
            
            test_image5 = (test_image5+1.0)*0.5
            test_image5 = test_image5.view(test_image5.shape[0], test_image5.shape[2], test_image5.shape[3], test_image5.shape[1])
            
            
            torch.save(best_generator_parameters, f'Saves/generator_{model_name}.pth')
            torch.save(best_discriminator1_parameters, f'Saves/discriminator_{model_name}.pth')
            torch.save(best_discriminator2_parameters, f'Saves/discriminator_{model_name}.pth')
            torch.save(best_discriminator3_parameters, f'Saves/discriminator_{model_name}.pth')
            torch.save(best_discriminator4_parameters, f'Saves/discriminator_{model_name}.pth')
            torch.save(best_discriminator5_parameters, f'Saves/discriminator_{model_name}.pth')
            print("Models saved!")

        if epoch % save_point == 0:
            saving_image1, saving_image2, saving_image3, saving_image4, saving_image5 = netG(noise)
            saving_image1, saving_image2 = saving_image1.detach().cpu(), saving_image2.detach().cpu()
            saving_image3, saving_image4 = saving_image3.detach().cpu(), saving_image4.detach().cpu()
            saving_image5 = saving_image5.detach().cpu()
            
            saving_image1 = saving_image1.view(saving_image1.shape[0], saving_image1.shape[2], saving_image1.shape[3], saving_image1.shape[1])
            saving_image1 = saving_image1.numpy()
            saving_image1 = (saving_image1+1.0)*0.5
            np.save(f'Saves/Outputs/{model_name}_{epoch}_4x4.npy', saving_image1[0], allow_pickle=True)
            
            saving_image2 = saving_image2.view(saving_image2.shape[0], saving_image2.shape[2], saving_image2.shape[3], saving_image2.shape[1])
            saving_image2 = saving_image2.numpy()
            saving_image2 = (saving_image2+1.0)*0.5
            np.save(f'Saves/Outputs/{model_name}_{epoch}_8x8.npy', saving_image2[0], allow_pickle=True)
            
            saving_image3 = saving_image3.view(saving_image3.shape[0], saving_image3.shape[2], saving_image3.shape[3], saving_image3.shape[1])
            saving_image3 = saving_image3.numpy()
            saving_image3 = (saving_image3+1.0)*0.5
            np.save(f'Saves/Outputs/{model_name}_{epoch}_16x16.npy', saving_image3[0], allow_pickle=True)
            
            saving_image4 = saving_image4.view(saving_image4.shape[0], saving_image4.shape[2], saving_image4.shape[3], saving_image4.shape[1])
            saving_image4 = saving_image4.numpy()
            saving_image4 = (saving_image4+1.0)*0.5
            np.save(f'Saves/Outputs/{model_name}_{epoch}_32x32.npy', saving_image4[0], allow_pickle=True)
            
            saving_image5 = saving_image5.view(saving_image5.shape[0], saving_image5.shape[2], saving_image5.shape[3], saving_image5.shape[1])
            saving_image5 = saving_image5.numpy()
            saving_image5 = (saving_image5+1.0)*0.5
            np.save(f'Saves/Outputs/{model_name}_{epoch}_64x64.npy', saving_image5[0], allow_pickle=True)
            
            print(f'Fake images saved!')
            
            imagem1 = np.load(f'Saves/Outputs/{model_name}_{epoch}_4x4.npy')
            imagem2 = np.load(f'Saves/Outputs/{model_name}_{epoch}_8x8.npy')
            imagem3 = np.load(f'Saves/Outputs/{model_name}_{epoch}_16x16.npy')
            imagem4 = np.load(f'Saves/Outputs/{model_name}_{epoch}_32x32.npy')
            imagem5 = np.load(f'Saves/Outputs/{model_name}_{epoch}_64x64.npy')
            plt.imshow(imagem1)
            plt.show()
            plt.imshow(imagem2)
            plt.show()
            plt.imshow(imagem3)
            plt.show()
            plt.imshow(imagem4)
            plt.show()
            plt.imshow(imagem5)
            plt.show()
