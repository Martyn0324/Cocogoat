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
    def __init__(self, ):
        super(Generator, self).__init__()
        self.transconv1 = nn.ConvTranspose2d(100, 3, 4, 1, 0, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(50)
        self.ReLU = nn.ReLU(True)
        self.transconv2 = nn.ConvTranspose2d(50, 3, 4, 2, 1, bias=False)
        self.lstm2 = nn.LSTM(input_size=8,hidden_size=8,num_layers=10, bias=False, batch_first=True, bidirectional=False)
        '''self.batchnorm2 = nn.BatchNorm2d(ngf * 4)
        self.transconv3 = nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(ngf * 2)
        self.transconv4 = nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(ngf)
        self.transconv5 = nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False)'''
        self.transconv6 = nn.ConvTranspose2d(3, 3, 1, 1, (4,0), bias=False)
        self.tanh = nn.Tanh()
        
    def forward(self, input):
        x = self.transconv1(input)
        x = self.batchnorm1(x)
        x = self.ReLU(x)
        x = self.transconv2(x)
        '''x = self.batchnorm2(x)
        x = self.ReLU(x)
        x = self.transconv3(x)
        x = self.batchnorm3(x)
        x = self.transconv4(x)
        x = self.batchnorm4(x)
        x = self.transconv5(x)
        print(f"transconv 5 shape: {x.shape}")'''
        R, G, B = self._preprocess(x)
        outR, _ = self.lstm2(R)
        outG, _ = self.lstm2(G)
        outB, _ = self.lstm2(B)
        output = np.stack([outR.detach().cpu(), outG.detach().cpu(), outB.detach().cpu()], axis=1) # (8-4, 3, 4, 8)
        # It's necessary to convert to (1, 3, 8, 8), but np.reshape isn't possible right now, since 4x3x4x8 != 1x3x8x8
        # Possibilities:
        # *Pass each color array into a transconv until we can get a monochromatic 8x8 image(R, G, B), then np.stack to obtain 8x8 RGB image;
        # *Reshape this output and pass it directly through a transconv, adjusting the parameters to get a 8x8 RGB image.
        output = np.reshape(output, (1, 3, 16, 8))
        output = torch.from_numpy(output)
        output = self.transconv6(output.to(device))
        output = self.tanh(x)

        return output

    def _preprocess(self, RGB):
        R = RGB[:,0,:,:] # [n_samples, RGB, width, height] ----> Returns (n_samples, width, height)
        G = RGB[:,1,:,:]
        B = RGB[:,2,:,:]

        R = R.detach().cpu().numpy()
        G = G.detach().cpu().numpy()
        B = B.detach().cpu().numpy()
        
        outputR = []
        outputG = []
        outputB = []

        for i in range(4, 8): # Using the last 4 pixels to define the next one.
            outputR.append(R[0, i-4:i,:])
            outputG.append(G[0, i-4:i,:])
            outputB.append(B[0, i-4:i,:]) # We could iterate through each sample in RGB, but this would make this process too expensive. It's better to use 1 sample at time.
        
        outputR = np.array(outputR, dtype='float32') # (n_samples, n_timesteps, n_features) ---> Since we have an 8x8 image, we have (8-4) samples and 8 features
        outputG = np.array(outputG, dtype='float32')
        outputB = np.array(outputB, dtype='float32')

        return torch.from_numpy(outputR).to(device), torch.from_numpy(outputG).to(device), torch.from_numpy(outputB).to(device)
      
      
netG = Generator().to(device)

noise = torch.randn(1, 100, 1,1, device=device)
output = netG(noise)
print(output.shape)
