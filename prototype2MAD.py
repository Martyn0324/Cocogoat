# GIMME FUEL! GIMME FIRE! GIMME THAT WHICH I DESIRE!

class MADGen(nn.Module):
    def __init__(self, ):
        super(MADGen, self).__init__()

        self.transconv1 = nn.ConvTranspose2d(100, 3, 4, 2, 0, bias=False) # 100 -> 75 -> 50 -> 25 -> 3
        self.batchnorm1 = nn.BatchNorm2d(3, momentum=0.8)
        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.stdtransconv = nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=False)
        self.transconv2 = nn.ConvTranspose2d(3, 100, 3, 1, 1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(100, momentum=0.8)
        self.transconv3 = nn.ConvTranspose2d(100, 75, 3, 1, 1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(75, momentum=0.8)
        self.transconv4 = nn.ConvTranspose2d(75, 50, 3, 1, 1, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(50, momentum=0.8)
        self.transconv5 = nn.ConvTranspose2d(50, 25, 3, 1, 1, bias=False)
        self.batchnorm5 = nn.BatchNorm2d(25, momentum=0.8)
        self.revtransconv = nn.ConvTranspose2d(25, 3, 4, 2, 1, bias=False)
        self.tanh = nn.Tanh()
    
    # Note: Consider adding LSTMs
    # Note: Consider using Batchnorm with momentum = 0.8

    def forward(self, input):
        # Level 1 ---> 1 transconv only
        x = self.transconv1(input)
        output1 = self.tanh(x) # 4x4
        x = self.transconv2(x)
        x = self.batchnorm2(x)
        x = self.LeakyReLU(x)
        x = self.transconv3(x)
        x = self.batchnorm3(x)
        x = self.LeakyReLU(x)
        x = self.transconv4(x)
        x = self.batchnorm4(x)
        x = self.LeakyReLU(x)
        x = self.transconv5(x)
        x = self.batchnorm5(x)
        x = self.LeakyReLU(x)
        x = self.revtransconv(x)
        output2 = self.tanh(x) # 8x8
        x = self.transconv2(x)
        x = self.batchnorm2(x)
        x = self.LeakyReLU(x)
        x = self.transconv3(x)
        x = self.batchnorm3(x)
        x = self.LeakyReLU(x)
        x = self.transconv4(x)
        x = self.batchnorm4(x)
        x = self.LeakyReLU(x)
        x = self.transconv5(x)
        x = self.batchnorm5(x)
        x = self.LeakyReLU(x)
        x = self.revtransconv(x)
        output3 = self.tanh(x) # 16x16
        x = self.transconv2(x)
        x = self.batchnorm2(x)
        x = self.LeakyReLU(x)
        x = self.transconv3(x)
        x = self.batchnorm3(x)
        x = self.LeakyReLU(x)
        x = self.transconv4(x)
        x = self.batchnorm4(x)
        x = self.LeakyReLU(x)
        x = self.transconv5(x)
        x = self.batchnorm5(x)
        x = self.LeakyReLU(x)
        x = self.revtransconv(x)
        output4 = self.tanh(x) # 32x32
        x = self.transconv2(x)
        x = self.batchnorm2(x)
        x = self.LeakyReLU(x)
        x = self.transconv3(x)
        x = self.batchnorm3(x)
        x = self.LeakyReLU(x)
        x = self.transconv4(x)
        x = self.batchnorm4(x)
        x = self.LeakyReLU(x)
        x = self.transconv5(x)
        x = self.batchnorm5(x)
        x = self.LeakyReLU(x)
        x = self.revtransconv(x)
        output = self.tanh(x) # 64x64
        return output1, output2, output3, output4, output
      
      
MADG = MADGen().to(device)

MADG.apply(weights_init)

summary(MADG, (100, 1,1))

# The discriminators can be the same ones as the SimpleGen...or you can use something bigger...no mercy for GPUs! Even less for TPUs!
