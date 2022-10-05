from torchsummary import summary
from torch import nn
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

PATH = "D:/Python/gallery-dl/Ganyu"

class Data(torch.utils.data.Dataset):
    def __init__(self, path, y, evaluate=False):
        
        images = []

        for directory, _, files in os.walk(path):
            for file in files:
                images.append(directory+'/'+file)

        images = [i for i in images if '.jpg' in i or '.png' in i]
        
        chunk1 = images[0:2000] # # Not even Paperspace's Free GPU could handle my dataset. But then...deviantart's images aren't for any GPU.
        chunk2 = images[2000:4000]
        chunk3 = images[4000:6000]
        chunk4 = images[6000:8000]
        chunk5 = images[8000:] # Continue on as you wish, but with care.
        
        chunk1 = self._create_chunk_array(chunk1)
        chunk2 = self._create_chunk_array(chunk2)
        chunk3 = self._create_chunk_array(chunk3)
        chunk4 = self._create_chunk_array(chunk4)
        chunk5 = self._create_chunk_array(chunk5)
        
        data12 = np.concatenate((chunk1, chunk2), axis=0)
        data34 = np.concatenate((chunk3, chunk4), axis=0)
        data14 = np.concatenate((data12, data34), axis=0)
        data = np.concatenate((data14, data5), axis=0)
        
        data = torch.from_numpy(data)
        
        data = data.view(data.shape[0], data.shape[3], data.shape[1], data.shape[2]) # (N_Samples, Channels, Height, Width)
        
        y = torch.tensor(y)
        y = y.long()
        
        print(f"Torch Data Size: {data.size()}\nTorch Labels Size: {y.size()}")
        
        self.data = data
        
        self.labels = y

    def __getitem__(self, idx):
        idx = idx-1
        data = self.data[idx]
        label = self.labels[idx]

        return data, label

    def __len__(self):

        return len(self.data)
      
      
    def _create_chunk_array(self, chunk_list, size=(500,500)):
        chunk = []

        for i in chunk_list:
            image = Image.open(i)
            image = image.resize(size)
            if image.mode != "RGB":
                image = image.convert("RGB")

            pic = np.array(image)
            image.close()

            chunk.append(pic)

        chunk = np.array(chunk)
        chunk = np.stack(chunk, 0)
        
        return chunk
      
class AlternativeDatasetInCaseYourKernelOrHardwareKeepsDying(torch.utils.data.Dataset):
    def __init__(self, PreprocessedX, Preprocessedy):
        
        supervised = len(Preprocessedy)

        self.features = PreprocessedX[0:supervised]
        self.labels = Preprocessedy

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        return feature, label

    def __len__(self):

        return len(self.features)
      
dataset = Data().to(device)


class ConvBlock(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, input_channels*2, kernel_size, stride, padding, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(input_channels*2)
        self.Relu = nn.ReLU()
        self.conv2 = nn.Conv2d(input_channels*2, input_channels*2, kernel_size, stride, padding, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(input_channels*2)
        self.conv3 = nn.Conv2d(input_channels*2, output_channels//2, kernel_size, stride, padding, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(output_channels//2)
        self.conv4 = nn.Conv2d(output_channels//2, output_channels, kernel_size, stride, padding, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(output_channels)
        self.conv5 = nn.Conv2d(output_channels, output_channels, kernel_size, stride, padding, bias=False)
        self.batchnorm5 = nn.BatchNorm2d(output_channels)

    
    def forward(self, input):

        x = self.conv1(input)
        x = self.batchnorm1(x)
        x = self.Relu(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.Relu(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.Relu(x)
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.Relu(x)
        x = self.conv5(x)
        output = self.batchnorm5(x)

        return output
      
      
class FilterNN(nn.Module):
    def __init__(self):
        super(FilterNN, self).__init__()
        
        self.convblock1 = ConvBlock(3, 9, 6, 1, 0) # 175x175
        self.convblock2 = ConvBlock(9, 27, 6, 1, 0) # 150x150
        self.convblock3 = ConvBlock(27, 30, 6, 1, 0) # 125x125
        self.convblock4 = ConvBlock(30, 90, 6, 1, 0) # 100x100
        self.convblock5 = ConvBlock(90, 30, 6, 1, 0) # 75x75
        self.convblock6 = ConvBlock(30, 27, 6, 1, 0) # 50x50
        self.convblock7 = ConvBlock(27, 9, 6, 1, 0) # 25x25
        self.conv8 = nn.Conv2d(9, 15, 6, 1, 0, bias=False) # 20x20
        self.batchnorm8 = nn.BatchNorm2d(15)
        self.conv9 = nn.Conv2d(15, 9, 6, 1, 0, bias=False) # 15x15
        self.batchnorm9 = nn.BatchNorm2d(9)
        self.conv10 = nn.Conv2d(9, 6, 6, 1, 0, bias=False) # 10x10
        self.batchnorm10 = nn.BatchNorm2d(6)
        self.conv11 = nn.Conv2d(6, 4, 6, 1, 0, bias=False) # 5x5
        self.batchnorm11 = nn.BatchNorm2d(4)
        self.conv12 = nn.Conv2d(4, 3, 5, 1, 0, bias=False) # 1x1
        
        
        self.convalt1 = nn.Conv2d(3, 9, 26, 1, 0, bias=False) # 175x175
        self.batchnorm1 = nn.BatchNorm2d(9)
        self.convalt2 = nn.Conv2d(9, 27, 26, 1, 0, bias=False) # 150x150
        self.batchnorm2 = nn.BatchNorm2d(27)
        self.convalt3 = nn.Conv2d(27, 30, 26, 1, 0, bias=False) # 125x125
        self.batchnorm3 = nn.BatchNorm2d(30)
        self.convalt4 = nn.Conv2d(30, 90, 26, 1, 0, bias=False) # 100x100
        self.batchnorm4 = nn.BatchNorm2d(90)
        self.convalt5 = nn.Conv2d(90, 30, 26, 1, 0, bias=False) # 75x75
        self.batchnorm5 = nn.BatchNorm2d(30)
        self.convalt6 = nn.Conv2d(30, 27, 26, 1, 0, bias=False) # 50x50
        self.batchnorm6 = nn.BatchNorm2d(27)
        self.convalt7 = nn.Conv2d(27, 9, 26, 1, 0, bias=False) # 25x25
        self.batchnorm7 = nn.BatchNorm2d(9)
        
        self.Relu = nn.ReLU()
        
    def forward(self, input):
        
        r1 = self.convalt1(input)
        r1 = self.batchnorm1(r1)
        r1 = self.Relu(r1)
        
        x = self.convblock1(input)
        x = self.Relu(x)
        
        x = x + r1
        
        r2 = self.convalt2(r1)
        r2 = self.batchnorm2(r2)
        r2 = self.Relu(r2)
        
        x = self.convblock2(x)
        x = self.Relu(x)
        
        x = x + r2
        
        r3 = self.convalt3(r2)
        r3 = self.batchnorm3(r3)
        r3 = self.Relu(r3)
        
        x = self.convblock3(x)
        x = self.Relu(x)
        
        x = x + r3
        
        r4 = self.convalt4(r3)
        r4 = self.batchnorm4(r4)
        r4 = self.Relu(r4)
        
        x = self.convblock4(x)
        x = self.Relu(x)
        
        x = x + r4
        
        r5 = self.convalt5(r4)
        r5 = self.batchnorm5(r5)
        r5 = self.Relu(r5)
        
        x = self.convblock5(x)
        x = self.Relu(x)
        
        x = x + r5
        
        r6 = self.convalt6(r5)
        r6 = self.batchnorm6(r6)
        r6 = self.Relu(r6)
        
        x = self.convblock6(x)
        x = self.Relu(x)
        
        x = x + r6
        
        r7 = self.convalt7(r6)
        r7 = self.batchnorm7(r7)
        r7 = self.Relu(r7)
        
        x = self.convblock7(x)
        x = self.Relu(x)
        
        x = x + r7
        
        x = r7
        
        x = self.conv8(x)
        x = self.batchnorm8(x)
        x = self.Relu(x)
        
        x = self.conv9(x)
        x = self.batchnorm9(x)
        x = self.Relu(x)
        
        x = self.conv10(x)
        x = self.batchnorm10(x)
        x = self.Relu(x)
        
        x = self.conv11(x)
        x = self.batchnorm11(x)
        x = self.Relu(x)
        
        x = self.conv12(x)
        
        output = self.Relu(x)
        
        return output
      
      
Filter = FilterNN().to(device)

optimizer = torch.optim.AdamW(Filter.parameters(), lr=1e-12, betas=(0, 0.99), weight_decay=1e-3)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

summary(Filter, (3, 200, 200))

def train(dataset=None, epochs=1000, batch_size=6,loss=nn.CrossEntropyLoss(), optimizer=optimizer, checkpoint=5000):
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for item, (data, labels) in enumerate(dataloader):
            Filter.zero_grad()
            output = Filter(data.to(device))
            output = output.view(output.shape[0], -1) # Cross Entropy ----> output = (Batch, 3); labels = (Batch)
            
            labels = labels.to(device)

            cost = loss(output, labels)

            cost.backward()
            optimizer.step()


            best_loss = float("inf")
            if cost < best_loss:
                best_params = Filter.state_dict()
    
            if item % checkpoint == 0 or epoch == epochs-1:
                print(f"{epoch}|{epochs}\t Iteration: {item}\t Model Loss: {cost}\t Last LR: {scheduler.get_last_lr()}")
                print(Filter.conv12.weight.grad) # Monitoring gradients

                torch.save(best_params, f'Filter.pth')

                print("Model saved!")
                
        #scheduler.step()
        
        
        
def predict(data=None, batch_size=6):
    output = []
    
    inputs = data[item*batch_size:min(item*batch_size+batch_size, len(dataset))]

    for item, data in enumerate(inputs):
        
        Filter.load_state_dict(torch.load("Filter.pth"))

        predicted = Filter(data.to(device))
        for label in predicted:
            output.append(label)
        
    output = np.array(output)

    return output
  
  
train(dataset=dataset, epochs=10000, checkpoint=100) # Beware of vanishing/exploding gradients.

unsupervised_labels = predict(data=PreprocessedX) 
