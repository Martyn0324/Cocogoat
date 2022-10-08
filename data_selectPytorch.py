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

        self.conv1 = nn.Conv2d(3, 1000, 2, 2, 0, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(1000)
        self.Relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(1000, 100, 2, 2, 0, bias=False) # 50x50
        self.batchnorm2 = nn.BatchNorm2d(100)
        
        self.convblock3 = ConvBlock(100, 75, 3, 1, 0)
        self.convblock4 = ConvBlock(75, 50, 3, 1, 0)
        self.convblock5 = ConvBlock(50, 25, 3, 1, 0)
        self.convblock6 = ConvBlock(25, 10, 3, 1, 0)
        self.conv7 = nn.Conv2d(10, 8, 3, 1, 0)
        self.batchnorm7 = nn.BatchNorm2d(8)
        self.conv8 = nn.Conv2d(8, 6, 4, 1, 0)
        self.batchnorm8 = nn.BatchNorm2d(6)
        self.conv9 = nn.Conv2d(6, 4, 4, 1, 0)
        self.batchnorm9 = nn.BatchNorm2d(4)
        self.conv10 = nn.Conv2d(4, 3, 2, 1, 0)
    
    def forward(self, input):
        
        x = self.conv1(input)
        x = self.batchnorm1(x)
        x = self.Relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.Relu(x)
        x = self.dropout(x)
        
        x = self.convblock3(x)
        x = self.Relu(x)
        x = self.dropout(x)
        x = self.convblock4(x)
        x = self.Relu(x)
        x = self.dropout(x)
        x = self.convblock5(x)
        x = self.Relu(x)
        x = self.dropout(x)
        x = self.convblock6(x)
        x = self.Relu(x)
        x = self.dropout(x)
        
        x = self.conv7(x)
        x = self.batchnorm7(x)
        x = self.Relu(x)
        x = self.dropout(x)
        x = self.conv8(x)
        x = self.batchnorm8(x)
        x = self.Relu(x)
        x = self.dropout(x)
        x = self.conv9(x)
        x = self.batchnorm9(x)
        x = self.Relu(x)
        x = self.dropout(x)
        x = self.conv10(x)
        
        output = self.Relu(x) # Pytorch's Cross Entropy includes a softmax function.

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

            output = output.view(output.shape[0], -1) # Cross Entropy ----> output = (Batch, 3); labels = (Batch) ---> Index encoding. This is why type(labels) = int
            
            labels = labels.to(device)

            cost = loss(output, labels)

            cost.backward()
            optimizer.step()


            best_loss = float("inf")
            if cost < best_loss:
                best_loss = cost
                best_params = Filter.state_dict()
    
            if item % checkpoint == 0 or epoch == epochs-1:
                print(f"{epoch}|{epochs}\t Iteration: {item}\t Model Loss: {cost}\t Last LR: {scheduler.get_last_lr()}")

                torch.save(best_params, f'Filter.pth')

                print("Model saved!")
                
        scheduler.step()
        
        
        
def predict(data=None, batch_size=6):
    output = []
    
    inputs = data[item*batch_size:min(item*batch_size+batch_size, len(dataset))]

    for item, data in enumerate(inputs):
        
        Filter.load_state_dict(torch.load("Dataset_Filter.pth"))

        predicted = Filter(data.to(device))
        for label in predicted:
            output.append(label)
        
    output = np.array(output)

    return output
  
  
train(dataset=dataset, epochs=10000, checkpoint=100) # Beware of vanishing/exploding gradients.

unsupervised_labels = predict(data=PreprocessedX) 
