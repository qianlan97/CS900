import torchvision.transforms as transforms
import torchvision
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch import nn
from torch.nn import DataParallel
import numpy as np
import time
from tqdm import tqdm
# from torchsummary import summary

torch.cuda.empty_cache()
device = torch.device("cpu")              
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"running on {device}")

train_tf = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.49139968,0.48215841,0.44653091],
                                       [0.24703223,0.24348513,0.26158784])]
)
valid_tf = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.49139968,0.48215841,0.44653091],
                                       [0.24703223,0.24348513,0.26158784])]
)

batch_size = 8
train_dataset = torchvision.datasets.CIFAR10(root = './dataset',
                           train = True,
                           transform = train_tf,
                           download = True)
test_dataset = torchvision.datasets.CIFAR10(root = './dataset',
                           train = False,
                           transform = valid_tf,
                           download = True)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, 
                                           batch_size = batch_size, 
                                           shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = batch_size,
                                          shuffle = False)
 


class VGG19(nn.Module):
    def __init__(self,in_dim,num_classes):
        super().__init__()
        self.features = nn.Sequential(
             nn.Conv2d(in_dim,64,kernel_size=3,padding=1),
             nn.BatchNorm2d(64),
             nn.ReLU(inplace=True),
             nn.Conv2d(64,64,kernel_size=3,padding=1),
             nn.BatchNorm2d(64),
             nn.ReLU(inplace=True),
             nn.MaxPool2d(kernel_size=2,stride=2),
 
             nn.Conv2d(64,128,kernel_size=3,padding=1),
             nn.BatchNorm2d(128),
             nn.ReLU(inplace=True),
             nn.Conv2d(128, 128, kernel_size=3, padding=1),
             nn.BatchNorm2d(128),
             nn.ReLU(inplace=True),
             nn.MaxPool2d(kernel_size=2,stride=2),
 
             nn.Conv2d(128, 256, kernel_size=3, padding=1),
             nn.BatchNorm2d(256),
             nn.ReLU(inplace=True),
             nn.Conv2d(256, 256, kernel_size=3, padding=1),
             nn.BatchNorm2d(256),
             nn.ReLU(inplace=True),
             nn.Conv2d(256, 256, kernel_size=3, padding=1),
             nn.BatchNorm2d(256),
             nn.ReLU(inplace=True),
             nn.Conv2d(256, 256, kernel_size=3, padding=1),
             nn.BatchNorm2d(256),
             nn.ReLU(inplace=True),
             nn.MaxPool2d(kernel_size=2,stride=2),
            
             nn.Conv2d(256, 512, kernel_size=3, padding=1),
             nn.BatchNorm2d(512),
             nn.ReLU(inplace=True),
             nn.Conv2d(512, 512, kernel_size=3, padding=1),
             nn.BatchNorm2d(512),
             nn.ReLU(inplace=True),
             nn.Conv2d(512, 512, kernel_size=3, padding=1),
             nn.BatchNorm2d(512),
             nn.ReLU(inplace=True),
             nn.Conv2d(512, 512, kernel_size=3, padding=1),
             nn.BatchNorm2d(512),
             nn.ReLU(inplace=True),
             nn.MaxPool2d(kernel_size=2,stride=2),
            
             nn.Conv2d(512, 512, kernel_size=3, padding=1),
             nn.BatchNorm2d(512),
             nn.ReLU(inplace=True),
             nn.Conv2d(512, 512, kernel_size=3, padding=1),
             nn.BatchNorm2d(512),
             nn.ReLU(inplace=True),
             nn.Conv2d(512, 512, kernel_size=3, padding=1),
             nn.BatchNorm2d(512),
             nn.ReLU(inplace=True),
             nn.Conv2d(512, 512, kernel_size=3, padding=1),
             nn.BatchNorm2d(512),
             nn.ReLU(inplace=True),
             nn.MaxPool2d(kernel_size=2,stride=2)
         )
 
        self.classifier = nn.Sequential(
             nn.Linear(512*7*7,4096),
             nn.ReLU(),
             nn.Dropout(0.5),
 
             nn.Linear(4096,4096),
             nn.ReLU(),
             nn.Dropout(0.5),
            
             nn.Linear(4096,num_classes)
         )


    def forward(self, x):
        layer_times = []
        
        for layer in self.features:
            start_time = time.time()
            x = layer(x)
            end_time = time.time()
            layer_times.append(end_time - start_time)
        
        x = x.view(x.size(0), -1)
        
        for layer in self.classifier:
            start_time = time.time()
            x = layer(x)
            end_time = time.time()
            layer_times.append(end_time - start_time)
        
        return x, layer_times

net = VGG19(3,10)
net.to(device)
# summary(net, input_size=(3, 224, 224))

num_images = 64
num_batches = num_images // batch_size

learning_rate = 1e-2
num_epochs = 1
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr = learning_rate,momentum=0.9)
net.train()

start_time = time.time()
for epoch in range(num_epochs):
    running_loss = 0.0
    correct_top1 = 0
    correct_top3 = 0
    total = 0
    batch_count = 0
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs, layer_times = net(images)
        
        # Print layer times
        for i, layer_time in enumerate(layer_times):
            print(f"Layer {i+1}: {layer_time:.4f} seconds")
        
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        break
        
    print(time.time() - start_time)