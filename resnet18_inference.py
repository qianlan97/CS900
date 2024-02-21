import numpy as np
from tqdm import tqdm
import time
import os

import torch
import torchvision
import time
from torch import nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils import data
from torch.utils.data import DataLoader
from torchsummary import summary

# import matplotlib.pyplot as plt
# # %matplotlib inline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"running on {device}")

start_time = time.time()

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.2, 0.2, 0.2))
])

# data_transform=torchvision.transforms.ToTensor()
test_set=torchvision.datasets.CIFAR10(root="./dataset",train=False,download=True,transform=trans)
test_dataset=data.DataLoader(test_set,batch_size=64)

# model = torch.load("resnet18-posttrained.pt")
model = torch.load("resnet18-posttrained.pt", map_location=torch.device('cpu'))
summary(model, input_size=(3,32,32))

model.to(device) ## deploy the model to the GPU

loss=nn.CrossEntropyLoss().to(device)

with torch.no_grad():
    accuracy=0.0
    for data in test_dataset:
        imgs,labels=data
        imgs=imgs.to(device)
        labels=labels.to(device)

        output=model.forward(imgs)
        lossnum=loss(output,labels)
        accuracy+=(output.argmax(1)==labels).sum()
        p_vector = torch.softmax(output, dim=1)

        print("The output vector's size is:", p_vector.size())
        output_array = p_vector.cpu().numpy()
        output_file = "result.txt"
        np.savetxt(output_file, output_array)
        print("Output vector saved to:", output_file)
        break   ## only get the output vector of the first batch in the dataloader(which includes 64 images)
    print("Accuracy on the test dataset is {}".format(accuracy/ len(test_set)))

print("--- %s seconds ---" % (time.time() - start_time))

# with torch.no_grad():
#     accuracy=0.0
#     for data in test_dataset:
#         imgs,labels=data
#         imgs=imgs.to(device)
#         labels=labels.to(device)

#         output=model.forward(imgs)
#         lossnum=loss(output,labels)
#         accuracy+=(output.argmax(1)==labels).sum()
#     print("Accuracy on the test dataset is {}".format(accuracy/ len(test_set)))

# print("--- %s seconds ---" % (time.time() - start_time))