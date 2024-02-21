import torchvision.transforms as transforms
import torchvision
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch import nn
import numpy as np
import time
import torch.nn.functional as F
import socket
import pickle
from datetime import datetime
import mmap
import os
import struct

device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"running on {device}")
file_path_1 = "shared_mmap_1.dat"
file_path_2 = "shared_mmap_2.dat"
status_path = "status.dat"
bn_time = 0
pool_time = 0
relu_time = 0
send_to_server_time = 0
read_from_server_time = 0


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

batch_size = 6

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

def write_data_to_mmap(file_path, data_tuple):
    global send_to_server_time
    tempstart = time.time()
    with open(file_path, "wb") as f:
        # Serialize shape information
        x, *other_params = data_tuple
        shape = x.shape
        f.write(struct.pack('I', len(shape)))
        for dim in shape:
            f.write(struct.pack('I', dim))
        # Write tensor data
        f.write(x.detach().numpy().tobytes())
        # Write other parameters
        for param in other_params:
            f.write(struct.pack('f', param))
    send_to_server_time += time.time() - tempstart

def read_tensor_from_mmap(file_path):
    global read_from_server_time
    tempstart = time.time()
    with open(file_path, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        # Read the number of dimensions
        num_dims = struct.unpack('I', mm.read(4))[0]
        # Read each dimension size
        shape = []
        for _ in range(num_dims):
            shape.append(struct.unpack('I', mm.read(4))[0])
        shape = tuple(shape)
        writable_data = np.frombuffer(mm.read(), dtype=np.float32).copy()
        tensor = torch.from_numpy(writable_data).view(shape)
        # tensor = torch.frombuffer(mm.read(), dtype=torch.float32).view(shape)
        mm.close()
    read_from_server_time += time.time() - tempstart
    return tensor

def signal_server_ready_conv(status_path):
    with open(status_path, 'w') as f:
        f.write('ready_conv')

def signal_server_ready_linear(status_path):
    with open(status_path, 'w') as f:
        f.write('ready_linear')

def wait_for_server_response(status_path):
    while True:
        if os.path.exists(status_path):
            with open(status_path, 'r') as f:
                if f.read() == 'processed':
                    break
        time.sleep(0.0001)

class CustomConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(CustomConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

    def forward(self, x):
        # x = x+x-x+x-x
        data_tuple = (x, self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.bias)
        write_data_to_mmap(file_path_1, data_tuple)
        signal_server_ready_conv(status_path)
        print("Tensor written to shared memory. Waiting for the server to process it.")
        wait_for_server_response(status_path)
        new_x = read_tensor_from_mmap(file_path_2).to(device)
        os.remove(file_path_2)
        new_x.requires_grad_()
        print("New tensor received from the server:")
        print("Conv2d layer: --- %s ---" % (time.time() - start_time))
        return new_x
        # return new_x+new_x


class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(CustomLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

    def forward(self, x):
        # x = x+x-x+x-x
        data_tuple = (x, self.in_features, self.out_features, self.bias)
        write_data_to_mmap(file_path_1, data_tuple)
        signal_server_ready_linear(status_path)
        print("Tensor written to shared memory. Waiting for the server to process it.")
        wait_for_server_response(status_path)
        new_x = read_tensor_from_mmap(file_path_2).to(device)
        os.remove(file_path_2)
        new_x.requires_grad_()
        print("New tensor received from the server:")
        print("Linear layer: --- %s ---" % (time.time() - start_time))
        return new_x
        # return new_x+new_x

class CustomBatchNorm2d(nn.BatchNorm2d):
    def forward(self, x):
        global bn_time
        tempstart = time.time()
        # normalized_x = super(CustomBatchNorm2d, self).forward(x)
        normalized_x = x
        bn_time += time.time() - tempstart
        return normalized_x

class CustomReLU(nn.ReLU):
    def forward(self, x):
        global relu_time
        tempstart = time.time()
        activated_x = super(CustomReLU, self).forward(x)
        # print("relu layer: --- %s ---" % (datetime.now()))
        relu_time += time.time() - tempstart
        return activated_x

class CustomMaxPool2d(nn.MaxPool2d):
    def forward(self, x):
        global pool_time
        tempstart = time.time()
        pooled_x = super(CustomMaxPool2d, self).forward(x)
        # print("max pool layer: --- %s ---" % (datetime.now()))
        pool_time += time.time() - tempstart
        return pooled_x

class VGG19(nn.Module):
    def __init__(self,in_dim,num_classes):
        super().__init__()
        self.features = nn.Sequential(
             CustomConv2d(in_dim,64,kernel_size=3,padding=1),
             CustomBatchNorm2d(64),
             CustomReLU(),
             CustomConv2d(64,64,kernel_size=3,padding=1),
             CustomBatchNorm2d(64),
             CustomReLU(),
             CustomMaxPool2d(kernel_size=2,stride=2),
 
             CustomConv2d(64,128,kernel_size=3,padding=1),
             CustomBatchNorm2d(128),
             CustomReLU(),
             CustomConv2d(128, 128, kernel_size=3, padding=1),
             CustomBatchNorm2d(128),
             CustomReLU(),
             CustomMaxPool2d(kernel_size=2,stride=2),
 
             CustomConv2d(128, 256, kernel_size=3, padding=1),
             CustomBatchNorm2d(256),
             CustomReLU(),
             CustomConv2d(256, 256, kernel_size=3, padding=1),
             CustomBatchNorm2d(256),
             CustomReLU(),
             CustomConv2d(256, 256, kernel_size=3, padding=1),
             CustomBatchNorm2d(256),
             CustomReLU(),
             CustomConv2d(256, 256, kernel_size=3, padding=1),
             CustomBatchNorm2d(256),
             CustomReLU(),
             CustomMaxPool2d(kernel_size=2,stride=2),
            
             CustomConv2d(256, 512, kernel_size=3, padding=1),
             CustomBatchNorm2d(512),
             CustomReLU(),
             CustomConv2d(512, 512, kernel_size=3, padding=1),
             CustomBatchNorm2d(512),
             CustomReLU(),
             CustomConv2d(512, 512, kernel_size=3, padding=1),
             CustomBatchNorm2d(512),
             CustomReLU(),
             CustomConv2d(512, 512, kernel_size=3, padding=1),
             CustomBatchNorm2d(512),
             CustomReLU(),
             CustomMaxPool2d(kernel_size=2,stride=2),
            
             CustomConv2d(512, 512, kernel_size=3, padding=1),
             CustomBatchNorm2d(512),
             CustomReLU(),
             CustomConv2d(512, 512, kernel_size=3, padding=1),
             CustomBatchNorm2d(512),
             CustomReLU(),
             CustomConv2d(512, 512, kernel_size=3, padding=1),
             CustomBatchNorm2d(512),
             CustomReLU(),
             CustomConv2d(512, 512, kernel_size=3, padding=1),
             CustomBatchNorm2d(512),
             CustomReLU(),
             CustomMaxPool2d(kernel_size=2,stride=2)
         )
 
        self.classifier = nn.Sequential(
             CustomLinear(512*7*7,4096),
             CustomReLU(),
             nn.Dropout(0.5),
 
             CustomLinear(4096,4096),
             CustomReLU(),
             nn.Dropout(0.5),
            
             CustomLinear(4096,num_classes)
         )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x


net = VGG19(3,10)
net.to(device)

num_images = 64
num_batches = num_images // batch_size

learning_rate = 1e-2
num_epoches = 1
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr = learning_rate,momentum=0.9)
net.train()
# start_time = time.time(

for epoch in range(num_epoches):
    batch_count = 0
    for i, (images, labels) in enumerate(train_loader):
        if batch_count >= num_batches:
            break  # Stop the loop after processing num_batches batches
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        start_time = time.time()
        outputs = net(images)
        tempstart = time.time()
        loss = criterion(outputs, labels)
        loss.backward()
        # print("backward: --- %s seconds ---" % (time.time() - start_time))
        backward_time = time.time() - tempstart
        optimizer.step()
        batch_count += 1
        print('batch #', batch_count, 'finished')
        break
total_time = time.time() - start_time

print("BatchNorm time: --- %s seconds ---" % (bn_time))
print("Pooling time: --- %s seconds ---" % (pool_time))
print("ReLU time: --- %s seconds ---" % (relu_time))
print("Backward time: --- %s seconds ---" % (backward_time))
print("TEE->GPU client side time: --- %s seconds ---" % (send_to_server_time))
print("GPU->TEE client side time: --- %s seconds ---" % (read_from_server_time))
print("Total time: --- %s seconds ---" % (total_time))
# print("All finished: --- %s seconds ---" % (time.time() - start_time))

# os.remove(status_path)