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
from datetime import datetime
import mmap
import os
import struct
import math

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
layer_times = {}

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

# def write_data_to_mmap(file_path, data_tuple):
#     global send_to_server_time
#     tempstart = time.time()
#     with open(file_path, "wb") as f:
#         # Serialize shape information
#         x, *other_params = data_tuple
#         shape = x.shape
#         f.write(struct.pack('I', len(shape)))
#         for dim in shape:
#             f.write(struct.pack('I', dim))
#         # Write tensor data
#         f.write(x.detach().numpy().tobytes())
#         # Write other parameters
#         for param in other_params:
#             f.write(struct.pack('f', param))
#     send_to_server_time += time.time() - tempstart

# def read_tensor_from_mmap(file_path):
#     global read_from_server_time
#     tempstart = time.time()
#     with open(file_path, "r+b") as f:
#         mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
#         # Read the number of dimensions
#         num_dims = struct.unpack('I', mm.read(4))[0]
#         # Read each dimension size
#         shape = []
#         for _ in range(num_dims):
#             shape.append(struct.unpack('I', mm.read(4))[0])
#         shape = tuple(shape)
#         writable_data = np.frombuffer(mm.read(), dtype=np.float32).copy()
#         tensor = torch.from_numpy(writable_data).view(shape)
#         # tensor = torch.frombuffer(mm.read(), dtype=torch.float32).view(shape)
#         mm.close()
#     read_from_server_time += time.time() - tempstart
#     return tensor
def write_data_to_mmap(file_path, data_tuple):
    global send_to_server_time
    tempstart = time.time()
    with open(file_path, "wb") as f:
        for item in data_tuple:
            if item is None:
                f.write(struct.pack('B', 0))
            elif isinstance(item, (int, float)):
                f.write(struct.pack('B', 1))
                f.write(struct.pack('f', item))
            elif isinstance(item, (tuple, list)):
                f.write(struct.pack('B', 2))
                f.write(struct.pack('I', len(item)))
                for subitem in item:
                    f.write(struct.pack('f', subitem))
            elif isinstance(item, torch.Tensor):
                f.write(struct.pack('B', 3))
                item_shape = item.shape
                f.write(struct.pack('I', len(item_shape)))
                for dim in item_shape:
                    f.write(struct.pack('I', dim))
                f.write(item.detach().numpy().tobytes())
            else:
                raise ValueError(f"Unsupported data type: {type(item)}")
    send_to_server_time += time.time() - tempstart

def read_tensor_from_mmap(file_path):
    global read_from_server_time
    tempstart = time.time()
    with open(file_path, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        num_dims = struct.unpack('I', mm.read(4))[0]
        shape = []
        for _ in range(num_dims):
            shape.append(struct.unpack('I', mm.read(4))[0])
        shape = tuple(shape)
        writable_data = np.frombuffer(mm.read(), dtype=np.float32).copy()
        tensor = torch.from_numpy(writable_data).view(shape)
        mm.close()
    read_from_server_time += time.time() - tempstart
    return tensor

def signal_server_ready_conv(status_path):
    with open(status_path, 'w') as f:
        f.write('ready_conv')

def signal_server_ready_linear(status_path):
    with open(status_path, 'w') as f:
        f.write('ready_linear')

def signal_server_ready_bn(status_path):
    with open(status_path, 'w') as f:
        f.write('ready_bn')

def signal_server_ready_relu(status_path):
    with open(status_path, 'w') as f:
        f.write('ready_relu')

def signal_server_ready_maxpool(status_path):
    with open(status_path, 'w') as f:
        f.write('ready_maxpool')

def wait_for_server_response(status_path):
    while True:
        if os.path.exists(status_path):
            with open(status_path, 'r') as f:
                if f.read() == 'processed':
                    break
        time.sleep(0.0001)

# class CustomConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
#         super(CustomConv2d, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.bias = bias

#     def forward(self, x):
#         # x = x+x-x+x-x
#         data_tuple = (x, self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.bias)
#         write_data_to_mmap(file_path_1, data_tuple)
#         signal_server_ready_conv(status_path)
#         print("Tensor written to shared memory. Waiting for the server to process it.")
#         wait_for_server_response(status_path)
#         new_x = read_tensor_from_mmap(file_path_2).to(device)
#         os.remove(file_path_2)
#         new_x.requires_grad_()
#         print("New tensor received from the server:")
#         print("Conv2d layer: --- %s ---" % (time.time() - start_time))
#         return new_x
#         # return new_x+new_x


# class CustomLinear(nn.Module):
#     def __init__(self, in_features, out_features, bias=True):
#         super(CustomLinear, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.bias = bias

#     def forward(self, x):
#         # x = x+x-x+x-x
#         data_tuple = (x, self.in_features, self.out_features, self.bias)
#         write_data_to_mmap(file_path_1, data_tuple)
#         signal_server_ready_linear(status_path)
#         print("Tensor written to shared memory. Waiting for the server to process it.")
#         wait_for_server_response(status_path)
#         new_x = read_tensor_from_mmap(file_path_2).to(device)
#         os.remove(file_path_2)
#         new_x.requires_grad_()
#         print("New tensor received from the server:")
#         print("Linear layer: --- %s ---" % (time.time() - start_time))
#         return new_x
#         # return new_x+new_x

# class CustomBatchNorm2d(nn.BatchNorm2d):
#     def forward(self, x):
#         global bn_time
#         tempstart = time.time()
#         # normalized_x = super(CustomBatchNorm2d, self).forward(x)
#         normalized_x = x
#         bn_time += time.time() - tempstart
#         return normalized_x

# class CustomReLU(nn.ReLU):
#     def forward(self, x):
#         global relu_time
#         tempstart = time.time()
#         activated_x = super(CustomReLU, self).forward(x)
#         # print("relu layer: --- %s ---" % (datetime.now()))
#         relu_time += time.time() - tempstart
#         return activated_x

# class CustomMaxPool2d(nn.MaxPool2d):
#     def forward(self, x):
#         global pool_time
#         tempstart = time.time()
#         pooled_x = super(CustomMaxPool2d, self).forward(x)
#         # print("max pool layer: --- %s ---" % (datetime.now()))
#         pool_time += time.time() - tempstart
#         return pooled_x
        

class CustomConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, layer_id=None):
        super(CustomConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.layer_id = layer_id

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        start_time = time.time()
        data_tuple = (x, self.weight, self.bias, self.stride, self.padding)
        write_data_to_mmap(file_path_1, data_tuple)
        signal_server_ready_conv(status_path)
        wait_for_server_response(status_path)
        new_x = read_tensor_from_mmap(file_path_2).to(device)
        os.remove(file_path_2)
        new_x.requires_grad_()
        layer_times[self.layer_id] = time.time() - start_time
        print("Layer",self.layer_id, "(conv) is done")
        return new_x

class CustomBatchNorm2d(nn.Module):
    def __init__(self, num_features, layer_id=None):
        super(CustomBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.layer_id = layer_id

        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.uniform_()
        self.bias.data.zero_()

    def forward(self, x):
        start_time = time.time()
        data_tuple = (x, self.weight, self.bias)
        write_data_to_mmap(file_path_1, data_tuple)
        signal_server_ready_bn(status_path)
        wait_for_server_response(status_path)
        new_x = read_tensor_from_mmap(file_path_2).to(device)
        os.remove(file_path_2)
        new_x.requires_grad_()
        layer_times[self.layer_id] = time.time() - start_time
        print("Layer",self.layer_id, "(bn) is done")
        return new_x

class CustomReLU(nn.Module):
    def __init__(self, layer_id=None):
        super(CustomReLU, self).__init__()
        self.layer_id = layer_id

    def forward(self, x):
        start_time = time.time()
        data_tuple = (x,)
        write_data_to_mmap(file_path_1, data_tuple)
        signal_server_ready_relu(status_path)
        wait_for_server_response(status_path)
        new_x = read_tensor_from_mmap(file_path_2).to(device)
        os.remove(file_path_2)
        new_x.requires_grad_()
        layer_times[self.layer_id] = time.time() - start_time
        print("Layer",self.layer_id, "(relu) is done")
        return new_x

class CustomMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, layer_id=None):
        super(CustomMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.layer_id = layer_id

    def forward(self, x):
        start_time = time.time()
        data_tuple = (x, self.kernel_size, self.stride, self.padding, self.dilation)
        write_data_to_mmap(file_path_1, data_tuple)
        signal_server_ready_maxpool(status_path)
        wait_for_server_response(status_path)
        new_x = read_tensor_from_mmap(file_path_2).to(device)
        os.remove(file_path_2)
        new_x.requires_grad_()
        layer_times[self.layer_id] = time.time() - start_time
        print("Layer",self.layer_id, "(maxpool) is done")
        return new_x

class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, layer_id=None):
        super(CustomLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layer_id = layer_id

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        start_time = time.time()
        data_tuple = (x, self.weight, self.bias)
        write_data_to_mmap(file_path_1, data_tuple)
        signal_server_ready_linear(status_path)
        wait_for_server_response(status_path)
        new_x = read_tensor_from_mmap(file_path_2).to(device)
        os.remove(file_path_2)
        new_x.requires_grad_()
        layer_times[self.layer_id] = time.time() - start_time
        print("Layer",self.layer_id, "(linear) is done")
        return new_x


class VGG19(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            CustomConv2d(in_dim, 64, kernel_size=3, padding=1, layer_id='Layer 1'),
            CustomBatchNorm2d(64, layer_id='Layer 2'),
            CustomReLU(layer_id='Layer 3'),
            CustomConv2d(64, 64, kernel_size=3, padding=1, layer_id='Layer 4'),
            CustomBatchNorm2d(64, layer_id='Layer 5'),
            CustomReLU(layer_id='Layer 6'),
            CustomMaxPool2d(kernel_size=2, stride=2, layer_id='Layer 7'),

            CustomConv2d(64, 128, kernel_size=3, padding=1, layer_id='Layer 8'),
            CustomBatchNorm2d(128, layer_id='Layer 9'),
            CustomReLU(layer_id='Layer 10'),
            CustomConv2d(128, 128, kernel_size=3, padding=1, layer_id='Layer 11'),
            CustomBatchNorm2d(128, layer_id='Layer 12'),
            CustomReLU(layer_id='Layer 13'),
            CustomMaxPool2d(kernel_size=2, stride=2, layer_id='Layer 14'),

            CustomConv2d(128, 256, kernel_size=3, padding=1, layer_id='Layer 15'),
            CustomBatchNorm2d(256, layer_id='Layer 16'),
            CustomReLU(layer_id='Layer 17'),
            CustomConv2d(256, 256, kernel_size=3, padding=1, layer_id='Layer 18'),
            CustomBatchNorm2d(256, layer_id='Layer 19'),
            CustomReLU(layer_id='Layer 20'),
            CustomConv2d(256, 256, kernel_size=3, padding=1, layer_id='Layer 21'),
            CustomBatchNorm2d(256, layer_id='Layer 22'),
            CustomReLU(layer_id='Layer 23'),
            CustomConv2d(256, 256, kernel_size=3, padding=1, layer_id='Layer 24'),
            CustomBatchNorm2d(256, layer_id='Layer 25'),
            CustomReLU(layer_id='Layer 26'),
            CustomMaxPool2d(kernel_size=2, stride=2, layer_id='Layer 27'),

            CustomConv2d(256, 512, kernel_size=3, padding=1, layer_id='Layer 28'),
            CustomBatchNorm2d(512, layer_id='Layer 29'),
            CustomReLU(layer_id='Layer 30'),
            CustomConv2d(512, 512, kernel_size=3, padding=1, layer_id='Layer 31'),
            CustomBatchNorm2d(512, layer_id='Layer 32'),
            CustomReLU(layer_id='Layer 33'),
            CustomConv2d(512, 512, kernel_size=3, padding=1, layer_id='Layer 34'),
            CustomBatchNorm2d(512, layer_id='Layer 35'),
            CustomReLU(layer_id='Layer 36'),
            CustomConv2d(512, 512, kernel_size=3, padding=1, layer_id='Layer 37'),
            CustomBatchNorm2d(512, layer_id='Layer 38'),
            CustomReLU(layer_id='Layer 39'),
            CustomMaxPool2d(kernel_size=2, stride=2, layer_id='Layer 40'),

            CustomConv2d(512, 512, kernel_size=3, padding=1, layer_id='Layer 41'),
            CustomBatchNorm2d(512, layer_id='Layer 42'),
            CustomReLU(layer_id='Layer 43'),
            CustomConv2d(512, 512, kernel_size=3, padding=1, layer_id='Layer 44'),
            CustomBatchNorm2d(512, layer_id='Layer 45'),
            CustomReLU(layer_id='Layer 46'),
            CustomConv2d(512, 512, kernel_size=3, padding=1, layer_id='Layer 47'),
            CustomBatchNorm2d(512, layer_id='Layer 48'),
            CustomReLU(layer_id='Layer 49'),
            CustomConv2d(512, 512, kernel_size=3, padding=1, layer_id='Layer 50'),
            CustomBatchNorm2d(512, layer_id='Layer 51'),
            CustomReLU(layer_id='Layer 52'),
            CustomMaxPool2d(kernel_size=2, stride=2, layer_id='Layer 53')
        )

        self.classifier = nn.Sequential(
            CustomLinear(512 * 7 * 7, 4096, layer_id='Layer 54'),
            CustomReLU(layer_id='Layer 55'),
            nn.Dropout(0.5),
            CustomLinear(4096, 4096, layer_id='Layer 57'),
            CustomReLU(layer_id='Layer 58'),
            nn.Dropout(0.5),
            CustomLinear(4096, num_classes, layer_id='Layer 60')
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
print("Layer-wise time costs:")
for layer_id, time_cost in layer_times.items():
    print(f"{layer_id}: {time_cost:.4f} seconds")
