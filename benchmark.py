import torch
import time
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


def wait_for_server_response(status_path):
    while True:
        if os.path.exists(status_path):
            with open(status_path, 'r') as f:
                if f.read() == 'processed':
                    break
        time.sleep(0.0001)

import torch
import torch.nn as nn
import time

def test_convolution_speed(input_size, in_channels, out_channels, kernel_size=3):
    # Define the convolution layer
    # conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)

    # Create a random tensor with the specified input size
    input_tensor = torch.randn(1, in_channels, input_size, input_size)

    # Start the timer
    start_time = time.time()
    data_tuple = (input_tensor, in_channels, out_channels, kernel_size)
    write_data_to_mmap(file_path_1, data_tuple)
    signal_server_ready_conv(status_path)
    wait_for_server_response(status_path)
    new_x = read_tensor_from_mmap(file_path_2).to(device)
    os.remove(file_path_2)

    # Perform the convolution operation
    # output = conv_layer(input_tensor)

    # End the timer
    end_time = time.time()

    # Calculate and print the time taken
    time_taken = end_time - start_time
    print(f"Time taken for input size {input_size}x{input_size} with {in_channels} channels: {time_taken:.6f} seconds")

# Test with different input sizes and channels
test_cases = [
    (224, 64, 128),
    (224, 64, 128),
    (112, 128, 256),
    (56, 256, 512),
    (28, 512, 1024)  # Increased out_channels for demonstration
]

for input_size, in_channels, out_channels in test_cases:
    test_convolution_speed(input_size, in_channels, out_channels)
