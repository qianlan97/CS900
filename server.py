import torch
import time
from torch import nn
import torch.nn.functional as F
import mmap
import os
import struct
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"running on {device}")
count = 0
file_path_1 = "shared_mmap_1.dat"
file_path_2 = "shared_mmap_2.dat"
status_path = "status.dat"
read_from_client_time = 0
send_to_client_time = 0
computation_time = 0
weight_generation_time = 0

def read_conv_from_mmap(file_path):
    global read_from_client_time
    tempstart = time.time()
    with open(file_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        # Read the number of dimensions
        num_dims = struct.unpack('I', mm.read(4))[0]
        # Read each dimension size
        shape = []
        for _ in range(num_dims):
            shape.append(struct.unpack('I', mm.read(4))[0])
        shape = tuple(shape)
        # Read tensor data
        writable_data = np.frombuffer(mm.read(torch.prod(torch.tensor(shape)) * 4), dtype=np.float32).copy()
        tensor = torch.from_numpy(writable_data).view(shape)
        # tensor = torch.frombuffer(mm.read(torch.prod(torch.tensor(shape)) * 4), dtype=torch.float32).view(shape)
        # Read other parameters (assuming they are all float32)
        in_channels = int(struct.unpack('f', mm.read(4))[0])
        out_channels = int(struct.unpack('f', mm.read(4))[0])
        kernel_size = int(struct.unpack('f', mm.read(4))[0])
        stride = int(struct.unpack('f', mm.read(4))[0])
        padding = int(struct.unpack('f', mm.read(4))[0])
        bias = struct.unpack('f', mm.read(4))[0]
        bias = False if bias == 0 else True
        mm.close()
    read_from_client_time += time.time() - tempstart
    return tensor, in_channels, out_channels, kernel_size, stride, padding, bias

def read_linear_from_mmap(file_path):
    global read_from_client_time
    tempstart = time.time()
    with open(file_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        # Read the number of dimensions
        num_dims = struct.unpack('I', mm.read(4))[0]
        # Read each dimension size
        shape = []
        for _ in range(num_dims):
            shape.append(struct.unpack('I', mm.read(4))[0])
        shape = tuple(shape)
        # Read tensor data
        writable_data = np.frombuffer(mm.read(torch.prod(torch.tensor(shape)) * 4), dtype=np.float32).copy()
        tensor = torch.from_numpy(writable_data).view(shape)
        # tensor = torch.frombuffer(mm.read(torch.prod(torch.tensor(shape)) * 4), dtype=torch.float32).view(shape)
        # Read other parameters (assuming they are all float32)
        in_features = int(struct.unpack('f', mm.read(4))[0])
        out_features = int(struct.unpack('f', mm.read(4))[0])
        bias = struct.unpack('f', mm.read(4))[0]
        bias = False if bias == 0 else True
        mm.close()
    read_from_client_time += time.time() - tempstart
    return tensor, in_features, out_features, bias

def write_tensor_to_mmap(file_path, tensor):
    global send_to_client_time
    tempstart = time.time()
    with open(file_path, "wb") as f:
        # Serialize shape information
        x = tensor
        shape = x.shape
        f.write(struct.pack('I', len(shape)))
        for dim in shape:
            f.write(struct.pack('I', dim))
        # Write tensor data
        # f.write(x.detach().numpy().tobytes())
        data = x.detach().numpy().tobytes()
        expected_size = torch.prod(torch.tensor(x.shape)) * 4
        print(shape)
        print(expected_size)
        print(len(data))
        assert len(data) == expected_size, f"Expected to write {expected_size} bytes, but writing {len(data)} bytes."
        f.write(data)
    send_to_client_time += time.time() - tempstart
        

def wait_for_client_data(status_path):
    while True:
        if os.path.exists(status_path):
            with open(status_path, 'r') as f:
                temp = f.read()
                # print(temp)
                if temp == 'ready_conv':
                    return 1
                elif temp == 'ready_linear':
                    return 2
        time.sleep(0.0001)

def signal_client_processed(status_path):
    with open(status_path, 'w') as f:
        f.write('processed')


while True:
    print("Server is listening...")
    optype = wait_for_client_data(status_path)
    print("optype is:", optype)
    if optype == 1:
        # Read the tensor from the memory-mapped file
        x, in_channels, out_channels, kernel_size, stride, padding, bias = read_conv_from_mmap(file_path_1)
        print("Conv data received from shared memory:")
        os.remove(file_path_1)

        tempstart = time.time()
        x = x.to(device)
        temp_weight_start = time.time()
        weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size).to(device)

        running_mean = torch.zeros(out_channels).to(device)
        running_var = torch.ones(out_channels).to(device)
        bn_weight = nn.Parameter(torch.ones(out_channels)).to(device)
        bn_bias = nn.Parameter(torch.zeros(out_channels)).to(device)

        weight_generation_time += time.time() - temp_weight_start
        if bias:
            bias = torch.randn(out_channels).to('cuda')
            print("Conv bias re-randomed.")
        else:
            bias = None
        x = F.conv2d(x, weight, bias, stride, padding)

        x = F.batch_norm(
            x, running_mean, running_var, bn_weight, bn_bias,
            training=True, momentum=0.1, eps=1e-5
        ).cpu()
        
        print("conv layer time: --- %s seconds ---" % (time.time() - tempstart))
        computation_time += time.time() - tempstart

        # Write the processed tensor back to shared memory
        write_tensor_to_mmap(file_path_2, x)
        signal_client_processed(status_path)
        print("Processed tensor written back to shared memory.")
        print("count:", count)
        count +=1
        time.sleep(0.0001)
    elif optype == 2:
        # Read the tensor from the memory-mapped file
        x, in_features, out_features, bias = read_linear_from_mmap(file_path_1)
        os.remove(file_path_1)
        print("Linear data received from shared memory:")
        tempstart = time.time()
        x = x.to(device)
        temp_weight_start = time.time()
        weight = torch.randn(out_features, in_features).to(device)
        weight_generation_time += time.time() - temp_weight_start
        if bias:
            bias = torch.randn(out_features).to('cuda')
            print("Linear bias re-randomed.")
        else:
            bias = None
        x = F.linear(x, weight, bias).cpu()
        print("linear layer time: --- %s seconds ---" % (time.time() - tempstart))
        computation_time += time.time() - tempstart

        # Write the processed tensor back to shared memory
        write_tensor_to_mmap(file_path_2, x)
        signal_client_processed(status_path)
        print("Processed tensor written back to shared memory.")
        print("count:", count)
        count +=1
        if count % 19 == 0:
            print("A batch is computed")
            print("TEE->GPU server side time: --- %s seconds ---" % (read_from_client_time))
            print("GPU->TEE server side time: --- %s seconds ---" % (send_to_client_time))
            print("Weight re-random time: --- %s seconds ---" % (weight_generation_time))
            print("GPU computation time: --- %s seconds ---" % (computation_time - weight_generation_time))
            read_from_client_time = 0
            send_to_client_time = 0
            computation_time = 0
            weight_generation_time = 0
        time.sleep(0.0001)

