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

def read_tensor_from_mmap(file_path):
    global read_from_client_time
    tempstart = time.time()
    with open(file_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        num_dims = struct.unpack('I', mm.read(4))[0]
        shape = []
        for _ in range(num_dims):
            shape.append(struct.unpack('I', mm.read(4))[0])
        shape = tuple(shape)
        writable_data = np.frombuffer(mm.read(torch.prod(torch.tensor(shape)) * 4), dtype=np.float32).copy()
        tensor = torch.from_numpy(writable_data).view(shape)
        mm.close()
    read_from_client_time += time.time() - tempstart
    return tensor

def read_parameter_from_mmap(mm):
    param_type = struct.unpack('B', mm.read(1))[0]
    if param_type == 0:
        return None
    elif param_type == 1:
        return struct.unpack('f', mm.read(4))[0]
    elif param_type == 2:
        length = struct.unpack('I', mm.read(4))[0]
        param = []
        for _ in range(length):
            param.append(struct.unpack('f', mm.read(4))[0])
        return tuple(param)
    elif param_type == 3:
        num_dims = struct.unpack('I', mm.read(4))[0]
        shape = []
        for _ in range(num_dims):
            shape.append(struct.unpack('I', mm.read(4))[0])
        shape = tuple(shape)
        writable_data = np.frombuffer(mm.read(torch.prod(torch.tensor(shape)) * 4), dtype=np.float32).copy()
        tensor = torch.from_numpy(writable_data).view(shape)
        return tensor
    elif param_type == 4:
        return struct.unpack('f', mm.read(4))[0]
    else:
        raise ValueError(f"Unsupported parameter type: {param_type}")

def read_conv_from_mmap(file_path):
    global read_from_client_time
    tempstart = time.time()
    with open(file_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        x = read_parameter_from_mmap(mm)
        weight = read_parameter_from_mmap(mm)
        bias = read_parameter_from_mmap(mm)
        stride = read_parameter_from_mmap(mm)
        padding = read_parameter_from_mmap(mm)
        mm.close()
    read_from_client_time += time.time() - tempstart

    if stride is None:
        stride = (1, 1)
    elif isinstance(stride, (float, int)):
        stride = (int(stride), int(stride))
    else:
        stride = tuple(stride)

    if padding is None:
        padding = (0, 0)
    elif isinstance(padding, (float, int)):
        padding = (int(padding), int(padding))
    else:
        padding = tuple(padding)

    return x, weight, bias, stride, padding

def read_bn_from_mmap(file_path):
    global read_from_client_time
    tempstart = time.time()
    with open(file_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        x = read_parameter_from_mmap(mm)
        weight = read_parameter_from_mmap(mm)
        bias = read_parameter_from_mmap(mm)
        mm.close()
    read_from_client_time += time.time() - tempstart
    return x, weight, bias

def read_relu_from_mmap(file_path):
    global read_from_client_time
    tempstart = time.time()
    with open(file_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        x = read_parameter_from_mmap(mm)
        mm.close()
    read_from_client_time += time.time() - tempstart
    return x

def read_maxpool_from_mmap(file_path):
    global read_from_client_time
    tempstart = time.time()
    with open(file_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        x = read_parameter_from_mmap(mm)
        kernel_size = read_parameter_from_mmap(mm)
        stride = read_parameter_from_mmap(mm)
        padding = read_parameter_from_mmap(mm)
        dilation = read_parameter_from_mmap(mm)
        mm.close()
    read_from_client_time += time.time() - tempstart

    if isinstance(kernel_size, (float, int)):
        kernel_size = (int(kernel_size), int(kernel_size))
    else:
        kernel_size = tuple(kernel_size)

    if stride is None:
        stride = kernel_size
    elif isinstance(stride, (float, int)):
        stride = (int(stride), int(stride))
    else:
        stride = tuple(stride)

    if padding is None:
        padding = (0, 0)
    elif isinstance(padding, (float, int)):
        padding = (int(padding), int(padding))
    else:
        padding = tuple(padding)

    if dilation is None:
        dilation = (1, 1)
    elif isinstance(dilation, (float, int)):
        dilation = (int(dilation), int(dilation))
    else:
        dilation = tuple(dilation)

    return x, kernel_size, stride, padding, dilation

def read_linear_from_mmap(file_path):
    global read_from_client_time
    tempstart = time.time()
    with open(file_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        x = read_parameter_from_mmap(mm)
        weight = read_parameter_from_mmap(mm)
        bias = read_parameter_from_mmap(mm)
        mm.close()
    read_from_client_time += time.time() - tempstart
    return x, weight, bias

def read_dropout_from_mmap(file_path):
    global read_from_client_time
    tempstart = time.time()
    with open(file_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        x = read_parameter_from_mmap(mm)
        p = read_parameter_from_mmap(mm)
        mm.close()
    read_from_client_time += time.time() - tempstart
    return x, p

def write_tensor_to_mmap(file_path, tensor):
    global send_to_client_time
    tempstart = time.time()
    with open(file_path, "wb") as f:
        x = tensor
        shape = x.shape
        f.write(struct.pack('I', len(shape)))
        for dim in shape:
            f.write(struct.pack('I', dim))
        data = x.detach().numpy().tobytes()
        expected_size = torch.prod(torch.tensor(x.shape)) * 4
        assert len(data) == expected_size, f"Expected to write {expected_size} bytes, but writing {len(data)} bytes."
        f.write(data)
    send_to_client_time += time.time() - tempstart

def wait_for_client_data(status_path):
    while True:
        if os.path.exists(status_path):
            with open(status_path, 'r') as f:
                temp = f.read()
                if temp == 'ready_conv':
                    return 1
                elif temp == 'ready_linear':
                    return 2
                elif temp == 'ready_bn':
                    return 3
                elif temp == 'ready_relu':
                    return 4
                elif temp == 'ready_maxpool':
                    return 5
                elif temp == 'ready_dropout':
                    return 6
        time.sleep(0.0001)

def signal_client_processed(status_path):
    with open(status_path, 'w') as f:
        f.write('processed')

while True:
    print("Server is listening...")
    optype = wait_for_client_data(status_path)
    print("optype is:", optype)
    if optype == 1:
        x, weight, bias, stride, padding = read_conv_from_mmap(file_path_1)
        print("Conv data received from shared memory:")
        os.remove(file_path_1)     
        if x is None or weight is None:
            print("Received None data for Conv2d layer. Skipping computation.")
            signal_client_processed(status_path)
            continue

        tempstart = time.time()
        x = x.to(device)
        weight = weight.to(device)
        stride = tuple(stride)
        padding = tuple(padding)
        bias = bias.to(device) if bias is not None else None
        x = F.conv2d(x, weight, bias, stride, padding)
        x = x.cpu()

        print("Conv layer time: --- %s seconds ---" % (time.time() - tempstart))
        computation_time += time.time() - tempstart

        write_tensor_to_mmap(file_path_2, x)
        signal_client_processed(status_path)
        print("Processed tensor written back to shared memory.")
        print("count:", count)
        count += 1
        time.sleep(0.0001)
    elif optype == 2:
        x, weight, bias = read_linear_from_mmap(file_path_1)
        os.remove(file_path_1)
        print("Linear data received from shared memory:")

        if x is None or weight is None:
            print("Received None data for Linear layer. Skipping computation.")
            signal_client_processed(status_path)
            continue

        tempstart = time.time()
        x = x.to(device)
        weight = weight.to(device)
        bias = bias.to(device) if bias is not None else None
        x = F.linear(x, weight, bias).cpu()

        print("Linear layer time: --- %s seconds ---" % (time.time() - tempstart))
        computation_time += time.time() - tempstart

        write_tensor_to_mmap(file_path_2, x)
        signal_client_processed(status_path)
        print("Processed tensor written back to shared memory.")
        print("count:", count)
        count += 1
        time.sleep(0.0001)
    elif optype == 3:
        x, weight, bias = read_bn_from_mmap(file_path_1)
        print("BatchNorm2d data received from shared memory:")
        os.remove(file_path_1)

        if x is None or weight is None or bias is None:
            print("Received None data for BatchNorm2d layer. Skipping computation.")
            signal_client_processed(status_path)
            continue

        tempstart = time.time()
        x = x.to(device)
        weight = weight.to(device)
        bias = bias.to(device)
        x = F.batch_norm(x, torch.zeros_like(weight), torch.ones_like(weight),
                         weight, bias, training=True, momentum=0.1, eps=1e-5).cpu()

        print("BatchNorm2d layer time: --- %s seconds ---" % (time.time() - tempstart))
        computation_time += time.time() - tempstart

        write_tensor_to_mmap(file_path_2, x)
        signal_client_processed(status_path)
        print("Processed tensor written back to shared memory.")
        print("count:", count)
        count += 1
        time.sleep(0.0001)
    elif optype == 4:
        x = read_relu_from_mmap(file_path_1)
        print("ReLU data received from shared memory:")
        os.remove(file_path_1)

        if x is None:
            print("Received None data for ReLU layer. Skipping computation.")
            signal_client_processed(status_path)
            continue

        tempstart = time.time()
        x = x.to(device)
        x = F.relu(x).cpu()

        print("ReLU layer time: --- %s seconds ---" % (time.time() - tempstart))
        computation_time += time.time() - tempstart

        write_tensor_to_mmap(file_path_2, x)
        signal_client_processed(status_path)
        print("Processed tensor written back to shared memory.")
        print("count:", count)
        count += 1
        time.sleep(0.0001)
    elif optype == 5:
        x, kernel_size, stride, padding, dilation = read_maxpool_from_mmap(file_path_1)
        print("MaxPool2d data received from shared memory:")
        os.remove(file_path_1)

        if x is None:
            print("Received None data for MaxPool2d layer. Skipping computation.")
            signal_client_processed(status_path)
            continue

        tempstart = time.time()
        x = x.to(device)
        kernel_size = tuple(kernel_size)
        stride = tuple(stride) if stride is not None else None
        padding = tuple(padding)
        dilation = tuple(dilation)
        x = F.max_pool2d(x, kernel_size, stride, padding, dilation).cpu()

        print("MaxPool2d layer time: --- %s seconds ---" % (time.time() - tempstart))
        computation_time += time.time() - tempstart

        write_tensor_to_mmap(file_path_2, x)
        signal_client_processed(status_path)
        print("Processed tensor written back to shared memory.")
        print("count:", count)
        count += 1
        if count % 19 == 0:
            print("A batch is computed")
            print("TEE->GPU server side time: --- %s seconds ---" % (read_from_client_time))
            print("GPU->TEE server side time: --- %s seconds ---" % (send_to_client_time))
            print("GPU computation time: --- %s seconds ---" % (computation_time))
            read_from_client_time = 0
            send_to_client_time = 0
            computation_time = 0
        time.sleep(0.0001)
    elif optype == 6:
        x, p = read_dropout_from_mmap(file_path_1)
        print("Dropout data received from shared memory:")
        os.remove(file_path_1)

        if x is None:
            print("Received None data for Dropout layer. Skipping computation.")
            signal_client_processed(status_path)
            continue

        tempstart = time.time()
        x = x.to(device)
        x = F.dropout(x, p, training=True).cpu()

        print("Dropout layer time: --- %s seconds ---" % (time.time() - tempstart))
        computation_time += time.time() - tempstart

        write_tensor_to_mmap(file_path_2, x)
        signal_client_processed(status_path)
        print("Processed tensor written back to shared memory.")
        print("count:", count)
        count += 1
        time.sleep(0.0001)