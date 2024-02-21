# server.py
import mmap
import torch
import os
import time

# Configuration
file_path = "shared_mmap.dat"
status_path = "status.dat"
tensor_shape = (1019, 10189)  # This will give a tensor close to 411042209 bytes for float32

def read_tensor_from_mmap(file_path):
    with open(file_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        tensor = torch.frombuffer(mm.read(), dtype=torch.float32).view(tensor_shape)
        mm.close()
    return tensor

def write_tensor_to_mmap(file_path, tensor):
    with open(file_path, "wb") as f:
        f.write(tensor.numpy().tobytes())

def wait_for_client_data(status_path):
    while True:
        if os.path.exists(status_path):
            with open(status_path, 'r') as f:
                if f.read() == 'ready':
                    break
        time.sleep(0.0001)

def signal_client_processed(status_path):
    with open(status_path, 'w') as f:
        f.write('processed')

while True:
    # Wait for the data to be ready
    print("Waiting for the client to write the data...")
    wait_for_client_data(status_path)

    # Read the tensor from the memory-mapped file
    tensor = read_tensor_from_mmap(file_path)
    print("Tensor received from shared memory:")

    # Process the tensor (for demonstration, we'll just multiply it by 2)
    # processed_tensor = tensor * 2

    # Write the processed tensor back to shared memory
    write_tensor_to_mmap(file_path, tensor)
    signal_client_processed(status_path)
    print("Processed tensor written back to shared memory.")

    # Wait a bit before next listen
    time.sleep(0.0001)
