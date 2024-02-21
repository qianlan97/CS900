# client.py
import mmap
import torch
import os
import time

# Configuration
file_path = "shared_mmap.dat"
status_path = "status.dat"
tensor_shape = (1019, 10189)  # This will give a tensor close to 411042209 bytes for float32

def write_tensor_to_mmap(file_path, tensor):
    with open(file_path, "wb") as f:
        f.write(tensor.numpy().tobytes())

def read_tensor_from_mmap(file_path, shape):
    with open(file_path, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        tensor = torch.frombuffer(mm.read(), dtype=torch.float32).view(shape)
        mm.close()
    return tensor

def signal_server_ready(status_path):
    with open(status_path, 'w') as f:
        f.write('ready')

def wait_for_server_response(status_path):
    while True:
        if os.path.exists(status_path):
            with open(status_path, 'r') as f:
                if f.read() == 'processed':
                    break
        time.sleep(0.0001)


# Create a random tensor
tensor = torch.rand(tensor_shape, dtype=torch.float32)
print('finished generating matrix')
# Write the tensor to the memory-mapped file
start_time = time.time()
write_tensor_to_mmap(file_path, tensor)
signal_server_ready(status_path)

print("Tensor written to shared memory. Waiting for the server to process it.")
print(time.time() - start_time)
# Wait for the server to write back the processed tensor
wait_for_server_response(status_path)

# Read the new tensor from the server
new_tensor = read_tensor_from_mmap(file_path, tensor_shape)
print("New tensor received from the server:")
print(new_tensor)
print(new_tensor.size())
print('total time taken:', time.time() - start_time)
