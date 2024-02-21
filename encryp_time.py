import torchvision.transforms as transforms
import torchvision
import torch
import torchvision.transforms as transforms
import time
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from base64 import b64encode, b64decode
import hashlib
import pickle
import numpy as np

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

batch_size = 1

train_dataset = torchvision.datasets.CIFAR10(root = './dataset',
                           train = True,
                           transform = train_tf,
                           download = True)
test_dataset = torchvision.datasets.CIFAR10(root = './dataset',
                           train = False,
                           transform = valid_tf,
                           download = True)

def pad(data):
    return data + (AES.block_size - len(data) % AES.block_size) * b'\x00'

def unpad(data):
    return data.rstrip(b'\x00')

def encrypt(byte_data, key):
    iv = get_random_bytes(AES.block_size)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    encrypted_data = iv + cipher.encrypt(pad(byte_data))
    return encrypted_data

def decrypt(encrypted_data, key):
    iv = encrypted_data[:AES.block_size]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted_data = unpad(cipher.decrypt(encrypted_data[AES.block_size:]))
    return decrypted_data

# Test the functions
key = hashlib.sha256('my_secure_password'.encode()).digest()  # 32 bytes key
data = pickle.dumps((train_dataset, test_dataset))

# print(len(data))
# message = "This is a plain text message."
# print('plain text is:', message)
start_time = time.time()
# Encrypt the message
encrypted_message = encrypt(data, key)
# print(f"Encrypted: {encrypted_message}")
print("data encrypted: --- %s seconds ---" % (time.time() - start_time))
# Decrypt the message
decrypted_message = decrypt(encrypted_message, key)
train_dataset, test_dataset = pickle.loads(data)
# print(f"Decrypted: {decrypted_message}")
print("data decrypted: --- %s seconds ---" % (time.time() - start_time))

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, 
                                           batch_size = batch_size, 
                                           shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = batch_size,
                                          shuffle = False)