# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import random
import time

import torch
import time
import socket
import pickle

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#########################################################
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('127.0.0.1', 11111)
client_socket.connect(server_address)

start_time = time.time()

abort_sigal = 0
# getting inputs from user
loaded_arrays = np.load('user_input.npy', allow_pickle=True)
c, alpha, xw, w = loaded_arrays
print("loaded from user: --- %s seconds ---" % (time.time() - start_time))

##### Implement permutation here
data_tuple = (c, w)
data_to_server = pickle.dumps(data_tuple)
print("saved to pickle: --- %s seconds ---" % (time.time() - start_time))
client_socket.sendall(len(data_to_server).to_bytes(8, byteorder='big'))
client_socket.sendall(data_to_server)
client_socket.close()
print("sent to server: --- %s seconds ---" % (time.time() - start_time))

##### getting computed result from server
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_address = ('127.0.0.1', 44444)
client_socket.bind(client_address)
client_socket.listen(3)
server_socket, server_address = client_socket.accept()

CT_size = int.from_bytes(server_socket.recv(8), byteorder='big')
CT = b""
while True:
    packet = server_socket.recv(CT_size)
    if not packet: break
    CT += packet
CT = pickle.loads(CT)
server_socket.close()
client_socket.close()
print("received from server: --- %s seconds ---" % (time.time() - start_time))
##### Implement de-permutation here

y = CT - alpha

if(np.array_equal(y, xw)):
    print("Verification pass: --- %s seconds ---" % (time.time() - start_time))
else:
    print("Verification fail: --- %s seconds ---" % (time.time() - start_time))
