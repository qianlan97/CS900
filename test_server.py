import socket
import pickle
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"running on {device}")

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('localhost', 11111)
server_socket.bind(server_address)
server_socket.listen(1)

print("Server is listening...")

# Accept incoming connection
client_socket, client_address = server_socket.accept()
start_time = time.time()
# Receive the sizes of the arrays first
data_size = int.from_bytes(client_socket.recv(8), byteorder='big')
# w_size = int.from_bytes(client_socket.recv(8), byteorder='big')
data = b""
while True:
    packet = client_socket.recv(data_size)
    if not packet: break
    data += packet
# c = client_socket.recv(c_size)
c, w = pickle.loads(data)
client_socket.close()
server_socket.close()
print("received: --- %s seconds ---" % (time.time() - start_time))
##### start computing
c_gpu = torch.from_numpy(c)
w_gpu = torch.from_numpy(w)
print("converted to pytorch: --- %s seconds ---" % (time.time() - start_time))
CT_gpu=torch.mm(c_gpu,w_gpu)
print("computed: --- %s seconds ---" % (time.time() - start_time))
CT = CT_gpu.numpy()
print("converted to numpy: --- %s seconds ---" % (time.time() - start_time))
##### send back
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_address = ('localhost', 44444)
server_socket.connect(client_address)
CT_to_tee = pickle.dumps(CT)
server_socket.sendall(len(CT_to_tee).to_bytes(8, byteorder='big'))
print("sent: --- %s seconds ---" % (time.time() - start_time))
server_socket.sendall(CT_to_tee)
server_socket.close()

