import torch
from torch import nn, optim
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import time
import socket
import pickle


device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"running on {device}")

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
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = ('127.0.0.1', 22222)
        client_socket.connect(server_address)
        op_type = 1 ## conv operation
        data_tuple = (x, self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.bias)
        data_to_server = pickle.dumps(data_tuple)
        client_socket.sendall(op_type.to_bytes(8, byteorder='big'))
        client_socket.sendall(len(data_to_server).to_bytes(8, byteorder='big'))
        client_socket.sendall(data_to_server)
        client_socket.close()
        print("sent to server: --- %s seconds ---" % (time.time() - start_time))

        client_socket_new = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_address_new = ('127.0.0.1', 55555)
        client_socket_new.bind(client_address_new)
        client_socket_new.listen(3)
        server_socket_new, server_address_new = client_socket_new.accept()
        x_size = int.from_bytes(server_socket_new.recv(8), byteorder='big')
        x = b""
        while True:
            packet = server_socket_new.recv(x_size)
            if not packet: break
            x += packet
        x = pickle.loads(x)
        server_socket_new.close()
        client_socket_new.close()
        print("received from server: --- %s seconds ---" % (time.time() - start_time))
        return x.to(device)


class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(CustomLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

    def forward(self, x):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = ('127.0.0.1', 22222)
        client_socket.connect(server_address)
        op_type = 2
        data_tuple = (x, self.in_features, self.out_features, self.bias)
        data_to_server = pickle.dumps(data_tuple)
        client_socket.sendall(op_type.to_bytes(8, byteorder='big'))
        client_socket.sendall(len(data_to_server).to_bytes(8, byteorder='big'))
        client_socket.sendall(data_to_server)
        client_socket.close()
        print("sent to server: --- %s seconds ---" % (time.time() - start_time))

        client_socket_new = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_address_new = ('127.0.0.1', 55555)
        client_socket_new.bind(client_address_new)
        client_socket_new.listen(3)
        server_socket_new, server_address_new = client_socket_new.accept()
        x_size = int.from_bytes(server_socket_new.recv(8), byteorder='big')
        x = b""
        while True:
            packet = server_socket_new.recv(x_size)
            if not packet: break
            x += packet
        x = pickle.loads(x)
        server_socket_new.close()
        client_socket_new.close()
        print("received from server: --- %s seconds ---" % (time.time() - start_time))
        return x.to(device)

class MyAlexNet(nn.Module):
    def __init__(self):
        super(MyAlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            CustomConv2d(in_channels=3, out_channels=96, kernel_size=11, stride=1, padding=1), nn.ReLU(),
            nn.BatchNorm2d(96), nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            CustomConv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2), nn.ReLU(),
            nn.BatchNorm2d(256), nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            CustomConv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1), nn.ReLU(),
            CustomConv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1), nn.ReLU(),
            CustomConv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Flatten(),
            CustomLinear(256 * 2 * 2, 1024), nn.ReLU(),
            nn.Dropout(p=0.5),
            CustomLinear(1024, 1024), nn.ReLU(),
            nn.Dropout(p=0.5),
            CustomLinear(1024, 10)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

if __name__ == '__main__':

    transform = torchvision.transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = torchvision.datasets.CIFAR10('./dataset', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10('./dataset', train=False, transform=transform, download=True)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    examples = enumerate(test_dataloader)
    batch_idx, (example_data, example_targets) = next(examples)
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = MyAlexNet()
    net.to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    total_train_step = 0
    train_losses = []
    train_acces = []
    eval_acces = []
    total_test_step = 0
    for epoch in range(1):
        net.train()
        train_acc = 0
        for imgs, targets in train_dataloader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            start_time = time.time()
            output = net(imgs)

            Loss = loss(output, targets)
            optimizer.zero_grad()
            Loss.backward()

            optimizer.step()

            _, pred = output.max(1)
            num_correct = (pred == targets).sum().item()
            acc = num_correct / 1
            train_acc += acc

            total_train_step = total_train_step + 1
            break
        print("--- %s seconds ---" % (time.time() - start_time))
        train_acces.append(train_acc / len(train_dataloader))
        train_losses.append(Loss.item())