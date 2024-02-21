import numpy as np
import random
from tqdm import tqdm
import time

import torch
import torchvision
import time
from torch import nn
import torchvision.transforms as transforms
from torch.utils import data
from torchsummary import summary
import torch.nn.functional as F
import socket
import pickle

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"running on {device}")

class CustomConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(CustomConv2d, self).__init__()
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

    def forward(self, x):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = ('127.0.0.1', 11111)
        client_socket.connect(server_address)
        op_type = 1 ## conv operation
        data_tuple = (x, self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.bias)
        data_to_server = pickle.dumps(data_tuple)
        client_socket.sendall(op_type.to_bytes(8, byteorder='big'))
        client_socket.sendall(len(data_to_server).to_bytes(8, byteorder='big'))
        client_socket.sendall(data_to_server)
        client_socket.close()
        print("sent to server: --- %s seconds ---" % (time.time() - start_time))

        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_address = ('127.0.0.1', 55555)
        client_socket.bind(client_address)
        client_socket.listen(3)
        server_socket, server_address = client_socket.accept()
        x_size = int.from_bytes(server_socket.recv(8), byteorder='big')
        x = b""
        while True:
            packet = server_socket.recv(x_size)
            if not packet: break
            x += packet
        x = pickle.loads(x)
        server_socket.close()
        client_socket.close()
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
        server_address = ('127.0.0.1', 11111)
        client_socket.connect(server_address)
        op_type = 2
        data_tuple = (x, self.in_features, self.out_features, self.bias)
        data_to_server = pickle.dumps(data_tuple)
        client_socket.sendall(op_type.to_bytes(8, byteorder='big'))
        client_socket.sendall(len(data_to_server).to_bytes(8, byteorder='big'))
        client_socket.sendall(data_to_server)
        client_socket.close()
        print("sent to server: --- %s seconds ---" % (time.time() - start_time))

        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_address = ('127.0.0.1', 55555)
        client_socket.bind(client_address)
        client_socket.listen(3)
        server_socket, server_address = client_socket.accept()
        x_size = int.from_bytes(server_socket.recv(8), byteorder='big')
        x = b""
        while True:
            packet = server_socket.recv(x_size)
            if not packet: break
            x += packet
        x = pickle.loads(x)
        server_socket.close()
        client_socket.close()
        print("received from server: --- %s seconds ---" % (time.time() - start_time))
        return x.to(device)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        DROPOUT = 0.1

        self.conv1 = CustomConv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(DROPOUT)
        self.conv2 = CustomConv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(DROPOUT)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                CustomConv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes),
                nn.Dropout(DROPOUT)
            )

    def forward(self, x):
        out = F.relu(self.dropout(self.bn1(self.conv1(x))))
        out = self.dropout(self.bn2(self.conv2(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = CustomConv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = CustomLinear(512*block.expansion, num_classes, bias=True)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return F.log_softmax(out, dim=-1)


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])



trans = transforms.Compose([
    # transforms.Resize(224),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.2, 0.2, 0.2))
])

train_set=torchvision.datasets.CIFAR10(root="./dataset",train=True,download=True,transform=trans)
test_set=torchvision.datasets.CIFAR10(root="./dataset",train=False,download=True,transform=trans)
train_dataset=data.DataLoader(train_set,batch_size=64,shuffle=True)
test_dataset=data.DataLoader(test_set,batch_size=64)


model = ResNet18().to(device)
# summary(model, input_size=(3,32,32))

optimizer1=torch.optim.SGD(model.parameters(),lr=0.01)          ## SGD more likely to get optimal. Adam converge faster
loss=nn.CrossEntropyLoss().to(device)

start_time = time.time()
for epoch in range(1):
    losssum=0.0
    total=0
    accuracy=0.0
    for i,(images,labels) in tqdm(enumerate(train_dataset)):
        imgs=images.to(device)
        labels=labels.to(device)

        optimizer1.zero_grad()                                            ## clear gradient
        output=model.forward(imgs)                                 ## predict
        lossnum=loss(output,labels)                                   ## nn.CrossEntropyLoss()
        lossnum.backward()                                                ## backpropagation
        optimizer1.step()                                                     ## update parameters using gradients(using SGD defined above)
        losssum=losssum+lossnum
        accuracy+=(output.argmax(1)==labels).sum()      ## count number of correct predictions by argmax()
        break
    print("Epoch: {}'s accuracy is {}".format(epoch, accuracy/len(train_set)))

# with torch.no_grad():
#     accuracy=0.0
#     for data in test_dataset:
#         imgs,labels=data
#         imgs=imgs.to(device)
#         labels=labels.to(device)

#         output=model.forward(imgs)
#         lossnum=loss(output,labels)
#         accuracy+=(output.argmax(1)==labels).sum()
#     print("Accuracy on the test dataset is {}".format(accuracy/ len(test_set)))

print("--- %s seconds ---" % (time.time() - start_time))