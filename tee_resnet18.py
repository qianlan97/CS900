import numpy as np
import random
from tqdm import tqdm
import time

import torch
import torchvision
import time
from torch import nn
import torchvision.transforms as transforms
from torch.nn import DataParallel
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

# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"running on {device}")

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        DROPOUT = 0.1

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(DROPOUT)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(DROPOUT)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes),
                nn.Dropout(DROPOUT)
            )

    def forward(self, x):
        out = self.relu1(self.dropout1(self.bn1(self.conv1(x))))
        out = self.relu2(self.dropout2(self.bn2(self.conv2(out))))
        # out = self.dropout(self.bn2(self.conv2(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        x = self.dequant(x)
        return F.log_softmax(out, dim=-1)

    def fuse_model(self):
        for m in self.modules():
            if type(m) == BasicBlock:
                torch.quantization.fuse_modules(m, ['conv1', 'bn1', 'relu1'], inplace=True)
                torch.quantization.fuse_modules(m, ['conv2', 'bn2', 'relu2'], inplace=True)
                if m.shortcut:
                    torch.quantization.fuse_modules(m.shortcut, ['0', '1'], inplace=True)


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

model = ResNet18()
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")

if torch.cuda.device_count() > 1:
    model = DataParallel(model)
model.to(device)
model.fuse_model()
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model = torch.quantization.prepare(model, inplace=False)
# summary(model, input_size=(3,32,32))

optimizer1=torch.optim.SGD(model.parameters(),lr=0.01)          ## SGD more likely to get optimal. Adam converge faster
loss=nn.CrossEntropyLoss().to(device)

start_time = time.time()
for epoch in range(5):
    losssum=0.0
    total=0
    correct_top1 = 0
    correct_top3 = 0
    progress_bar = tqdm(enumerate(train_dataset), total=len(train_dataset), desc=f'Epoch {epoch+1}/{5}', leave=False)
    accuracy=0.0
    for i,(images,labels) in progress_bar:
        imgs=images.to(device)
        labels=labels.to(device)

        optimizer1.zero_grad()                                            ## clear gradient
        output=model.forward(imgs)                                 ## predict
        lossnum=loss(output,labels)                                   ## nn.CrossEntropyLoss()
        lossnum.backward()                                                ## backpropagation
        optimizer1.step()                                                     ## update parameters using gradients(using SGD defined above)
        losssum=losssum+lossnum
        accuracy+=(output.argmax(1)==labels).sum()      ## count number of correct predictions by argmax()
        _, predicted = torch.max(output, 1)
        _, top3 = torch.topk(output, 3, dim=1)
        total += labels.size(0)
        correct_top1 += (predicted == labels).sum().item()
        correct_top3 += sum([labels[i] in top3[i] for i in range(labels.size(0))])

        top1_accuracy = 100 * correct_top1 / total
        top3_accuracy = 100 * correct_top3 / total
        progress_bar.set_postfix(loss=losssum/(i+1), top1_acc=top1_accuracy, top3_acc=top3_accuracy)
    # print("Epoch: {}'s accuracy is {}".format(epoch, accuracy/len(train_set)))

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
with torch.no_grad():
    correct_top1 = 0
    correct_top3 = 0
    total = 0
    for images, labels in test_dataset:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        # Top-1 accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct_top1 += (predicted == labels).sum().item()

        # Top-3 accuracy
        _, top3 = torch.topk(outputs, 3, dim=1)
        correct_top3 += sum([labels[i] in top3[i] for i in range(labels.size(0))])

    top1_accuracy = 100 * correct_top1 / total
    top3_accuracy = 100 * correct_top3 / total
    print(f'Test - Top-1 Accuracy: {top1_accuracy:.2f}%, Top-3 Accuracy: {top3_accuracy:.2f}%')

print("--- %s seconds ---" % (time.time() - start_time))