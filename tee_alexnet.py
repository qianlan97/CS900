import torch
from torch import nn, optim
import torchvision
import torch.nn.functional as F
from torch.nn import DataParallel
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import time
from tqdm import tqdm

# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"running on {device}")

class MyAlexNet(nn.Module):
    def __init__(self):
        super(MyAlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11, 11), stride=1, padding=1), 
            nn.BatchNorm2d(96), nn.ReLU(), nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=1, padding=2),
            nn.BatchNorm2d(256), nn.ReLU(),nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 1024), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 10)
        )
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        # x = self.quant(x)
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.dequant(x)
        return x
    
    def fuse_model(self):
    # Fuse Conv2d + BatchNorm2d + ReLU
        torch.quantization.fuse_modules(self.layer1, ['0', '1', '2'], inplace=True)
        torch.quantization.fuse_modules(self.layer1, ['4', '5', '6'], inplace=True)

def train_imgClassification(net, num_epoch, batch_size, loss, optimizer, train_dataloader, test_dataloader, device):
    total_train_step = 0
    train_losses = []
    train_acces = []
    eval_acces = []
    total_test_step = 0
    net.train()
    for epoch in range(num_epoch):
        train_acc = 0
        correct_top1 = 0
        correct_top3 = 0
        total = 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epoch}', leave=False)

        start_time = time.time()
        for imgs, targets in progress_bar:
            imgs = imgs.to(device)
            targets = targets.to(device)
            output = net(imgs)

            Loss = loss(output, targets)
            optimizer.zero_grad()
            Loss.backward()

            optimizer.step()

            _, predicted = output.max(1)
            _, top3 = torch.topk(output, 3, dim=1)
            total += targets.size(0)
            correct_top1 += (predicted == targets).sum().item()
            correct_top3 += sum([targets[i] in top3[i] for i in range(targets.size(0))])

            top1_accuracy = 100 * correct_top1 / total
            top3_accuracy = 100 * correct_top3 / total
            progress_bar.set_postfix(loss=Loss.item(), top1_acc=top1_accuracy, top3_acc=top3_accuracy)

            total_train_step = total_train_step + 1

        # print("--- %s seconds ---" % (time.time() - start_time))
        train_acces.append(train_acc / len(train_dataloader))
        train_losses.append(Loss.item())
    net.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        correct_top1 = 0
        correct_top3 = 0
        total = 0
        for imgs, targets in tqdm(test_dataloader, desc='Evaluating', leave=False):
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = net(imgs)

            _, predicted = outputs.max(1)
            _, top3 = torch.topk(outputs, 3, dim=1)
            total += targets.size(0)
            correct_top1 += (predicted == targets).sum().item()
            correct_top3 += sum([targets[i] in top3[i] for i in range(targets.size(0))])

        top1_accuracy = 100 * correct_top1 / total
        top3_accuracy = 100 * correct_top3 / total
        print(f'Test - Top-1 Accuracy: {top1_accuracy:.2f}%, Top-3 Accuracy: {top3_accuracy:.2f}%')

if __name__ == '__main__':

    transform = torchvision.transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = torchvision.datasets.CIFAR10('./dataset', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10('./dataset', train=False, transform=transform, download=True)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    examples = enumerate(test_dataloader) 
    batch_idx, (example_data, example_targets) = next(examples)
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = MyAlexNet()
    # net.fuse_model()
    # net.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    # net = torch.quantization.prepare(net, inplace=False)
    # total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # print(f"Total trainable parameters: {total_params}")
    if torch.cuda.device_count() > 1:
        net = DataParallel(net)
    net.to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    train_imgClassification(net, 5, 64, loss, optimizer, train_dataloader, test_dataloader, device)