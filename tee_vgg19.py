import torchvision.transforms as transforms
import torchvision
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch import nn
from torch.nn import DataParallel
import numpy as np
import time
from tqdm import tqdm
# from torchsummary import summary

torch.cuda.empty_cache()
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

batch_size = 8
train_dataset = torchvision.datasets.CIFAR10(root = './dataset',
                           train = True,
                           transform = train_tf,
                           download = True)
test_dataset = torchvision.datasets.CIFAR10(root = './dataset',
                           train = False,
                           transform = valid_tf,
                           download = True)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, 
                                           batch_size = batch_size, 
                                           shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = batch_size,
                                          shuffle = False)
 


class VGG19(nn.Module):
    def __init__(self,in_dim,num_classes):
        super().__init__()
        self.features = nn.Sequential(
             nn.Conv2d(in_dim,64,kernel_size=3,padding=1),
             nn.BatchNorm2d(64),
             nn.ReLU(inplace=True),
             nn.Conv2d(64,64,kernel_size=3,padding=1),
             nn.BatchNorm2d(64),
             nn.ReLU(inplace=True),
             nn.MaxPool2d(kernel_size=2,stride=2),
 
             nn.Conv2d(64,128,kernel_size=3,padding=1),
             nn.BatchNorm2d(128),
             nn.ReLU(inplace=True),
             nn.Conv2d(128, 128, kernel_size=3, padding=1),
             nn.BatchNorm2d(128),
             nn.ReLU(inplace=True),
             nn.MaxPool2d(kernel_size=2,stride=2),
 
             nn.Conv2d(128, 256, kernel_size=3, padding=1),
             nn.BatchNorm2d(256),
             nn.ReLU(inplace=True),
             nn.Conv2d(256, 256, kernel_size=3, padding=1),
             nn.BatchNorm2d(256),
             nn.ReLU(inplace=True),
             nn.Conv2d(256, 256, kernel_size=3, padding=1),
             nn.BatchNorm2d(256),
             nn.ReLU(inplace=True),
             nn.Conv2d(256, 256, kernel_size=3, padding=1),
             nn.BatchNorm2d(256),
             nn.ReLU(inplace=True),
             nn.MaxPool2d(kernel_size=2,stride=2),
            
             nn.Conv2d(256, 512, kernel_size=3, padding=1),
             nn.BatchNorm2d(512),
             nn.ReLU(inplace=True),
             nn.Conv2d(512, 512, kernel_size=3, padding=1),
             nn.BatchNorm2d(512),
             nn.ReLU(inplace=True),
             nn.Conv2d(512, 512, kernel_size=3, padding=1),
             nn.BatchNorm2d(512),
             nn.ReLU(inplace=True),
             nn.Conv2d(512, 512, kernel_size=3, padding=1),
             nn.BatchNorm2d(512),
             nn.ReLU(inplace=True),
             nn.MaxPool2d(kernel_size=2,stride=2),
            
             nn.Conv2d(512, 512, kernel_size=3, padding=1),
             nn.BatchNorm2d(512),
             nn.ReLU(inplace=True),
             nn.Conv2d(512, 512, kernel_size=3, padding=1),
             nn.BatchNorm2d(512),
             nn.ReLU(inplace=True),
             nn.Conv2d(512, 512, kernel_size=3, padding=1),
             nn.BatchNorm2d(512),
             nn.ReLU(inplace=True),
             nn.Conv2d(512, 512, kernel_size=3, padding=1),
             nn.BatchNorm2d(512),
             nn.ReLU(inplace=True),
             nn.MaxPool2d(kernel_size=2,stride=2)
         )
 
        self.classifier = nn.Sequential(
             nn.Linear(512*7*7,4096),
             nn.ReLU(),
             nn.Dropout(0.5),
 
             nn.Linear(4096,4096),
             nn.ReLU(),
             nn.Dropout(0.5),
            
             nn.Linear(4096,num_classes)
         )
        # self.quant = torch.quantization.QuantStub()
        # self.dequant = torch.quantization.DeQuantStub()
    
    # def fuse_model(self):
    #     for module_name, module in self.features.named_children():
    #         if 'Conv' in module_name:
    #             torch.quantization.fuse_modules(module, ['0', '1', '2'], inplace=True)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)

        # x = self.quant(x)
        # x = self.features(x)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        # x = self.dequant(x)

        return x

net = VGG19(3,10)
# net.qconfig = torch.quantization.get_default_qconfig('fbgemm')
# net.fuse_model()
# net = torch.quantization.prepare(net, inplace=False)

# if torch.cuda.device_count() > 1:
#     net = DataParallel(net)
net.to(device)
# summary(net, input_size=(3, 224, 224))

num_images = 64
num_batches = num_images // batch_size

learning_rate = 1e-2
num_epochs = 1
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr = learning_rate,momentum=0.9)
net.train()

start_time = time.time()
for epoch in range(num_epochs):
    running_loss = 0.0
    correct_top1 = 0
    correct_top3 = 0
    total = 0
    # progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
    
    batch_count = 0
    for i, (images, labels) in enumerate(train_loader):
    # for i, (images, labels) in progress_bar:
        # if batch_count >= num_batches:
        #     break
        images = images.to(device)
        labels = labels.to(device)
        # print(images)
        outputs = net(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # _, predicted = torch.max(outputs, 1)
        # _, top3 = torch.topk(outputs, 3, dim=1)
        # total += labels.size(0)
        # correct_top1 += (predicted == labels).sum().item()
        # correct_top3 += sum([labels[i] in top3[i] for i in range(labels.size(0))])

        # top1_accuracy = 100 * correct_top1 / total
        # top3_accuracy = 100 * correct_top3 / total
        # progress_bar.set_postfix(loss=running_loss/(i+1), top1_acc=top1_accuracy, top3_acc=top3_accuracy)

        # batch_count += 1
        break
#     epoch_loss = running_loss / len(train_loader)
#     epoch_top1_accuracy = 100 * correct_top1 / total
#     epoch_top3_accuracy = 100 * correct_top3 / total
#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Top-1 Accuracy: {epoch_top1_accuracy:.2f}%, Top-3 Accuracy: {epoch_top3_accuracy:.2f}%')
    print(time.time()-start_time)
            
# net.eval()

# Disable gradient calculation
# with torch.no_grad():
#     correct_top1 = 0
#     correct_top3 = 0
#     total = 0

#     for images, labels in test_loader:
#         images = images.to(device)
#         labels = labels.to(device)

#         # Forward pass
#         outputs = net(images)

#         # Top-1 accuracy
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct_top1 += (predicted == labels).sum().item()

#         # Top-3 accuracy
#         _, top3 = torch.topk(outputs, 3, dim=1)
#         correct_top3 += sum([labels[i] in top3[i] for i in range(labels.size(0))])

#     top1_accuracy = 100 * correct_top1 / total
#     top3_accuracy = 100 * correct_top3 / total

#     print(f'Top-1 Accuracy: {top1_accuracy:.2f}%')
#     print(f'Top-3 Accuracy: {top3_accuracy:.2f}%')
# print("--- %s seconds ---" % (time.time() - start_time))
