# Loosely based on https://github.com/rianrajagede/iris-python

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

from des_torch import DESOptimizer

# load
torch.manual_seed(1235)

# hyperparameters
hl = 10
lr = 0.01
#  num_epoch = 25000
num_epoch = 300000
DES_TRAINING = True
#DEVICE = torch.device("cpu")
DEVICE = torch.device("cuda")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 32, 5)
        self.fc1 = nn.Linear(4*4*32, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        #  x = F.leaky_relu(self.conv1(x), 0.1)
        x = torch.tanh(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        #  x = F.leaky_relu(self.conv2(x), 0.1)
        x = torch.tanh(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*32)
        #  x = F.leaky_relu(self.fc1(x), 0.1)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# class Net(nn.Module):
#
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(784, 10)
#         #  self.fc2 = nn.Linear(10, 10)
#
#     def forward(self, x):
#         x = x.view(x.size()[0], -1)
#         #  x = F.relu(self.fc1(x))
#         #  x = self.fc2(x)
#         x = self.fc1(x)
#         return x


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def bootstrap(model, device, num_batches=10):
    train_dataset = datasets.MNIST(
        '../data', train=True, download=True,
        transform=transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize((0.1307,), (0.3081,))
       ])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx >= num_batches:
            print('Loss after bootstrap: {:.6f}'.format(loss.item()))
            return model


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == "__main__":
    mean = (125.30691805, 122.95039414, 113.86538318)
    std = (62.99321928, 62.08870764, 66.70489964)
    train_dataset = datasets.MNIST(
        '../data', train=True, download=True,
        transform=transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize((0.1307,), (0.3081,))
       ])
    )
    test_dataset = datasets.MNIST(
        '../data', train=False, transform=transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize((0.1307,), (0.3081,))
        ])
    )

# build model
    model = Net().to(DEVICE)
    #  for layer in model.parameters():
        #  torch.nn.init.zeros_(layer)
    # model = bootstrap(model, DEVICE, num_batches=200)
    print(f"Num params: {sum([param.nelement() for param in model.parameters()])}")
    #  print(model.conv1.weight)

# choose optimizer and loss function
    criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer = torch.optim.Adam(model.parameters())


    if DES_TRAINING:
        model.eval()
        for x, y in torch.utils.data.DataLoader(train_dataset,
                                             batch_size=len(train_dataset),
                                             shuffle=True):
            #  print(f"Before dataset: {torch.cuda.memory_allocated(DEVICE) / (1024**3)}")
            x_train, y_train = x.to(DEVICE), y.to(DEVICE)
            #  print(f"After dataset: {torch.cuda.memory_allocated(DEVICE) / (1024**3)}")

        x_train = x_train[:1000, :]
        y_train = y_train[:1000]
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1000, shuffle=True)
        des_optim = DESOptimizer(model, criterion, x_train, y_train, restarts=3,
                                 lower=-2., upper=2., budget=num_epoch, tol=1e-6,
                                 nn_train=True, lambda_=4000, history=16,
                                 log_best_val=False, device=DEVICE)
        model = des_optim.run(lambda x: test(x, DEVICE, test_loader))
        test(model, DEVICE, test_loader)
        # torch.save(model, 'model.pt')
        torch.save({'state_dict': model.state_dict()}, 'model.pth.tar')
    else:
        model.train()
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1000, shuffle=True)

        for epoch in range(1, num_epoch + 1):
            train(model, DEVICE, train_loader, optimizer, epoch)
            test(model, DEVICE, test_loader)
