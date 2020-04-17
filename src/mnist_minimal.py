# Loosely based on https://github.com/rianrajagede/iris-python

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

from des_torch import DESOptimizer
import random

# load
def seed_everything():
    torch.manual_seed(1235)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(42)
    np.random.seed(42)

# hyperparameters
hl = 10
lr = 0.01
num_epoch = 100000
DES_TRAINING = True
#DEVICE = torch.device("cpu")
DEVICE = torch.device("cuda")


class Net(nn.Module):
    def __init__(self, act):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 5)
        self.fc2 = nn.Linear(5, 10)
        self.activation = act

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.activation(x)
        #  x = F.leaky_relu(self.fc1(x), 0.1)
        #  x = torch.tanh(self.fc1(x) / 5)
        #  x = torch.exp(-self.fc1(x)**2 / 2)
        #  x = (self.fc1(x) >= 0.0).float()
        #  x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
        #  x = x.view(-1, 28*28)
        #  x = self.fc1(x)
        #  return F.log_softmax(x, dim=1)

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
    return test_loss, correct


def mnist_run(activation):
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
    seed_everything()
# build model
    model = Net(activation).to(DEVICE)
    for layer in model.parameters():
        torch.nn.init.zeros_(layer)
    #  torch.nn.init.zeros_(model.fc1.weight)
    #  torch.nn.init.zeros_(model.fc1.bias)
    #  torch.nn.init.ones_(model.fc2.weight)
    #  torch.nn.init.ones_(model.fc2.bias)
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

        des_optim = DESOptimizer(model, criterion, x_train, y_train, restarts=2,
                                 lower=-2., upper=2., budget=num_epoch, tol=1e-6,
                                 nn_train=True, lambda_=8000, history=16,
                                 log_best_val=False, device=DEVICE)
        model = des_optim.run()
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1000, shuffle=True)
        return test(des_optim.model, DEVICE, test_loader)
        # torch.save(model, 'model.pt')
        #  torch.save({'state_dict': model.state_dict()}, 'model.pth.tar')
    else:
        model.train()
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1000, shuffle=True)

        for epoch in range(1, num_epoch + 1):
            train(model, DEVICE, train_loader, optimizer, epoch)
            test(model, DEVICE, test_loader)


if __name__ == "__main__":
    activations = {
        'relu': F.relu,
        'tanh': torch.tanh,
        '5 * tanh(x/5)': lambda x: 5 * torch.tanh(x / 5),
        'tanh(x/5)': lambda x: torch.tanh(x / 5),
        '5 * tanh': lambda x: 5 * torch.tanh(x),
        'sigmoid': torch.sigmoid,
        'softplus': F.softplus,
        'softsign': F.softsign,
        'gauss': lambda x: torch.exp(-x**2 / 2),
        'binary step': lambda x: (x >= 0.).float(),
    }
    results = {}
    for name, act in activations.items():
        res = mnist_run(act)
        results[name] = res
    for name, (loss, correct) in results.items():
        print(f"{name}: Loss: {loss:.4f} Accuracy: {correct / 1e2:.2f}%")
