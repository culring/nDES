# Loosely based on https://github.com/rianrajagede/iris-python

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from des_torch import DESOptimizer

DEVICE = torch.device("cuda:0")  # CUDA GPU 0

# load
datatrain = pd.read_csv('../data/iris_train.csv')

# change string value to numeric
datatrain.loc[datatrain['species'] == 'Iris-setosa', 'species'] = 0
datatrain.loc[datatrain['species'] == 'Iris-versicolor', 'species'] = 1
datatrain.loc[datatrain['species'] == 'Iris-virginica', 'species'] = 2
datatrain = datatrain.apply(pd.to_numeric)

# change dataframe to array
datatrain_array = datatrain.values

# split x and y (feature and target)
xtrain = datatrain_array[:, :4]
ytrain = datatrain_array[:, 4]

torch.manual_seed(1235)

# hyperparameters
hl = 10
lr = 0.01
num_epoch = 5000
DES_TRAINING = True


# build model
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, hl)
        self.fc2 = nn.Linear(hl, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = Net().to(DEVICE)

# choose optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr)

if DES_TRAINING:
    X = torch.Tensor(xtrain).float().to(DEVICE)
    Y = torch.Tensor(ytrain).long().to(DEVICE)
    des_optim = DESOptimizer(net, criterion, X, Y, restarts=10,
                             lower=-3., upper=3., budget=num_epoch, tol=1e-6,
                             nn_train=True, device=DEVICE)
    des_optim.run()

else:
    for name, param in net.named_parameters():
        param.requires_grad = False
        if name == 'fc2.weight':
            param.requires_grad = True
    # train
    for epoch in range(num_epoch):
        X = torch.Tensor(xtrain, device=DEVICE).float()
        Y = torch.Tensor(ytrain, device=DEVICE).long()

        # feedforward - backprop
        optimizer.zero_grad()
        out = net(X)
        loss = criterion(out, Y)
        loss.backward()
        optimizer.step()

        if (epoch) % 50 == 0:
            print(f'Epoch [{epoch + 1}/{num_epoch}] Loss: {loss.data:.4f}')
    for name, param in net.named_parameters():
        if name == 'fc2.weight':
            np.save("sgd_weights.npy", param.data.numpy())

# load
datatest = pd.read_csv('../data/iris_test.csv')

# change string value to numeric
datatest.loc[datatest['species'] == 'Iris-setosa', 'species'] = 0
datatest.loc[datatest['species'] == 'Iris-versicolor', 'species'] = 1
datatest.loc[datatest['species'] == 'Iris-virginica', 'species'] = 2
datatest = datatest.apply(pd.to_numeric)

# change dataframe to array
datatest_array = datatest.values

# split x and y (feature and target)
xtest = datatest_array[:, :4]
ytest = datatest_array[:, 4]

# get prediction
X = torch.Tensor(xtest).float().to(DEVICE)
Y = torch.Tensor(ytest).long().to(DEVICE)
out = net(X)
_, predicted = torch.max(out.data, 1)

# get accuration
print(f'Accuracy of the network {100 * torch.sum(Y == predicted) / 30}%')
