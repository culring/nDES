import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
from math import ceil

from ndes import NDESOptimizer
from utils import bootstrap, train_via_ndes, train_via_gradient, seed_everything

#  EPOCHS = 25000
EPOCHS = 4 * 1200000
NDES_TRAINING = False
#  DEVICE = torch.device("cpu")
DEVICE = torch.device("cuda:0")
BOOTSTRAP_BATCHES = None
MODEL_NAME = "fashion_ndes.pth.tar"
LOAD_WEIGHTS = False
SEED_OFFSET = 0
BATCH_SIZE = 64
VALIDATION_SIZE = 10000
STRATIFY = True


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5)
        self.conv2 = nn.Conv2d(20, 32, 5)
        self.fc1 = nn.Linear(5 * 5 * 32, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        #  x = F.leaky_relu(self.conv1(x), 0.1)
        x = F.softsign(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        #  x = F.leaky_relu(self.conv2(x), 0.1)
        x = F.softsign(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5 * 5 * 32)
        #  x = F.leaky_relu(self.fc1(x), 0.1)
        x = F.softsign(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def cycle(loader):
    while True:
        for element in enumerate(loader):
            yield element


if __name__ == "__main__":
    seed_everything(SEED_OFFSET)

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    train_dataset = datasets.CIFAR10(
        "../data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        ),
    )
    test_dataset = datasets.CIFAR10(
        "../data",
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        ),
    )

    model = Net().to(DEVICE)
    if LOAD_WEIGHTS:
        model.load_state_dict(torch.load(MODEL_NAME)["state_dict"])

    for x, y in torch.utils.data.DataLoader(
        train_dataset, batch_size=len(train_dataset), shuffle=True
    ):
        #  print(f"Before dataset: {torch.cuda.memory_allocated(DEVICE) / (1024**3)}")
        x_train, y_train = x.to(DEVICE), y.to(DEVICE)
        #  print(f"After dataset: {torch.cuda.memory_allocated(DEVICE) / (1024**3)}")

    train_idx, val_idx = train_test_split(
        np.arange(0, len(train_dataset)),
        test_size=VALIDATION_SIZE,
        stratify=y_train.cpu().numpy(),
    )

    x_val = x_train[val_idx, :]
    y_val = y_train[val_idx]
    x_train = x_train[train_idx, :]
    y_train = y_train[train_idx]

    if STRATIFY:
        indices_x = []
        indices_y = []
        splitter = StratifiedKFold(
            n_splits=ceil(len(train_dataset) / BATCH_SIZE),
            # random_state=(42 + SEED_OFFSET),
        )
        reordering = [
            i
            for _, batch in splitter.split(
                np.arange(0, x_train.shape[0]), y_train.cpu().numpy()
            )
            for i in batch
        ]
        x_train = x_train[reordering, :]
        y_train = y_train[reordering]

    print(y_train.unique(return_counts=True))

    train_dataset = TensorDataset(x_train, y_train)

    if BOOTSTRAP_BATCHES is not None:
        model = bootstrap(model, train_dataset, DEVICE, num_batches=BOOTSTRAP_BATCHES)
    print(f"Num params: {sum([param.nelement() for param in model.parameters()])}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    train_loader = cycle(
        torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    )

    if NDES_TRAINING:
        ndes_optim = NDESOptimizer(
            model,
            criterion,
            #  x_train,
            #  y_train,
            train_loader,
            None,
            ewma_alpha=0.3,
            num_batches=ceil(len(train_dataset) / BATCH_SIZE),
            x_val=x_val,
            y_val=y_val,
            restarts=None,
            Ft=1,
            ccum=0.97,
            # cp=0.1,
            lower=-2.0,
            upper=2.0,
            budget=EPOCHS,
            tol=1e-6,
            nn_train=True,
            lambda_=4000,
            history=16,
            log_best_val=False,
            device=DEVICE,
        )
        #  train_via_des(model, des_optim, DEVICE, test_dataset, 'des_' + MODEL_NAME)
        train_via_ndes(model, ndes_optim, DEVICE, test_dataset, MODEL_NAME)
    else:
        #  train_via_gradient(
        #  model, criterion, optimizer, train_dataset, test_dataset, EPOCHS, DEVICE
        #  )
        train_dataset = TensorDataset(x_train, y_train)
        train_via_gradient(
            model, criterion, optimizer, train_dataset, test_dataset, EPOCHS, DEVICE
        )
