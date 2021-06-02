import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import datasets, transforms

import random

from ndes_optimizer_original import BasenDESOptimizer as BasenDESOptimizerOld
from ndes_optimizer_rewrited import BasenDESOptimizer as BasenDESOptimizerNew


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 32, 5)
        self.fc1 = nn.Linear(4*4*32, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        #  x = F.leaky_relu(self.conv1(x), 0.1)
        x = F.softsign(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        #  x = F.leaky_relu(self.conv2(x), 0.1)
        x = F.softsign(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*32)
        #  x = F.leaky_relu(self.fc1(x), 0.1)
        x = F.softsign(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def _seed_everything():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cycle(batches):
    while True:
        for idx, batch in enumerate(batches):
            yield idx, batch


class MyDataset(data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples[0])

    def __getitem__(self, idx):
        return self.samples[0][idx, :, :], self.samples[1][idx]


def extract_two_class_dataset_mnist(dataset):
    idxs = torch.logical_or(dataset.train_labels == 0, dataset.train_labels == 1)
    samples = (dataset.train_data[idxs, :, :].float(), dataset.train_labels[idxs])

    return samples


def check_accuracy(model):
    mean = (0.2860405969887955,)
    std = (0.3530242445149223,)
    DEVICE = torch.device("cuda:0")

    test_dataset = datasets.FashionMNIST(
        '../data', train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    )

    test_samples = extract_two_class_dataset_mnist(test_dataset)
    # CNN expects batches to be of shape (n_samples, channels, height_width)
    test_samples = (test_samples[0].unsqueeze(1).to(DEVICE), test_samples[1].to(DEVICE))
    my_test_dataset = MyDataset(test_samples)
    test_data_loader = data.DataLoader(my_test_dataset, 1024)

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in test_data_loader:
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')

    model.train()


def get_train_batches():
    DEVICE = torch.device("cuda:0")

    mean = (0.2860405969887955,)
    std = (0.3530242445149223,)
    train_dataset = datasets.FashionMNIST(
        '../data', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    )

    samples = extract_two_class_dataset_mnist(train_dataset)
    # CNN expects batches to be of shape (n_samples, channels, height_width)
    samples = (samples[0].unsqueeze(1).to(DEVICE), samples[1].to(DEVICE))
    my_dataset = MyDataset(samples)
    # data_loader = data.DataLoader(my_dataset, len(samples[1]))
    data_loader = data.DataLoader(my_dataset, 1024)
    batches = [x for x in data_loader]

    return batches


def test_old():
    DEVICE = torch.device("cuda:0")

    batches = get_train_batches()
    data_gen = cycle(batches)

    model = Net().to(DEVICE)

    check_accuracy(model)

    criterion = nn.CrossEntropyLoss()

    des_optim = BasenDESOptimizerOld(
        model,
        criterion,
        data_gen,
        restarts=2,
        lower=-2.,
        upper=2.,
        budget=100000,
        tol=1e-6,
        nn_train=True,
        lambda_=4000,
        history=16,
        log_best_val=False,
        device=DEVICE
    )

    model = des_optim.run()

    check_accuracy(model)


def test_new():
    DEVICE = torch.device("cuda:0")
    devices = [torch.device("cuda:0")]
    # devices = [torch.device("cuda:0"), torch.device("cuda:1")]

    batches = get_train_batches()
    batches = [(idx, x) for idx, x in enumerate(batches)]

    model = Net().to(DEVICE)

    check_accuracy(model)

    criterion = nn.CrossEntropyLoss()

    des_optim = BasenDESOptimizerNew(
        model,
        criterion,
        None,
        restarts=2,
        lower=-2.,
        upper=2.,
        budget=100000,
        tol=1e-6,
        nn_train=True,
        lambda_=4000,
        history=16,
        log_best_val=False,
        device=DEVICE,
        devices=devices,
        batches=batches
    )

    model = des_optim.run()

    check_accuracy(model)


if __name__ == "__main__":
    _seed_everything()
    # test_new()
    test_old()
    test_old()
