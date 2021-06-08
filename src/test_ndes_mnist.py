import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import datasets, transforms

import random
from timeit import default_timer as timer

from ndes_optimizer_original import BasenDESOptimizer as BasenDESOptimizerOld
from ndes_optimizer_rewrited import BasenDESOptimizer as BasenDESOptimizerNew


DEVICE = torch.device("cuda:0")
DEVICES = [torch.device("cuda:0")]
DRAW_CHARTS = False


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
        for batch in batches:
            yield batch


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


def get_test_data():
    mean = (0.2860405969887955,)
    std = (0.3530242445149223,)

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

    # return data.DataLoader(my_test_dataset, 1024)
    test_data_loader = data.DataLoader(my_test_dataset, len(my_test_dataset))
    x_test, y_test = next(iter(test_data_loader))
    return x_test, y_test


def check_accuracy(model, x_val, y_val):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        scores = model(x_val)
        _, predictions = scores.max(1)
        num_correct += (predictions == y_val).sum()
        num_samples += predictions.size(0)
        # print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')
    model.train()

    return float(num_correct) / float(num_samples) * 100


def get_train_batches():
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
    batches = [(idx, x) for idx, x in enumerate(data_loader)]
    data_gen = cycle(batches)

    return batches, data_gen


def plot_fitnesses(fitnesses_iterations):
    fig, axs = plt.subplots(len(fitnesses_iterations), squeeze=False)
    # plt.yscale("log")
    for idx, fitnesses in enumerate(fitnesses_iterations):
        axs[idx, 0].plot(range(len(fitnesses)), fitnesses)
    plt.show()


def test_old(data_gen, kwargs, test_func=None):
    model = Net().to(DEVICE)

    des_optim = BasenDESOptimizerOld(
        model=model,
        data_gen=data_gen,
        **kwargs
    )

    best, fitnesses_iterations = des_optim.run(test_func)

    if DRAW_CHARTS:
        plot_fitnesses(fitnesses_iterations)

    return best


def test_new(batches, kwargs, test_func=None):
    model = Net().to(DEVICE)

    des_optim = BasenDESOptimizerNew(
        model=model,
        data_gen=None,
        batches=batches,
        devices=DEVICES,
        **kwargs
    )

    best, fitnesses_iterations = des_optim.run(test_func)

    if DRAW_CHARTS:
        plot_fitnesses(fitnesses_iterations)

    return best


def test_func_wrapper(x_val, y_val):
    criterion = nn.CrossEntropyLoss()

    def test_func(model):
        model.eval()
        with torch.no_grad():
            out = model(x_val)
            loss = criterion(out, y_val)
        model.train()
        return loss

    return test_func


def test():
    _seed_everything()

    x_val, y_val = get_test_data()
    kwargs = {
        "restarts": 2,
        "criterion": nn.CrossEntropyLoss(),
        "budget": 100000,
        "history": 16,
        "nn_train": True,
        "lower": -2,
        "upper": 2,
        "tol": 1e-6,
        "worst_fitness": 3,
        "device": DEVICE,
        "lambda_": 4000,
        "x_val": x_val,
        "y_val": y_val
    }

    train_batches, train_data_gen = get_train_batches()
    test_func = test_func_wrapper(x_val, y_val)
    # model_old = test_old(train_data_gen, kwargs, test_func)
    model_new = test_new(train_batches, kwargs, test_func)

    accuracy_old = 95.8
    # accuracy_old = check_accuracy(model_old, x_val, y_val)
    accuracy_new = check_accuracy(model_new, x_val, y_val)

    assert abs(accuracy_old - accuracy_new) < 0.5, "Models don't match"


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    begin = timer()
    test()
    end = timer()
    print(end - begin)
