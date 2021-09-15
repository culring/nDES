import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import datasets, transforms

from test_utils import Cycler


DEVICE = torch.device("cuda:0")


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
    data_gen = Cycler(batches)

    return batches, data_gen


def test_func_wrapper(x_val, y_val):
    criterion = nn.CrossEntropyLoss()

    def test_func(model):
        model.eval()
        with torch.no_grad():
            out = model(x_val)
            loss = criterion(out, y_val).item()
        model.train()

        acc = check_accuracy(model, x_val, y_val)

        return loss, acc

    return test_func
