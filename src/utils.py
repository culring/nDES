import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.nn as nn
import random
import numpy as np


def seed_everything(offset=0):
    torch.manual_seed(1235 + offset)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(42 + offset)
    np.random.seed(42 + offset)


def seconds_to_human_readable(seconds):
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    if seconds < 1:
        seconds = 1
    locals_ = locals()
    magnitudes_str = (
        "{n} {magnitude}".format(n=int(locals_[magnitude]), magnitude=magnitude)
        for magnitude in ("days", "hours", "minutes", "seconds")
        if locals_[magnitude]
    )
    eta_str = ", ".join(magnitudes_str)
    return eta_str


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
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def bootstrap(model, train_dataset, device, num_batches=10):
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    cnt = 0
    while True:
        cnt += 1
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        if cnt >= num_batches:
            print("Loss after bootstrap: {:.6f}".format(loss.item()))
            return model


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return test_loss, 100.0 * correct / len(test_loader.dataset),


def train_via_des(model, des, device, test_dataset, model_name):
    model.eval()
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1000, shuffle=True
    )

    model = des.run(lambda x: test(x, device, test_loader))
    test(model, device, test_loader)
    torch.save({"state_dict": model.state_dict()}, f"{model_name}_{des.start}.pth.tar")


def train_via_gradient(
    model, criterion, optimizer, train_dataset, test_dataset, num_epoch, device
):
    model.train()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1000, shuffle=True
    )

    for epoch in range(1, num_epoch + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
