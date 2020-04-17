import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from des_torch import DESOptimizer
from utils import bootstrap, train_via_des, train_via_gradient, seed_everything

#  EPOCHS = 25000
EPOCHS = 600000
DES_TRAINING = True
#  DEVICE = torch.device("cpu")
DEVICE = torch.device("cuda:0")
BOOTSTRAP_BATCHES = None
MODEL_NAME = "fashion_des.pth.tar"
LOAD_WEIGHTS = False
SEED_OFFSET = 0
BATCH_SIZE = 1000


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 32, 5)
        self.fc1 = nn.Linear(4 * 4 * 32, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        #  x = F.leaky_relu(self.conv1(x), 0.1)
        x = torch.tanh(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        #  x = F.leaky_relu(self.conv2(x), 0.1)
        x = torch.tanh(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 32)
        #  x = F.leaky_relu(self.fc1(x), 0.1)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    seed_everything(SEED_OFFSET)

    mean = (0.2860405969887955,)
    std = (0.3530242445149223,)
    train_dataset = datasets.FashionMNIST(
        "../data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        ),
    )
    test_dataset = datasets.FashionMNIST(
        "../data",
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        ),
    )

    model = Net().to(DEVICE)
    if LOAD_WEIGHTS:
        model.load_state_dict(torch.load(MODEL_NAME)["state_dict"])
    if BOOTSTRAP_BATCHES is not None:
        model = bootstrap(model, DEVICE, num_batches=BOOTSTRAP_BATCHES)
    print(f"Num params: {sum([param.nelement() for param in model.parameters()])}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    if DES_TRAINING:
        for x, y in torch.utils.data.DataLoader(
            train_dataset, batch_size=len(train_dataset), shuffle=True
        ):
            #  print(f"Before dataset: {torch.cuda.memory_allocated(DEVICE) / (1024**3)}")
            x_train, y_train = x.to(DEVICE), y.to(DEVICE)
            #  print(f"After dataset: {torch.cuda.memory_allocated(DEVICE) / (1024**3)}")

        if BATCH_SIZE is not None:
            x_train = x_train[:BATCH_SIZE, :]
            y_train = y_train[:BATCH_SIZE]

        des_optim = DESOptimizer(
            model,
            criterion,
            x_train,
            y_train,
            restarts=6,
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
        train_via_des(model, des_optim, DEVICE, test_dataset, MODEL_NAME)
    else:
        train_via_gradient(
            model, criterion, optimizer, train_dataset, test_dataset, EPOCHS, DEVICE
        )
