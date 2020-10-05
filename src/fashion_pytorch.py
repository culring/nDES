import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.metrics import functional as FM
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
from math import ceil

from des_torch import DESOptimizer
from utils import bootstrap, train_via_des, seed_everything
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

#  EPOCHS = 25000
POPULATION_MULTIPLIER = 8
#  EPOCHS = int(POPULATION_MULTIPLIER * 4 * 1200000)
EPOCHS = int(POPULATION_MULTIPLIER * 4000 * 1200)
POPULATION = int(POPULATION_MULTIPLIER * 4000)
DES_TRAINING = True

DEVICE = torch.device("cuda:0")
BOOTSTRAP_BATCHES = True
MODEL_NAME = "fashion_des_bootstrapped"
LOAD_WEIGHTS = False
SEED_OFFSET = 0
BATCH_SIZE = 64
VALIDATION_SIZE = 10000
STRATIFY = True


class Net(pl.LightningModule):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 32, 5)
        self.fc1 = nn.Linear(4 * 4 * 32, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.softsign(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.softsign(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 32)
        x = F.softsign(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def prepare_data(self):
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
        self.test_dataset = datasets.FashionMNIST(
            "../data",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)]
            ),
        )

        for x, y in torch.utils.data.DataLoader(
            train_dataset, batch_size=len(train_dataset), shuffle=True
        ):
            x_train, y_train = x.to(DEVICE), y.to(DEVICE)

        train_idx, val_idx = train_test_split(
            np.arange(0, len(train_dataset)),
            test_size=VALIDATION_SIZE,
            stratify=y_train.cpu().numpy(),
        )

        x_val = x_train[val_idx, :]
        y_val = y_train[val_idx]
        x_train = x_train[train_idx, :]
        y_train = y_train[train_idx]

        self.train_dataset = TensorDataset(x_train, y_train)
        self.val_dataset = TensorDataset(x_val, y_val)

        return x_train, y_train, x_val, y_val

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=64, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=64
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=64
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.nll_loss(self(x), y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        # identifying number of correct predections in a given batch
        correct=y_hat.argmax(dim=1).eq(y).sum().item()

        # identifying total number of labels in a given batch
        total=len(y)

        loss = F.nll_loss(y_hat, y)
        return {
            'val_loss': loss,
            "correct": correct,
            "total": total
        }

    def test_step(self, batch, batch_idx):
        result = self.validation_step(batch, batch_idx)
        return result

    def validation_epoch_end(self, outputs):
        # called at the end of a validation epoch
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}]
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        correct=sum([x["correct"] for  x in outputs])
        total=sum([x["total"] for  x in outputs])
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': correct / total}
        return {'avg_val_loss': avg_loss, 'val_acc': correct / total, 'log': tensorboard_logs}

    def test_epoch_end(self, outputs):
        result = self.validation_epoch_end(outputs)
        for old_key, new_key in {'val_acc': 'test_acc', 'avg_val_loss':
                                 'avg_test_loss'}.items():
            result[new_key] = result.pop(old_key)
        return result


def cycle(loader):
    while True:
        for element in enumerate(loader):
            yield element


class MyDatasetLoader:
    def __init__(self, x_train, y_train, batch_size):
        self.x_train = x_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.num_batches = int(ceil(x_train.shape[0] / batch_size))

    def cycle(self):
        while True:
            for i in range(self.num_batches - 1):
                idx = i * self.batch_size
                yield (i, (self.x_train[idx:idx+self.batch_size],
                       self.y_train[idx:idx+self.batch_size]))
            idx = (self.num_batches - 1) * self.batch_size
            yield (self.num_batches - 1, (self.x_train[idx:], self.y_train[idx:]))


if __name__ == "__main__":
    seed_everything(SEED_OFFSET)

    model = Net().to(DEVICE)
    if LOAD_WEIGHTS:
        model.load_state_dict(torch.load(MODEL_NAME)["state_dict"])

    if DES_TRAINING:
        x_train, y_train, x_val, y_val = model.prepare_data()
        test_dataset = model.test_dataset

        if STRATIFY:
            indices_x = []
            indices_y = []
            splitter = StratifiedKFold(
                n_splits=ceil(len(x_train) / BATCH_SIZE),
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

        if BOOTSTRAP_BATCHES is not None:
            early_stop_callback = EarlyStopping(
               monitor='val_loss',
               min_delta=0.00,
               patience=3,
               verbose=False,
               mode='min'
            )
            trainer = Trainer(gpus=1, early_stop_callback=early_stop_callback)
            trainer.fit(model)
            trainer.test(model)
        print(f"Num params: {sum([param.nelement() for param in model.parameters()])}")
        torch.save({"state_dict": model.state_dict()}, "boostrap_adam.pth.tar")

        criterion = nn.CrossEntropyLoss()
        train_loader = MyDatasetLoader(x_train, y_train, BATCH_SIZE)
        des_optim = DESOptimizer(
            model,
            criterion,
            train_loader.cycle(),
            None,
            ewma_alpha=0.3,
            num_batches=train_loader.num_batches,
            x_val=x_val,
            y_val=y_val,
            restarts=None,
            Ft=1,
            ccum=0.96,
            # cp=0.1,
            lower=-2.0,
            upper=2.0,
            budget=EPOCHS,
            tol=1e-6,
            nn_train=True,
            lambda_=POPULATION,
            history=16,
            worst_fitness=3,
            device=DEVICE,
        )
        train_via_des(model, des_optim, DEVICE, test_dataset, MODEL_NAME)
    else:
        early_stop_callback = EarlyStopping(
           monitor='val_loss',
           min_delta=0.00,
           patience=3,
           verbose=False,
           mode='min'
        )
        trainer = Trainer(gpus=1, early_stop_callback=early_stop_callback)
        trainer.fit(model)
        trainer.test(model)
