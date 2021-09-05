import numpy as np
import matplotlib.pyplot as plt
import random
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence

from random import randint
import sys
from timeit import default_timer as timer

from ndes import RNNnDESOptimizer as RNNnDESOptimizerNew
from ndes_optimizer_original import RNNnDESOptimizer as RNNnDESOptimizerOld
from utils_rnn import DummyDataGenerator
import ndes


DEVICE = torch.device("cuda:0")
DEVICES = [torch.device("cuda:0")]
SEQUENCE_LENGTH = 20


def cycle(batches):
    while True:
        for x in batches:
            yield x


def dataset_generator(num_samples, min_sample_length, seed=0, device=DEVICE):
    dataset = []
    gts = []
    max_sample_length = min_sample_length + int(min_sample_length / 10)
    sizes = (
        np.random.default_rng(seed)
        .uniform(min_sample_length, max_sample_length, num_samples)
        .astype(int)
    )
    for size in sizes:
        values = np.random.default_rng(seed).uniform(-1, 1, size)
        x1_index = randint(0, min(10, size - 1))
        x2_index = x1_index
        while x2_index == x1_index:
            x2_index = randint(0, int(min_sample_length / 2) - 1)

        mask = np.zeros_like(values)
        mask[x1_index] = 1
        mask[x2_index] = 1
        mask[0] = -1
        mask[-1] = -1

        x1 = values[x1_index] if x1_index != 0 else 0
        x2 = values[x2_index] if x2_index != 0 else 0
        gt = 0.5 + ((x1 + x2) / 4)
        dataset.append(torch.tensor((values, mask)).permute(1, 0).float().to(device))
        gts.append(gt)
    return dataset, sizes.tolist(), torch.tensor(gts).float().to(device)


class Net(pl.LightningModule):
    def __init__(self):
        super(Net, self).__init__()
        self.rnn = nn.RNN(2, 4, batch_first=True)
        self.output = nn.Linear(4, 1)

    def forward(self, x, hidden=None):
        out, _ = self.rnn(x, hidden)
        seq_unpacked, lens_unpacked = pad_packed_sequence(out, batch_first=True)
        lens_unpacked -= 1
        lens = lens_unpacked.unsqueeze(-1)
        indices = lens.repeat(1, 4)
        self.indices = indices.unsqueeze(1).to(x.data.device)
        out = torch.gather(seq_unpacked, 1, self.indices)
        return self.output(out).flatten()

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.mse_loss(self(x), y)
        # tensorboard_logs = {'train_loss': loss}
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


def generate_dataset(num_batches, num_samples=5000):
    batches = []
    for i in range(num_batches):
        data_generator = DummyDataGenerator(
            *dataset_generator(num_samples, SEQUENCE_LENGTH), DEVICE
        )
        batches.append(next(data_generator))

    return batches, cycle(batches)


def plot_fitnesses(fitnesses_iterations):
    fig, axs = plt.subplots(len(fitnesses_iterations), squeeze=False)
    # plt.yscale("log")
    for idx, fitnesses in enumerate(fitnesses_iterations):
        axs[idx, 0].plot(range(len(fitnesses)), fitnesses)
    plt.show()


def eval(model, test_data):
    _, (x, y) = test_data
    out = model(x)
    return F.mse_loss(out, y).item()


def _seed_everything():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test_func_wrapper(x_val, y_val):
    def test_func(model):
        model.eval()
        with torch.no_grad():
            out = model(x_val)
            loss = F.mse_loss(out, y_val)
        model.train()
        return loss

    return test_func


def test_rnn_addition_old(data_gen, kwargs, test_func=None):
    net = Net().to(DEVICE)

    ndes = RNNnDESOptimizerOld(
        model=net,
        data_gen=data_gen,
        **kwargs
    )

    best, fitnesses_iterations = ndes.run(test_func)

    plot_fitnesses(fitnesses_iterations)

    return best


def test_rnn_addition_new(batches, data_gen, kwargs, test_func=None):
    net = Net().to(DEVICE)

    ndes = RNNnDESOptimizerNew(
        model=net,
        data_gen=data_gen,
        batches=batches,
        nodes=NODES,
        **kwargs
    )

    best, fitnesses_iterations = ndes.run(test_func)

    plot_fitnesses(fitnesses_iterations)

    return best


def test_rnn_addition_single_batch_generic():
    _seed_everything()

    budget = 72000
    kwargs = {
        "criterion": F.mse_loss,
        "budget": budget,
        "history": 16,
        "nn_train": True,
        "lower": -2,
        "upper": 2,
        "tol": 1e-6,
        "worst_fitness": 3,
        "device": DEVICE,
        "log_dir": f"rnn_addition_{SEQUENCE_LENGTH}",
        "num_batches_on_device": 1,
    }

    batches, data_gen = generate_dataset(1)

    # model_old = test_rnn_addition_single_batch_old(data_gen)
    model_new = test_rnn_addition_new(batches, data_gen, kwargs)

    test_batches, _ = generate_dataset(1, 1000)
    accuracy_old = 0.0082
    # accuracy_old = eval(model_old, test_batches[0], criterion)
    # print(eval_old)
    accuracy_new = eval(model_new, test_batches[0])
    print(accuracy_old, accuracy_new)

    assert abs(accuracy_new - accuracy_old) < 0.0005, \
        f"Model don't match: old_acc = {accuracy_old}, new_acc = {accuracy_new}"


def test_rnn_addition_two_batches_generic():
    _seed_everything()

    budget = 60000
    test_batches, _ = generate_dataset(1, 1000)
    _, (x_val, y_val) = test_batches[0]
    kwargs = {
        "criterion": F.mse_loss,
        "budget": budget,
        "history": 16,
        "nn_train": True,
        "lower": -2,
        "upper": 2,
        "tol": 1e-6,
        "worst_fitness": 3,
        "device": DEVICE,
        "log_dir": f"rnn_addition_{SEQUENCE_LENGTH}",
        "x_val": x_val,
        "y_val": y_val
    }

    batches, data_gen = generate_dataset(2, 5000)

    test_func = test_func_wrapper(x_val, y_val)
    # model_old = test_rnn_addition_old(data_gen, kwargs, test_func)
    model_new = test_rnn_addition_new(batches, kwargs, test_func)

    accuracy_old = 0.0095
    # accuracy_old = eval(model_old, test_batches[0])
    # print(eval_old)
    accuracy_new = eval(model_new, test_batches[0])
    print(accuracy_old, accuracy_new)

    assert abs(accuracy_new - accuracy_old) < 0.0005, \
        f"Model don't match: old_acc = {accuracy_old}, new_acc = {accuracy_new}"


if __name__ == "__main__":
    machine = sys.argv[1]
    if machine == "pc":
        NODES = [ndes.GPUNode(torch.device(0))]
    elif machine == "server":
        NODES = [ndes.GPUNode(torch.device(0)), ndes.GPUNode(torch.device(1))]

    torch.multiprocessing.set_start_method('spawn')
    begin = timer()
    test_rnn_addition_single_batch_generic()
    end = timer()
    print(end - begin)
