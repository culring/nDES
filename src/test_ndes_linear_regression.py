import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.utils.data as Data

from timeit import default_timer as timer

from ndes_optimizer_rewrited import BasenDESOptimizer as BasenDESOptimizerNew
from ndes_optimizer_original import BasenDESOptimizer as BasenDESOptimizerOld


DEVICE = torch.device("cuda:0")
DEVICES = [torch.device("cuda:0")]
#DEVICES = [torch.device("cuda:0"), torch.device("cuda:1")]
DRAW_CHARTS = False


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.leaky_relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x


def get_train_data():
    x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1).to(DEVICE)
    y = x.pow(2) + 0.2 * torch.rand(x.size()).to(DEVICE)

    torch_dataset = Data.TensorDataset(x, y)

    loader = Data.DataLoader(dataset=torch_dataset, batch_size=64)

    batches = [(idx, z) for idx, z in enumerate(loader)]
    # data_gen = cycle(batches)
    data_gen = Cycler(batches)

    return batches, data_gen


def get_test_data():
    x = torch.unsqueeze(torch.linspace(-1, 1, 200), dim=1).to(DEVICE)
    y = x.pow(2) + 0.2 * torch.rand(x.size()).to(DEVICE)

    return x, y


class Cycler:
    def __init__(self, batches):
        self.batch_idx = 0
        self.batches = batches

    def __iter__(self):
        return self

    def __next__(self):
        batch_idx = self.batch_idx
        self.batch_idx = (self.batch_idx + 1) % len(self.batches)
        return self.batches[batch_idx]

    def __len__(self):
        return len(self.batches)


def cycle(batches):
    while True:
        for batch in batches:
            yield batch


def _seed_everything():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test():
    _seed_everything()

    train_batches, train_data_gen = get_train_data()
    x_val, y_val = get_test_data()
    kwargs = {
        "criterion": F.mse_loss,
        "budget": 100000,
        # "budget": 500,
        "history": 16,
        "nn_train": True,
        "lower": -2,
        "upper": 2,
        "tol": 1e-6,
        "device": DEVICE,
        "x_val": x_val,
        "y_val": y_val,

        # "use_fitness_ewma": True
    }
    test_func = test_func_wrapper(x_val, y_val)

    # model_old = test_old(train_data_gen, kwargs, test_func)
    model_new = test_new(train_batches, kwargs, test_func)

    old_score = 0.0048
    new_score = eval(model_new, x_val, y_val)
    # old_score = eval(model_old, x_val, y_val)
    # print(old_score, new_score)
    # print(old_score)
    print(new_score)

    if DRAW_CHARTS:
        draw_predictions(model_new, x_val, y_val)
        # draw_predictions(model_old, x_val, y_val)

    assert abs(old_score - new_score) <= 0.0006, "The difference between the scores is too high."


def test_func_wrapper(x_val, y_val):
    def test_func(model):
        return eval(model, x_val, y_val)

    return test_func


def test_old(data_gen, kwargs, test_func=None):
    net = get_model()

    ndes = BasenDESOptimizerOld(
        model=net,
        data_gen=data_gen,
        **kwargs
    )

    best, fitnesses_iterations = ndes.run(test_func)

    if DRAW_CHARTS:
        plot_fitnesses(fitnesses_iterations)

    return best


def test_new(batches, kwargs, test_func=None):
    net = get_model()

    ndes = BasenDESOptimizerNew(
        model=net,
        data_gen=None,
        batches=batches,
        devices=DEVICES,
        **kwargs
    )

    best, fitnesses_iterations = ndes.run(test_func)

    if DRAW_CHARTS:
        plot_fitnesses(fitnesses_iterations)

    return best


def get_model():
    return Net(1, 10, 1).to(DEVICE)


def eval(model, x_val, y_val):
    model.eval()
    with torch.no_grad():
        out = model(x_val)
        loss = F.mse_loss(out, y_val)
    model.train()
    return loss


def draw_predictions(model, x, y):
    x_cpu = x.cpu()
    y_cpu = y.cpu()

    model.eval()
    with torch.no_grad():
        preds = model(x).cpu()
        plt.scatter(x_cpu, y_cpu, c='black')
        plt.scatter(x_cpu, preds, c='red')
        plt.show()
    model.train()


def plot_fitnesses(fitnesses_iterations):
    fig, axs = plt.subplots(len(fitnesses_iterations), squeeze=False)
    # plt.yscale("log")
    for idx, fitnesses in enumerate(fitnesses_iterations):
        axs[idx, 0].plot(range(len(fitnesses)), fitnesses)
    plt.show()


if __name__ == "__main__":
    # import cProfile
    # cProfile.run("test()", "profile_old.txt")

    torch.multiprocessing.set_start_method('spawn')
    begin = timer()
    test()
    end = timer()
    print(end - begin)
