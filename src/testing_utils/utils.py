import random

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def seed_everything():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_fitnesses(fitnesses_iterations):
    fig, axs = plt.subplots(len(fitnesses_iterations), squeeze=False)
    # plt.yscale("log")
    for idx, fitnesses in enumerate(fitnesses_iterations):
        axs[idx, 0].plot(range(len(fitnesses)), fitnesses)
    plt.show()


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
