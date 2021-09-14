import torch
import torch.nn as nn

import random
import string
import sys

from ndes.ndes_optimizer import NDESOptimizer as NDESOptimizerNew
import ndes
from testing_utils.two_class_mnist import Net
from testing_utils.two_class_mnist import get_test_data
from testing_utils.two_class_mnist import get_train_batches
from testing_utils.utils import test_func_wrapper

DEVICE = torch.device("cuda:0")
DRAW_CHARTS = False


def test_new(batches, data_gen, kwargs, test_func=None):
    model = Net().to(DEVICE)

    des_optim = NDESOptimizerNew(
        model=model,
        data_gen=data_gen,
        batches=batches,
        nodes=NODES,
        **kwargs
    )

    model, fitness_iterations = des_optim.run(test_func)

    return model, fitness_iterations


def single_test(train_batches, train_data_gen, test_data):
    x_val, y_val = test_data
    kwargs = {
        "restarts": 2,
        "criterion": nn.CrossEntropyLoss(),
        # "budget": 100000,
        "budget": 3000,
        "history": 16,
        "nn_train": True,
        "lower": -2,
        "upper": 2,
        "tol": 1e-6,
        "worst_fitness": 3,
        "device": DEVICE,
        "lambda_": 500,
        "x_val": x_val,
        "y_val": y_val,
        "use_fitness_ewma": True
    }
    test_func = test_func_wrapper(x_val, y_val)
    model, fitness_iterations = test_new(train_batches, train_data_gen, kwargs, test_func)

    return model, fitness_iterations


def test():
    train_batches, train_data_gen = get_train_batches()
    test_data = get_test_data()
    num_tests = 20

    for i in range(num_tests):
        model, fitness_iterations = single_test(train_batches, train_data_gen, test_data)
        identifier = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        torch.save(model.state_dict(), f"models/{identifier}_model.pt")
        torch.save(fitness_iterations, f"models/{identifier}_fitness_iterations.pt")


if __name__ == "__main__":
    machine = sys.argv[1]
    if machine == "pc":
        NODES = [ndes.GPUNode(torch.device(0))]
    elif machine == "server":
        NODES = [ndes.GPUNode(torch.device(0)), ndes.GPUNode(torch.device(1))]
    torch.multiprocessing.set_start_method('spawn')
    test()
