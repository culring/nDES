import torch
import torch.nn as nn

import os
import random
import string
import sys

from ndes.ndes_optimizer import NDESOptimizer as NDESOptimizerNew
import ndes
from testing_utils.two_class_mnist import Net, check_accuracy, test_func_wrapper
from testing_utils.two_class_mnist import get_test_data
from testing_utils.two_class_mnist import get_train_batches
from testing_utils.utils import plot_fitnesses

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
    model, logs = test_new(train_batches, train_data_gen, kwargs, test_func)

    return model, logs


def test():
    train_batches, train_data_gen = get_train_batches()
    test_data = get_test_data()
    num_tests = 20

    for i in range(num_tests):
        model, logs = single_test(train_batches, train_data_gen, test_data)
        identifier = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        torch.save(model.state_dict(), f"models/{identifier}_model.pt")
        torch.save(logs, f"models/{identifier}_logs.pt")


def evaluate_models():
    directory_filenames = os.listdir("models/")
    for filename in directory_filenames:
        if filename.find("model") == -1:
            continue
        identifier = filename[:6]
        state_dict = torch.load(f"models/{filename}")
        model = Net().to(DEVICE)
        model.load_state_dict(state_dict)

        logs_filename = identifier + "_logs.pt"
        logs = torch.load(f"models/{logs_filename}")
        plot_fitnesses([logs[0][0], logs[1][0]])
        plot_fitnesses([logs[0][1], logs[1][1]])
        plot_fitnesses([logs[0][2], logs[1][2]])


if __name__ == "__main__":
    machine = sys.argv[1]
    if machine == "pc":
        NODES = [ndes.GPUNode(torch.device(0))]
    elif machine == "server":
        NODES = [ndes.GPUNode(torch.device(0)), ndes.GPUNode(torch.device(1))]
    torch.multiprocessing.set_start_method('spawn')
    test()
    # evaluate_models()
