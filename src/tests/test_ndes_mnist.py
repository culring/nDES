import torch
import torch.nn as nn

import sys
from timeit import default_timer as timer

from ndes.ndes_optimizer import NDESOptimizer as NDESOptimizerNew
from ndes_optimizer_original import BasenDESOptimizer as BasenDESOptimizerOld
import ndes
from testing_utils.two_class_mnist import check_accuracy, test_func_wrapper
from testing_utils.two_class_mnist import Net
from testing_utils.two_class_mnist import get_test_data
from testing_utils.two_class_mnist import get_train_batches
from testing_utils.utils import plot_fitnesses, seed_everything

DEVICE = torch.device("cuda:0")
DRAW_CHARTS = False


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


def test_new(batches, data_gen, kwargs, test_func=None):
    model = Net().to(DEVICE)

    des_optim = NDESOptimizerNew(
        model=model,
        data_gen=data_gen,
        batches=batches,
        nodes=NODES,
        **kwargs
    )

    best, logs = des_optim.run(test_func)

    if DRAW_CHARTS:
        plot_fitnesses([logs[0][0], logs[1][0]])
        plot_fitnesses([logs[0][1], logs[1][1]])
        plot_fitnesses([logs[0][2], logs[1][2]])

    return best


def test():
    seed_everything()

    x_val, y_val = get_test_data()
    kwargs = {
        "restarts": 2,
        "criterion": nn.CrossEntropyLoss(),
        "budget": 100000,
        # "budget": 3000,
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

    train_batches, train_data_gen = get_train_batches()
    test_func = test_func_wrapper(x_val, y_val)
    # model_old = test_old(train_data_gen, kwargs, test_func)
    model_new = test_new(train_batches, train_data_gen, kwargs, test_func)

    accuracy_old = 98.75
    # accuracy_old = check_accuracy(model_old, x_val, y_val)
    accuracy_new = check_accuracy(model_new, x_val, y_val)

    print(f"accuracy_old = {accuracy_old}, accuracy_new = {accuracy_new}")

    assert abs(accuracy_old - accuracy_new) < 0.5, "Models don't match"


if __name__ == "__main__":
    machine = sys.argv[1]
    if machine == "pc":
        NODES = [ndes.GPUNode(torch.device(0))]
    elif machine == "server":
        NODES = [ndes.GPUNode(torch.device(0)), ndes.GPUNode(torch.device(1))]

    torch.multiprocessing.set_start_method('spawn')
    begin = timer()
    test()
    end = timer()
    print(end - begin)
