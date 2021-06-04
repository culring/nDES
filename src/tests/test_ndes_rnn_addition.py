import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn.functional as F

from ndes_optimizer_rewrited import RNNnDESOptimizer as RNNnDESOptimizerNew
from ndes_optimizer_original import RNNnDESOptimizer as RNNnDESOptimizerOld
from utils_rnn import DummyDataGenerator
from rnn_addition_experiment import dataset_generator
from rnn_addition_experiment import Net


DEVICE = torch.device("cuda:0")
DEVICES = [torch.device("cuda:0")]
#DEVICES = [torch.device("cuda:0"), torch.device("cuda:1")]
SEQUENCE_LENGTH = 20


def cycle(batches):
    while True:
        for batch in batches:
            yield batch


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
        batches=None,
        **kwargs
    )

    best, fitnesses_iterations = ndes.run(test_func)

    plot_fitnesses(fitnesses_iterations)

    return best


def test_rnn_addition_new(batches, kwargs, test_func=None):
    net = Net().to(DEVICE)

    ndes = RNNnDESOptimizerNew(
        model=net,
        data_gen=None,
        batches=batches,
        devices=DEVICES,
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
        "log_dir": f"rnn_addition_{SEQUENCE_LENGTH}"
    }

    batches, data_gen = generate_dataset(1)

    # model_old = test_rnn_addition_single_batch_old(data_gen)
    model_new = test_rnn_addition_new(batches, kwargs)

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
    test_rnn_addition_single_batch_generic()
