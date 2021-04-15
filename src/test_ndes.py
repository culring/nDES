import numpy as np
import random
from random import randint
import torch
import torch.nn.functional as F

from ndes import NDES
from ndes_optimizer import RNNnDESOptimizer
from utils_rnn import DummyDataGenerator
from rnn_addition_experiment import dataset_generator
from rnn_addition_experiment import Net


def test_one_batch():
    best_model = _optimize_addition_experiment_model(1)
    parameters = best_model.parameters()
    parameter1_expected = np.array(
        [[-0.03167936, 0.15802976, 0.11570834, -0.28565332, 0.25937796, 0.23147143, -0.30402797, 0.04191783]])
    parameter1 = next(parameters).flatten().cpu().numpy()
    assert np.allclose(parameter1, parameter1_expected)


def test_multiple_batches_1():
    best_model = _optimize_addition_experiment_model(2)
    parameters = best_model.parameters()
    parameter1_expected = np.array(
        [[0.20979181, 0.666692, 0.04987898, 0.08414253, 0.04452102, 0.30242744, -0.54285395, -0.11967324]])
    parameter1 = next(parameters).flatten().cpu().numpy()
    assert np.allclose(parameter1, parameter1_expected)


def test_multiple_batches_2():
    best_model = _optimize_addition_experiment_model(10)
    parameters = best_model.parameters()
    parameter1_expected = np.array(
        [[0.20142989, 0.16085166, -1.008347, -0.4765443, -0.62992036, 0.53645694, 0.10092552, 1.034528]])
    parameter1 = next(parameters).flatten().cpu().numpy()
    assert np.allclose(parameter1, parameter1_expected)


def test_multiple_batches_3():
    best_model = _optimize_addition_experiment_model(50)
    parameters = best_model.parameters()
    parameter1_expected = np.array(
        [[0.13170685, 0.04440612, -0.2721198, -0.46091688, -0.32127187, -0.00898296, 0.01116597, 0.10626985]])
    parameter1 = next(parameters).flatten().cpu().numpy()
    assert np.allclose(parameter1, parameter1_expected)


def _optimize_addition_experiment_model(n_batches):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    DEVICE = torch.device("cuda:0")
    sequence_length = 20

    batches = []
    for i in range(n_batches):
        data_generator = DummyDataGenerator(
            *dataset_generator(500, sequence_length, seed=i), DEVICE
        )
        _, batch = next(data_generator)
        batches.append(batch)

    net = Net().to(DEVICE)

    cost_function = F.mse_loss

    ndes = RNNnDESOptimizer(
        model=net,
        criterion=cost_function,
        data_gen=data_generator,
        # budget=1000000,
        budget=1000,
        history=16,
        nn_train=True,
        lower=-2,
        upper=2,
        tol=1e-6,
        worst_fitness=3,
        device=DEVICE,
        devices=[torch.device("cuda:0")],
        batches=batches,
        log_dir=f"rnn_addition_{sequence_length}",
    )

    best_model = ndes.run()

    return best_model


if __name__ == "__main__":
    pass
