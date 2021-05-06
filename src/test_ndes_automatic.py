import numpy as np
import random
from random import randint
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence
from timeit import default_timer as timer

from ndes_optimizer import RNNnDESOptimizer as RNNnDESOptimizerParallelized
from ndes_optimizer_original import RNNnDESOptimizer as RNNnDESOptimizerOriginal
from utils_rnn import DummyDataGenerator
from rnn_addition_experiment import dataset_generator
from rnn_addition_experiment import Net

import ndes
import ndes_original


def test_one_batch():
    best_model = _optimize_addition_experiment_model_original(1)
    parameters = best_model.parameters()
    parameter1_expected = np.array(
        [[-0.03167936, 0.15802976, 0.11570834, -0.28565332, 0.25937796, 0.23147143, -0.30402797, 0.04191783]])
    parameter1 = next(parameters).flatten().cpu().numpy()
    assert np.allclose(parameter1, parameter1_expected)


def test_multiple_batches_1():
    best_model = _optimize_addition_experiment_model_original(2)
    parameters = best_model.parameters()
    parameter1_expected = np.array(
        [[0.20979181, 0.666692, 0.04987898, 0.08414253, 0.04452102, 0.30242744, -0.54285395, -0.11967324]])
    parameter1 = next(parameters).flatten().cpu().numpy()
    assert np.allclose(parameter1, parameter1_expected)


def test_multiple_batches_2():
    best_model_original = _optimize_addition_experiment_model_original(10)
    parameters_original = best_model_original.parameters()
    parameter1_original = next(parameters_original).flatten().cpu().numpy()

    torch.set_printoptions(precision=10)
    for fitness in ndes_original.fitnesses:
        print(fitness)

    # best_model_parallelized = _optimize_addition_experiment_model_parallelized(10)
    # parameters_parallelized = best_model_parallelized.parameters()
    # parameter1_parallelized = next(parameters_parallelized).flatten().cpu().numpy()

    # torch.set_printoptions(precision=10)
    # for fitness in ndes.fitnesses:
    #     print(fitness)

    # print(parameter1_original)
    # print()
    # print(parameter1_parallelized)
    #
    # print()
    # for fitness_parallelized, fitness_original in zip(ndes.fitnesses, ndes_original.fitnesses):
    #     torch.set_printoptions(precision=10)
    #     print(fitness_parallelized)
    #     print(fitness_original)
    #     print(fitness_parallelized == fitness_original)

    # start = timer()
    # best_model = _optimize_addition_experiment_model_original(10)
    # end = timer()
    # print(end - start)
    # parameters = best_model.parameters()
    # parameter1_expected = np.array(
    #     [[0.20142989, 0.16085166, -1.008347, -0.4765443, -0.62992036, 0.53645694, 0.10092552, 1.034528]])
    # parameter1 = next(parameters).flatten().cpu().numpy()
    # assert np.allclose(parameter1, parameter1_expected)


def test_multiple_batches_3():
    best_model = _optimize_addition_experiment_model_original(50)
    parameters = best_model.parameters()
    parameter1_expected = np.array(
        [[0.13170685, 0.04440612, -0.2721198, -0.46091688, -0.32127187, -0.00898296, 0.01116597, 0.10626985]])
    parameter1 = next(parameters).flatten().cpu().numpy()
    assert np.allclose(parameter1, parameter1_expected)


def _optimize_addition_experiment_model_parallelized(n_batches):
    randomStateManager.restore()

    DEVICE = torch.device("cuda:0")
    # devices = [torch.device("cuda:0"), torch.device("cuda:1")]
    # N_DEVICES = 2
    devices = [torch.device("cuda:0")]
    N_DEVICES = 1
    sequence_length = 20

    device_to_batches = {}
    for i in range(N_DEVICES):
        batches = []
        device = torch.device(f"cuda:{i}")
        for i in range(n_batches):
            data_generator = DummyDataGenerator(
                *dataset_generator(500, sequence_length, seed=i, device=device), device
            )
            _, batch = next(data_generator)
            batches.append(batch)
        device_to_batches[str(device)] = batches

    net = Net().to(DEVICE)
    models = dict()
    models[str(DEVICE)] = net
    # for device in devices:
    #     models[str(device)] = Net().to(device)

    cost_function = F.mse_loss

    print(next(models[str(DEVICE)].parameters()).flatten())

    randomStateManager.restore()
    ndes = RNNnDESOptimizerParallelized(
        model=net,
        criterion=cost_function,
        data_gen=None,
        budget=1000,
        # budget=1000,
        history=16,
        nn_train=True,
        lower=-2,
        upper=2,
        tol=1e-6,
        worst_fitness=3,
        device=DEVICE,
        devices=devices,
        batches=batches,
        log_dir=f"rnn_addition_{sequence_length}",
        models=models,
        device_to_batches=device_to_batches
    )

    best_model = ndes.run()

    return best_model


class CycleBatchesLoader:
    def __init__(self, batches):
        self.batches = batches
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        batch = self.batches[self.idx]
        self.idx = (self.idx + 1) % len(self.batches)
        return self.idx - 1, batch

    def __len__(self):
        return len(self.batches)


class RandomStateManager:
    def __init__(self):
        self._random_state = None
        self._torch_state = None
        self._numpy_state = None

    def save(self):
        self._random_state = random.getstate()
        self._torch_state = torch.random.get_rng_state()
        self._numpy_state = np.random.get_state()

    def restore(self):
        random.setstate(self._random_state)
        torch.set_rng_state(self._torch_state)
        np.random.set_state(self._numpy_state)


def _optimize_addition_experiment_model_original(n_batches):
    randomStateManager.restore()

    DEVICE = torch.device("cuda:0")
    sequence_length = 20

    batches = []
    for i in range(n_batches):
        data_generator = DummyDataGenerator(
            *dataset_generator(500, sequence_length, seed=i, device=DEVICE), DEVICE
        )
        _, batch = next(data_generator)
        batches.append(batch)
    data_generator = CycleBatchesLoader(batches)

    net = Net().to(DEVICE)

    cost_function = F.mse_loss

    print(next(net.parameters()).flatten())

    randomStateManager.restore()
    ndes = RNNnDESOptimizerOriginal(
        model=net,
        criterion=cost_function,
        data_gen=data_generator,
        budget=1000,
        history=16,
        nn_train=True,
        lower=-2,
        upper=2,
        tol=1e-6,
        worst_fitness=3,
        device=DEVICE,
        batches=batches,
        log_dir=f"rnn_addition_{sequence_length}"
    )

    best_model = ndes.run()

    return best_model


def _seed_everything():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    _seed_everything()
    randomStateManager = RandomStateManager()
    randomStateManager.save()

    test_multiple_batches_2()
