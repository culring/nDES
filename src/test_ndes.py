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


def test_lamarckian_fitness_population():
    device = torch.device("cuda:0")
    x = initial_value = torch.ones(3, 2, device=device)
    fn = torch.sum
    lower = 0
    upper = 2
    population_initializer = None
    ndes = NDES(initial_value, fn, lower, upper, population_initializer, device=device)
    reference = torch.tensor([3.0, 3.0], device=device)
    fitness = ndes._fitness_lamarckian(x)
    assert fitness.equal(reference)

def test_lamarckian_fitness_population_low_budget():
    device = torch.device("cuda:0")
    x = initial_value = torch.ones(3, 2, device=device)
    fn = torch.sum
    lower = 0
    upper = 2
    population_initializer = None
    kwargs = {"budget": 1, "worst_fitness": 10, "device": device}
    ndes = NDES(initial_value, fn, lower, upper, population_initializer, **kwargs)
    reference = torch.tensor([3.0, 10.0], device=device)
    fitness = ndes._fitness_lamarckian(x)
    assert fitness.equal(reference)

def test_lamarckian_fitness_single_individual():
    device = torch.device("cuda:0")
    x = initial_value = torch.ones(3, 1, device=device)
    fn = torch.sum
    lower = 0
    upper = 2
    population_initializer = None
    ndes = NDES(initial_value, fn, lower, upper, population_initializer, device=device)
    reference = torch.tensor(3.0, device=device)
    fitness = ndes._fitness_lamarckian(x)
    assert fitness.equal(reference)

def test_lamarckian_fitness_single_individual_no_budget():
    device = torch.device("cuda:0")
    x = initial_value = torch.ones(3, 1, device=device)
    fn = torch.sum
    lower = 0
    upper = 2
    population_initializer = None
    kwargs = {"budget": 0, "worst_fitness": 10, "device": device}
    ndes = NDES(initial_value, fn, lower, upper, population_initializer, **kwargs)
    fitness = ndes._fitness_lamarckian(x)
    assert fitness == 10

def test_ndes():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    DEVICE = torch.device("cuda:0")
    sequence_length = 20

    data_generator = DummyDataGenerator(
        *dataset_generator(5000, sequence_length, deterministic=True), DEVICE
    )

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
        devices=[torch.device("cuda:0"), torch.device("cpu")],
        log_dir=f"rnn_addition_{sequence_length}",
    )

    best = ndes.run()

    parameters = best.parameters()
    parameter1_expected = np.array([-0.18890078, -0.05024691, 0.0426968, -0.02934495, -0.070004, 0.05936356, 0.2720328,  -0.32246602])
    parameter1 = next(parameters).flatten().cpu().numpy()
    assert np.allclose(parameter1, parameter1_expected)
