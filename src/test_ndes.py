import torch

from ndes import NDES


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
