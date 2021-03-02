import numpy as np
import random
import torch
import torch.nn.functional as F

from fashion_mnist_experiment import train as train_mnist
from ndes_optimizer import RNNnDESOptimizer
from utils_rnn import DummyDataGenerator
from rnn_addition_experiment import dataset_generator
from rnn_addition_experiment import Net


def test_ndes_rnn():
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
        devices=[torch.device("cuda:0")],
        log_dir=f"rnn_addition_{sequence_length}",
    )

    best = ndes.run()

    parameters = best.parameters()
    parameter1_expected = np.array([-0.18890078, -0.05024691, 0.0426968, -0.02934495, -0.070004, 0.05936356, 0.2720328,  -0.32246602])
    parameter1 = next(parameters).flatten().cpu().numpy()
    assert np.allclose(parameter1, parameter1_expected)


def test_ndes_fashion_mnist():
    model = train_mnist()
    print(next(model.parameters()))

# test_ndes_rnn()
test_ndes_fashion_mnist()
