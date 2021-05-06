import numpy as np
import random
import torch
import torch.nn.functional as F

from ndes_optimizer_new import RNNnDESOptimizer as RNNnDESOptimizerNew
from ndes_optimizer_original import RNNnDESOptimizer as RNNnDESOptimizerOld
from utils_rnn import DummyDataGenerator
from rnn_addition_experiment import dataset_generator
from rnn_addition_experiment import Net

import sys


def test_1_old():
    _seed_everything()
    DEVICE = torch.device("cuda:0")

    sequence_length = 20

    data_generator = DummyDataGenerator(
        *dataset_generator(5000, sequence_length), DEVICE
    )

    net = Net().to(DEVICE)

    cost_function = F.mse_loss

    ndes = RNNnDESOptimizerOld(
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
        log_dir=f"rnn_addition_{sequence_length}",
    )

    best = ndes.run()
    torch.save(best.state_dict(), "model_1_old.pt")


def test_1_new():
    _seed_everything()
    DEVICE = torch.device("cuda:0")

    sequence_length = 20

    data_generator = DummyDataGenerator(
        *dataset_generator(5000, sequence_length), DEVICE
    )

    net = Net().to(DEVICE)

    cost_function = F.mse_loss

    ndes = RNNnDESOptimizerNew(
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
        log_dir=f"rnn_addition_{sequence_length}",
    )

    best = ndes.run()
    torch.save(best.state_dict(), "model_1_new.pt")


def test_1():
    model_old = Net()
    model_old.load_state_dict(torch.load(f"model_1_old.pt"))

    model_new = Net()
    model_new.load_state_dict(torch.load(f"model_1_new.pt"))

    compare_models(model_old, model_new)


def test_2():
    model_old = Net()
    model_old.load_state_dict(torch.load("model_2_old.pt"))

    model_new = Net()
    model_new.load_state_dict(torch.load("model_2_new.pt"))

    compare_models(model_old, model_new)


def cycle(batches):
    while True:
        for batch in batches:
            yield batch


def test_multiple_batches_old(number_of_batches, filename):
    _seed_everything()
    DEVICE = torch.device("cuda:0")

    sequence_length = 20

    batches = []
    for i in range(number_of_batches):
        data_generator = DummyDataGenerator(
            *dataset_generator(5000, sequence_length), DEVICE
        )
        batch = next(data_generator)
        batches.append(batch)
    data_generator = cycle(batches)

    net = Net().to(DEVICE)

    cost_function = F.mse_loss

    ndes = RNNnDESOptimizerOld(
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
        log_dir=f"rnn_addition_{sequence_length}",
    )

    best = ndes.run()
    torch.save(best.state_dict(), filename)


def test_2_old():
    test_multiple_batches_old(2, "model_2_old.pt")


def test_multiple_batches_new(number_of_batches, filename):
    _seed_everything()
    DEVICE = torch.device("cuda:0")

    sequence_length = 20

    batches = []
    for i in range(number_of_batches):
        data_generator = DummyDataGenerator(
            *dataset_generator(5000, sequence_length), DEVICE
        )
        batch = next(data_generator)
        batches.append(batch)
    data_generator = cycle(batches)

    net = Net().to(DEVICE)

    cost_function = F.mse_loss

    ndes = RNNnDESOptimizerNew(
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
        log_dir=f"rnn_addition_{sequence_length}",
    )

    best = ndes.run()
    torch.save(best.state_dict(), filename)


def test_2_new():
    test_multiple_batches_new(2, "model_2_new.pt")


def _seed_everything():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')


def hello():
    print("hello")


if __name__ == "__main__":
    method_to_call = getattr(sys.modules[__name__], sys.argv[1])
    method_to_call()
