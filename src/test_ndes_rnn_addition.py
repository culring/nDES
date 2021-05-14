import numpy as np
import random
import torch
import torch.nn.functional as F

from ndes_optimizer_rewrited import RNNnDESOptimizer as RNNnDESOptimizerNew
from ndes_optimizer_original import RNNnDESOptimizer as RNNnDESOptimizerOld
from utils_rnn import DummyDataGenerator
from rnn_addition_experiment import dataset_generator
from rnn_addition_experiment import Net


def test_rnn_addition_single_batch_old():
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
        budget=55000,
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

    _, (x, y) = next(data_generator)
    best_fitness = eval(best, x, y, cost_function)

    return best_fitness
    # torch.save(best.state_dict(), "model_1_old.pt")


def test_rnn_addition_single_batch_new():
    _seed_everything()
    DEVICE = torch.device("cuda:0")
    devices = [torch.device("cuda:0")]
    #devices = [torch.device("cuda:0"), torch.device("cuda:1")]

    sequence_length = 20

    data_generator = DummyDataGenerator(
        *dataset_generator(5000, sequence_length), DEVICE
    )
    batches = [next(data_generator)]

    net = Net().to(DEVICE)

    cost_function = F.mse_loss

    ndes = RNNnDESOptimizerNew(
        model=net,
        criterion=cost_function,
        data_gen=None,
        budget=55000,
        history=16,
        nn_train=True,
        lower=-2,
        upper=2,
        tol=1e-6,
        worst_fitness=3,
        device=DEVICE,
        log_dir=f"rnn_addition_{sequence_length}",
        batches=batches,
        devices=devices
    )

    best = ndes.run()

    _, (x, y) = next(data_generator)
    best_fitness = eval(best, x, y, cost_function)

    return best_fitness
    # torch.save(best.state_dict(), "model_1_new.pt")


def eval(model, x, y, criterion):
    out = model(x)
    return criterion(out, y).item()


def test_rnn_addition_single_batch():
    accuracy_old = test_rnn_addition_single_batch_old()
    accuracy_new = test_rnn_addition_single_batch_new()

    assert abs(accuracy_new - accuracy_old) < 0.0005, \
        f"Model don't match: old_acc = {accuracy_old}, new_acc = {accuracy_new}"


def cycle_data_generators(generators):
    while True:
        for generator in generators:
            yield next(generator)


def test_rnn_addition_two_batches_old():
    _seed_everything()
    DEVICE = torch.device("cuda:0")

    sequence_length = 20

    generators = []
    for i in range(2):
        data_generator = DummyDataGenerator(
            *dataset_generator(5000, sequence_length), DEVICE
        )
        generators.append(data_generator)
    cycler = cycle_data_generators(generators)

    net = Net().to(DEVICE)

    cost_function = F.mse_loss

    ndes = RNNnDESOptimizerOld(
        model=net,
        criterion=cost_function,
        data_gen=cycler,
        budget=55000,
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

    accuracy = eval_multiple_batches(best, generators)

    return accuracy


def eval_multiple_batches(model, generators):
    total_accuracy = 0
    for generator in generators:
        idx, batch = next(generator)
        x, y = batch
        accuracy = eval(model, x, y, F.mse_loss)
        total_accuracy += accuracy

    return total_accuracy / len(generators)


def test_rnn_addition_two_batches_new():
    _seed_everything()
    DEVICE = torch.device("cuda:0")
    devices = [torch.device("cuda:0")]
    # devices = [torch.device("cuda:0"), torch.device("cuda:1")]

    sequence_length = 20

    batches = []
    generators = []
    for i in range(2):
        data_generator = DummyDataGenerator(
            *dataset_generator(5000, sequence_length), DEVICE
        )
        generators.append(data_generator)
        batches.append(next(data_generator))

    net = Net().to(DEVICE)

    cost_function = F.mse_loss

    ndes = RNNnDESOptimizerNew(
        model=net,
        criterion=cost_function,
        data_gen=None,
        budget=55000,
        history=16,
        nn_train=True,
        lower=-2,
        upper=2,
        tol=1e-6,
        worst_fitness=3,
        device=DEVICE,
        log_dir=f"rnn_addition_{sequence_length}",
        batches=batches,
        devices=devices
    )

    best = ndes.run()

    accuracy = eval_multiple_batches(best, generators)

    return accuracy


def test_rnn_addition_two_batches():
    accuracy_old = test_rnn_addition_two_batches_old()
    accuracy_new = test_rnn_addition_two_batches_new()

    assert abs(accuracy_old - accuracy_new) < 0.005, \
        f"Model don't match: old_acc = {accuracy_old}, new_acc = {accuracy_new}"


def _seed_everything():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test_rnn_addition_three_batches_old():
    _seed_everything()
    DEVICE = torch.device("cuda:0")

    sequence_length = 20

    generators = []
    for i in range(3):
        data_generator = DummyDataGenerator(
            *dataset_generator(5000, sequence_length), DEVICE
        )
        generators.append(data_generator)
    cycler = cycle_data_generators(generators)

    net = Net().to(DEVICE)

    cost_function = F.mse_loss

    ndes = RNNnDESOptimizerOld(
        model=net,
        criterion=cost_function,
        data_gen=cycler,
        budget=55000,
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

    accuracy = eval_multiple_batches(best, generators)

    return accuracy


def test_rnn_addition_three_batches_new():
    _seed_everything()
    DEVICE = torch.device("cuda:0")
    devices = [torch.device("cuda:0")]
    # devices = [torch.device("cuda:0"), torch.device("cuda:1")]

    sequence_length = 20

    batches = []
    generators = []
    for i in range(3):
        data_generator = DummyDataGenerator(
            *dataset_generator(5000, sequence_length), DEVICE
        )
        generators.append(data_generator)
        batches.append(next(data_generator))

    net = Net().to(DEVICE)

    cost_function = F.mse_loss

    ndes = RNNnDESOptimizerNew(
        model=net,
        criterion=cost_function,
        data_gen=None,
        budget=55000,
        history=16,
        nn_train=True,
        lower=-2,
        upper=2,
        tol=1e-6,
        worst_fitness=3,
        device=DEVICE,
        log_dir=f"rnn_addition_{sequence_length}",
        batches=batches,
        devices=devices
    )

    best = ndes.run()

    accuracy = eval_multiple_batches(best, generators)

    return accuracy


def test_rnn_addition_three_batches():
    accuracy_old = test_rnn_addition_three_batches_old()
    accuracy_new = test_rnn_addition_three_batches_new()

    assert abs(accuracy_old - accuracy_new) < 0.005, \
        f"Models don't match: old_acc = {accuracy_old}, new_acc = {accuracy_new}"


if __name__ == "__main__":
    test_rnn_addition_three_batches()
