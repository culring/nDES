import copy
import gc
from math import sqrt
import time
from timeit import default_timer as timer

import cma
import numpy as np
import torch

from ndes_rewrited import NDES, SecondaryMutation
from population_initializers import (
    StartFromUniformPopulationInitializer,
    XavierMVNPopulationInitializer,
)
from utils import seconds_to_human_readable


def test(size):
    return size*[0]


def wrapper(*args):
    import cProfile
    cProfile.runctx('_evaluate_device(*args)', globals(), locals(), 'process.prof')


def worker(model, queue_query, queue_result, job_id):
    while True:
        print(f"[job_{job_id}] waiting for queue")
        message = queue_query.get()
        if type(message) == str and message == "STOP":
            print(f"[job_{job_id}] exiting from process")
            exit(0)
        individuals, batch_iterator, context = message
        print(f"[job_{job_id}] got from queue")
        fitnesses = _evaluate_device(individuals, model, batch_iterator, context)
        queue_result.put((fitnesses, batch_iterator.batches_idxs_logger))


def _evaluate_device(individuals, model, batch_iterator, context):
    fitnesses = []

    for i in range(individuals.shape[1]):
        individual = individuals[:, i]
        batch = next(batch_iterator)
        fitness = _infer_for_population_algorithm(individual, model, batch, context)
        fitnesses.append(fitness)

    return fitnesses


# @profile
def _infer_for_population_algorithm(weights, model, batch, context):
    """Custom objective function for the DES optimizer."""
    batch_idx, (b_x, y) = batch
    _reweight_model(model, weights, context)
    if context["secondary_mutation"] == SecondaryMutation.Gradient:
        gradient = []
        with torch.enable_grad():
            model.zero_grad()
            out = model(b_x)
            loss = context["criterion"](out, y)
            loss.backward()
            for param in model.parameters():
                gradient.append(param.grad.flatten())
            gradient = torch.cat(gradient, 0)
            # In-place mutation of the weights
            weights -= context["lr"] * gradient
    else:
        out = model(b_x)
        loss = context["criterion"](out, y)
    loss = loss.item()
    return loss


def _reweight_model(model, individual, context):
    for param, layer in zip(model.parameters(), unzip_layers(individual, context)):
        param.data.copy_(layer)


def unzip_layers(zipped_layers, context):
    start = 0
    for offset, shape in context["_layers_offsets_shapes"]:
        yield zipped_layers[start:offset].view(shape)
        start = offset


class FitnessEWMALogger:
    """Logger for the fitness values of data batches"""

    def __init__(self, data_gen, model, criterion, num_batches):
        self.ewma_alpha = 1
        self.iter_counter = 1
        self.num_batches = num_batches
        self.ewma = torch.zeros(self.num_batches)
        # FIXME
        # sum of losses per batch for the current iteration
        self.current_losses = torch.zeros(self.num_batches)
        # count of evaluations per batch for the current iteration
        self.current_counts = torch.zeros(self.num_batches)
        self.set_initial_losses(data_gen, model, criterion)

    def set_initial_losses(self, data_gen, model, criterion):
        # XXX this is really ugly
        for batch_idx, (b_x, y) in data_gen:
            out = model(b_x)
            loss = criterion(out, y).item()
            self.ewma[batch_idx] = loss
            if batch_idx >= self.num_batches - 1:
                break

    def update_batch(self, batch_idx, loss):
        self.current_losses[batch_idx] += loss
        self.current_counts[batch_idx] += 1
        return loss - self.ewma[batch_idx]

    def update_after_iteration(self):
        self.ewma *= 1 - self.ewma_alpha
        # calculate normal average for each batch and include it in the EWMA
        self.ewma += self.ewma_alpha * (self.current_losses / self.current_counts)
        # reset stats for the new iteration
        self.current_losses = torch.zeros(self.num_batches)
        # XXX ones to prevent 0 / 0
        self.current_counts = torch.ones(self.num_batches)
        self.ewma_alpha = 1 / (self.iter_counter ** (1 / 3))
        self.iter_counter += 1


class BasenDESOptimizer:
    """Base interface for the nDES optimizer for the neural networks optimization."""

    def __init__(
        self,
        model,
        criterion,
        data_gen,
        x_val=None,
        y_val=None,
        use_fitness_ewma=False,
        population_initializer=XavierMVNPopulationInitializer,
        restarts=None,
        lr=1e-3,
        batches=None,
        devices=None,
        **kwargs,
    ):
        """
        Args:
            model: ``pytorch``'s model
            criterion: Loss function, must be minimizable.
            data_gen: Data generator, should yield batches: (batch_idx, (x, y))
            x_val: Validation data
            y_val: Validation ground truth
            use_fitness_ewma: is ``True`` will use EWMA fitness loss tracker
            population_initializer: Class of the population initialization strategy
            lr: Learning rate, only used if secondary_mutation is set to gradient
            restarts: Optional number of NDES's restarts.
            **kwargs: Keyword arguments for NDES optimizer
        """
        self._layers_offsets_shapes = []
        self.model = model
        self.criterion = criterion
        self.population_initializer = population_initializer
        self.data_gen = data_gen
        self.x_val = x_val
        self.y_val = y_val
        self.use_fitness_ewma = use_fitness_ewma
        self.kwargs = kwargs
        self.restarts = restarts
        self.start = timer()
        if restarts is not None and self.kwargs.get("budget") is not None:
            self.kwargs["budget"] //= restarts
        self.initial_value = self.zip_layers(model.parameters())
        self.xavier_coeffs = self.calculate_xavier_coefficients(model.parameters())
        self.secondary_mutation = kwargs.get("secondary_mutation", None)
        self.lr = lr

        self.devices = devices
        self.device_to_batches = {}
        self.init_batches(batches)
        self.batches = batches
        self.batch_idx = 0
        self.num_batches = len(batches)
        self.device_to_model = {}
        self.init_models(model)

        if use_fitness_ewma:
            self.ewma_logger = FitnessEWMALogger(self.data_gen_for_ewma(), model, criterion, self.num_batches)
            self.kwargs["iter_callback"] = self.ewma_logger.update_after_iteration

        self.jobs = []
        self.queues = []

    def cleanup_jobs(self):
        print("cleanup method invoked")
        for queue_query, queue_result in self.queues:
            queue_query.put("STOP")
            queue_query.close()
            queue_result.close()
        time.sleep(1)
        for job in self.jobs:
            job.join()

        self.jobs = []
        self.queues = []

    def data_gen_for_ewma(self):
        yield self.get_next_batch()

    def init_batches(self, batches):
        for device in self.devices:
            batches_device = self.move_batches_to_device(batches, device)
            self.device_to_batches[str(device)] = batches_device

    def move_batches_to_device(self, batches, device):
        batches_device = []
        for batch in batches:
            _, (b_x, y) = batch
            batches_device.append((b_x.to(device), y.to(device)))

        return batches_device

    def init_models(self, model):
        for device in self.devices:
            self.device_to_model[str(device)] = copy.deepcopy(model).to(device)

    def init_jobs(self):
        for i, device in enumerate(self.devices):
            model = self.device_to_model[str(device)]
            queue_query = torch.multiprocessing.Queue()
            queue_result = torch.multiprocessing.Queue()
            self.queues.append((queue_query, queue_result))
            job = torch.multiprocessing.Process(target=worker,
                                                args=(model, queue_query, queue_result, i))
            job.start()
            self.jobs.append(job)
        time.sleep(3)

    def zip_layers(self, layers_iter):
        """Concatenate flattened layers into a single 1-D tensor.
        This method also saves shapes of layers and their offsets in the final
        tensor, allowing for a fast unzip operation.

        Args:
            layers_iter: Iterator over model's layers.
        """
        self._layers_offsets_shapes = []
        tensors = []
        current_offset = 0
        for param in layers_iter:
            shape = param.shape
            tmp = param.flatten()
            current_offset += len(tmp)
            self._layers_offsets_shapes.append((current_offset, shape))
            tensors.append(tmp)
        return torch.cat(tensors, 0).contiguous()

    @staticmethod
    def calculate_xavier_coefficients(layers_iter):
        xavier_coeffs = []
        for param in layers_iter:
            param_num_elements = param.numel()
            if len(param.shape) > 1:
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(param)
                xavier_coeffs.extend(
                    [sqrt(6 / (fan_in + fan_out))] * param_num_elements
                )
            else:
                xavier_coeffs.extend([xavier_coeffs[-1]] * param_num_elements)
        return torch.tensor(xavier_coeffs)

    def unzip_layers(self, zipped_layers):
        """Iterator over 'unzipped' layers, with their proper shapes.

        Args:
            zipped_layers: Flattened representation of layers.
        """
        start = 0
        for offset, shape in self._layers_offsets_shapes:
            yield zipped_layers[start:offset].view(shape)
            start = offset

    def _objective_function_population(self, population):
        fitnesses = []

        device_to_population = self._transfer_population_to_devices(population)
        num_devices = len(self.devices)
        num_individuals_per_device = population.shape[1] // num_devices
        remaining_individuals = population.shape[1] - num_individuals_per_device * num_devices
        current_individual_idx = 0

        for i in range(num_devices):
            num_individuals = num_individuals_per_device
            if remaining_individuals > 0:
                remaining_individuals -= 1
                num_individuals += 1

            current_device = self.devices[i % num_devices]
            population = device_to_population[str(current_device)]

            end_individual_idx = current_individual_idx + num_individuals
            individuals = population[:, current_individual_idx:end_individual_idx]

            batch_loader = self.BatchLoader(self.batch_idx, current_device, 16, self.batches)
            batch_iterator = iter(batch_loader)

            context = {
                "_layers_offsets_shapes": self._layers_offsets_shapes,
                "secondary_mutation": self.secondary_mutation,
                "criterion": self.criterion,
                "lr": self.lr
            }

            queue_query, _ = self.queues[i]
            queue_query.put((individuals, batch_iterator, context))

            current_individual_idx += num_individuals
            self.batch_idx = (self.batch_idx + num_individuals) % len(self.batches)

        print("waiting for queues")

        for i, (_, queue_result) in enumerate(self.queues):
            result, batches_idxs = queue_result.get()
            if self.use_fitness_ewma:
                result = self.get_fitness_values_with_ewma(result, batches_idxs)
            fitnesses.extend(result)

        return fitnesses

    def get_fitness_values_with_ewma(self, fitnesses, batches_idxs):
        fitnesses_updated = []
        for fitness, batch_idx in zip(fitnesses, batches_idxs):
            fitness_updated = self.ewma_logger.update_batch(batch_idx, fitness)
            fitnesses_updated.append(fitness_updated)

        return fitnesses_updated

    class BatchLoader:
        def __init__(self, start_batch_idx, device, num_batches_on_device, batches):
            self.start_batch_idx = start_batch_idx
            self.device = device
            self.num_batches_on_device = num_batches_on_device
            self.batches = batches
            self.batches_idxs_logger = []

        def __iter__(self):
            self.batch_device_idx = -1
            self.batch_idx = self.start_batch_idx
            self.batches_device = None
            return self

        def __next__(self):
            self.batch_device_idx = (self.batch_device_idx + 1) % self.num_batches_on_device
            if self.batch_device_idx == 0:
                del self.batches_device
                batches_to_move = self._get_n_next_elements_from_array_with_loop(
                    self.batches, self.batch_idx, self.num_batches_on_device)
                self.batches_device = self._move_batches_to_device(batches_to_move, self.device)
                self.batch_idx = (self.batch_idx + self.num_batches_on_device) % len(self.batches)
            batch_idx = (self.batch_idx - self.num_batches_on_device + self.batch_device_idx) % len(self.batches)
            self.batches_idxs_logger.append(batch_idx)
            return batch_idx, self.batches_device[self.batch_device_idx]

        @staticmethod
        def _move_batches_to_device(batches, device):
            batches_device = []
            for batch in batches:
                _, (b_x, y) = batch
                batches_device.append((b_x.to(device), y.to(device)))

            return batches_device

        @staticmethod
        def _get_n_next_elements_from_array_with_loop(array, start_idx, n):
            if start_idx + n <= len(array):
                return array[start_idx:start_idx + n]
            size_to_end = len(array) - start_idx
            size_from_beginning = n - size_to_end
            return array[start_idx:] + array[:size_from_beginning]

    def _transfer_population_to_devices(self, population):
        device_to_population = {}
        for device in self.devices:
            device_to_population[str(device)] = population.to(device)

        return device_to_population

    def _infer_for_individual_algorithm(self, weights):
        """Custom objective function for the DES optimizer."""
        self._reweight_model(self.model, weights)
        batch_idx, (b_x, y) = self.get_next_batch()
        if self.secondary_mutation == SecondaryMutation.Gradient:
            gradient = []
            with torch.enable_grad():
                self.model.zero_grad()
                out = self.model(b_x)
                loss = self.criterion(out, y)
                loss.backward()
                for param in self.model.parameters():
                    gradient.append(param.grad.flatten())
                gradient = torch.cat(gradient, 0)
                # In-place mutation of the weights
                weights -= self.lr * gradient
        else:
            out = self.model(b_x)
            loss = self.criterion(out, y)
        loss = loss.item()
        if self.use_fitness_ewma:
            return self.ewma_logger.update_batch(batch_idx, loss)
        return loss

    def get_next_batch(self, device=torch.device("cuda:0")):
        idx = self.batch_idx
        self.batch_idx = (self.batch_idx + 1) % self.num_batches
        current_device_to_batches = self.device_to_batches[str(device)]
        return idx, current_device_to_batches[idx]

    # @profile
    def run(self, test_func=None):
        """Optimize model's weights wrt. the given criterion.

        Returns:
            Optimized model.
        """
        self.init_jobs()
        self.test_func = test_func
        best_value = self.initial_value
        with torch.no_grad():
            requires_grad = self.secondary_mutation == SecondaryMutation.Gradient
            for param in self.model.parameters():
                param.requires_grad = requires_grad
            population_initializer_args = [
                self.xavier_coeffs,
                self.kwargs["device"],
                self.kwargs.get("lambda_", None),
            ]
            population_initializer = self.population_initializer(
                best_value, *population_initializer_args
            )
            if self.x_val is not None:
                test_func = self.validate_and_test
            else:
                test_func = None
            self.kwargs.update(
                dict(
                    initial_value=best_value,
                    fn=self._infer_for_individual_algorithm,
                    fn_population=self._objective_function_population,
                    xavier_coeffs=self.xavier_coeffs,
                    population_initializer=population_initializer,
                    test_func=test_func,
                )
            )
            fitnesses_iterations = []
            if self.restarts is not None:
                for i in range(self.restarts):
                    self.kwargs["population_initializer"] = self.population_initializer(
                        best_value, *population_initializer_args
                    )
                    ndes = NDES(log_id=i, **self.kwargs)
                    best_value, fitnesses = ndes.run()
                    fitnesses_iterations.append(fitnesses)
                    del ndes
                    if self.test_func is not None:
                        self.test_model(best_value)
                    gc.collect()
                    torch.cuda.empty_cache()
            else:
                ndes = NDES(**self.kwargs)
                best_value, fitnesses = ndes.run()
                fitnesses_iterations.append(fitnesses)
            base_cuda_device = self.device_to_model[str(torch.device("cuda:0"))]
            self._reweight_model(base_cuda_device, best_value)
            self.cleanup_jobs()
            return self.model, fitnesses_iterations

    # @profile
    def _reweight_model(self, model, individual):
        for param, layer in zip(model.parameters(), self.unzip_layers(individual)):
            param.data.copy_(layer)

    # @profile
    def test_model(self, weights):
        end = timer()
        model = self.model
        self._reweight_model(model, weights)
        print(f"\nPerf after {seconds_to_human_readable(end - self.start)}")
        return self.test_func(model)

    def iter_callback(self):
        pass

    # @profile
    def find_best(self, population):
        min_loss = torch.finfo(torch.float32).max
        best_idx = None
        for i in range(population.shape[1]):
            self._reweight_model(self.model, population[:, i])
            out = self.model(self.x_val)
            loss = self.criterion(out, self.y_val).item()
            if loss < min_loss:
                min_loss = loss
                best_idx = i
        return population[:, best_idx].clone()

    def validate_and_test(self, population):
        best_individual = self.find_best(population)
        return self.test_model(best_individual), best_individual


class RNNnDESOptimizer(BasenDESOptimizer):
    """nDES optimizer for RNNs, uses different initialization strategy than base
    optimizer."""

    def __init__(
        self,
        *args,
        population_initializer=StartFromUniformPopulationInitializer,
        secondary_mutation=SecondaryMutation.RandomNoise,
        **kwargs,
    ):
        super().__init__(
            *args,
            population_initializer=population_initializer,
            secondary_mutation=secondary_mutation,
            **kwargs,
        )

    def calculate_xavier_coefficients(self, layers_iter):
        return torch.ones_like(self.initial_value) * 0.4


class CMAESOptimizerRNN:
    def __init__(self, model, criterion, data_gen, restarts=None, **kwargs):
        self.model = model
        self.criterion = criterion
        self.data_gen = data_gen

    def _objective_function(self, weights):
        """Custom objective function for the DES optimizer."""
        self.model.set_weights(weights)
        _, (x_data, y_data) = next(self.data_gen)
        predicted = np.array([self.model.forward(x) for x in x_data])
        loss = self.criterion(y_data, predicted)
        return loss

    def run(self, test_func=None):
        """Optimize model's weights wrt. the given criterion.

        Returns:
            Optimized model.
        """
        es = cma.CMAEvolutionStrategy(11 * [0], 1.5, {"verb_disp": 1, "maxiter": 1000})
        es.optimize(self._objective_function)
        return es.best.get()[0]
