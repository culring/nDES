import gc
from math import sqrt
from timeit import default_timer as timer

import cma
import numpy as np
import torch

from ndes import NDES, SecondaryMutation
from population_initializers import (
    StartFromUniformPopulationInitializer,
    XavierMVNPopulationInitializer,
)
from utils import seconds_to_human_readable


class FitnessEWMALogger:
    """Logger for the fitness values of data batches"""

    def __init__(self, data_gen, model, criterion):
        self.ewma_alpha = 1
        self.iter_counter = 1
        self.num_batches = len(data_gen)
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
        models=None,
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
        self.devices = kwargs.get("devices")
        self.num_devices = len(self.devices)
        self.batches = kwargs.get("batches")
        self.num_batches = len(self.batches)
        self.device_to_model = models
        self.device_to_batches = kwargs.get("device_to_batches")
        self.batch_idx = 0
        self.population_step = kwargs.get("population_step", 2)
        # self.load_batches()
        if use_fitness_ewma:
            self.ewma_logger = FitnessEWMALogger(data_gen, model, criterion)
            self.kwargs["iter_callback"] = self.ewma_logger.update_after_iteration

    def load_batches(self):
        self.device_to_batches = dict()
        for device in self.devices:
            batches_device = []
            for batch in self.batches:
                batch_device = batch[0].to(device), batch[1].to(device)
                batches_device.append(batch_device)
            self.device_to_batches[str(device)] = batches_device

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

    # def _objective_function_population(self, population):
    #     individual_idx = 0
    #     population_size = population.shape[1]
    #     individuals_per_device = population_size // self.num_devices
    #     surplus = population_size - self.num_devices * individuals_per_device
    #     fitnesses_total = []
    #     population_devices = []
    #     for device in self.devices:
    #         population_devices.append(population.to(device))
    #     for i, device in enumerate(self.devices):
    #         individuals = individuals_per_device if surplus <= 0 else individuals_per_device + 1
    #         population_device = population_devices[i]
    #         fitnesses = self._evaluate_individuals(device, population_device, individual_idx, individual_idx + individuals - 1, self.batch_idx)
    #         fitnesses_total.extend(fitnesses)
    #         individual_idx += individuals
    #         self.batch_idx = (self.batch_idx + individuals) % self.num_batches
    #         surplus -= 1
    #
    #     return fitnesses_total

    def _objective_function_population(self, population):
        population_size = population.shape[1]
        fitnesses_total = []
        population_devices = []
        for device in self.devices:
            population_devices.append(population.to(device))
        for current_individual_idx in range(0, population_size, self.population_step):
            current_device_idx = (current_individual_idx//self.population_step) % self.num_devices
            current_device = self.devices[current_device_idx]
            population_device = population_devices[current_device_idx]
            if population_size - current_individual_idx >= self.population_step:
                number_of_individuals = self.population_step
            else:
                number_of_individuals = population_size - current_individual_idx
            fitnesses = self._evaluate_individuals_on_device(current_device, population_device, current_individual_idx, self.batch_idx, number_of_individuals)
            fitnesses_total.extend(fitnesses)
            self.batch_idx = (self.batch_idx + number_of_individuals) % self.num_batches

        return fitnesses_total

    def _evaluate_individuals_on_device(self, device, population, start_individual_idx, start_batch_idx, count):
        batches_device = self.device_to_batches[str(device)]
        fitnesses = []
        model = self.device_to_model[str(device)]
        for i in range(count):
            individual_idx = start_individual_idx + i
            individual = population[:, individual_idx]
            batch_idx = (start_batch_idx + i) % self.num_batches
            batch = batches_device[batch_idx]
            fitness = self._infer(model, individual, batch)
            fitnesses.append(fitness)

        return fitnesses

    def _infer(self, model, individual, batch):
        """Custom objective function for the DES optimizer."""
        self._reweight_model(model, individual)
        b_x, y = batch
        if self.secondary_mutation == SecondaryMutation.Gradient:
            gradient = []
            with torch.enable_grad():
                model.zero_grad()
                out = model(b_x)
                loss = self.criterion(out, y)
                loss.backward()
                for param in model.parameters():
                    gradient.append(param.grad.flatten())
                gradient = torch.cat(gradient, 0)
                # In-place mutation of the weights
                individual -= self.lr * gradient
        else:
            out = model(b_x)
            loss = self.criterion(out, y)
        loss = loss.item()
        # if self.use_fitness_ewma:
        #     return self.ewma_logger.update_batch(batch_idx, loss)
        return loss

    # @profile
    def _objective_function(self, weights):
        """Custom objective function for the DES optimizer."""
        self._reweight_model(self.model, weights)
        batch_idx, (b_x, y) = next(self.data_gen)
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

    # @profile
    def run(self, test_func=None):
        """Optimize model's weights wrt. the given criterion.

        Returns:
            Optimized model.
        """
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
                    fn=self._objective_function,
                    fn_population=self._objective_function_population,
                    xavier_coeffs=self.xavier_coeffs,
                    population_initializer=population_initializer,
                    test_func=test_func,
                )
            )
            if self.restarts is not None:
                for i in range(self.restarts):
                    self.kwargs["population_initializer"] = self.population_initializer(
                        best_value, *population_initializer_args
                    )
                    ndes = NDES(log_id=i, **self.kwargs)
                    best_value = ndes.run()
                    del ndes
                    if self.test_func is not None:
                        self.test_model(best_value)
                    gc.collect()
                    torch.cuda.empty_cache()
            else:
                ndes = NDES(**self.kwargs)
                best_value = ndes.run()
            self._reweight_model(self.model, best_value)
            return self.model

    # @profile
    def _reweight_model(self, model, individual):
        for param, layer in zip(model.parameters(), self.unzip_layers(individual)):
            param.data.copy_(layer)

    # @profile
    def test_model(self, weights):
        end = timer()
        model = self.model
        self._reweight_model(weights)
        print(f"\nPerf after {seconds_to_human_readable(end - self.start)}")
        return self.test_func(model)

    def iter_callback(self):
        pass

    # @profile
    def find_best(self, population):
        min_loss = torch.finfo(torch.float32).max
        best_idx = None
        for i in range(population.shape[1]):
            self._reweight_model(population[:, i])
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
