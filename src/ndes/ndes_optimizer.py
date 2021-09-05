import gc
from math import sqrt
from timeit import default_timer as timer

import cma
import numpy as np
import torch

from ndes.distribution_strategy.complete_loading_strategy import CompleteLoadingStrategy
from ndes.fitness_processing.fitness_ewma_logger import EWMAFitnessProcessing
from ndes.fitness_processing.identity_fitness_processing import IdentityFitnessProcessing
from ndes.forward_backward_pass import ForwardBackwardPass
from ndes.node.gpu_node.gpu_node import GPUNode
from ndes.tensor_model_converter import TensorModelConverter
from ndes.ndes import NDES, SecondaryMutation
from ndes.population_initializers import (
    StartFromUniformPopulationInitializer,
    XavierMVNPopulationInitializer,
)
from ndes.utils import seconds_to_human_readable
from ndes.fitness_processing.fitness_processing import FitnessProcessingType
from ndes.node.gpu_node.gpu_node_config import GPUNode as GPUNodeType


class NDESOptimizer:
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
            nodes=None,
            fitness_processing_type=FitnessProcessingType.IDENTITY,
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
        self.tensor_model_converter = TensorModelConverter(model.parameters())
        self.initial_value = self.tensor_model_converter.zip_layers(model.parameters())
        self.xavier_coeffs = self.calculate_xavier_coefficients(model.parameters())
        self.secondary_mutation = kwargs.get("secondary_mutation", None)
        self.lr = lr

        self.batches = batches
        self.forward_backward_pass = ForwardBackwardPass(self.secondary_mutation,
                                                         self.criterion,
                                                         self.lr,
                                                         self.tensor_model_converter)
        self.nodes = []
        self.initialize_nodes(nodes)
        self.distribution_strategy = CompleteLoadingStrategy(self.nodes, self.data_gen, self.batches)
        self.fitness_processing = None
        self.setup_fitness_processing(fitness_processing_type)

    def initialize_nodes(self, nodes_configs):
        for config in nodes_configs:
            if type(config) == GPUNodeType:
                node = GPUNode(config.device, self.model, self.forward_backward_pass)
                self.nodes.append(node)
            else:
                raise TypeError("Unknown node type")

    def cleanup_nodes(self):
        for node in self.nodes:
            node.cleanup()

    def setup_fitness_processing(self, fitness_processing_type):
        if fitness_processing_type == FitnessProcessingType.EWMA:
            self.fitness_processing = EWMAFitnessProcessing(self.data_gen,
                                                            self.model,
                                                            self.criterion,
                                                            len(self.batches))
        elif fitness_processing_type == FitnessProcessingType.IDENTITY:
            self.fitness_processing = IdentityFitnessProcessing()
        self.kwargs["iter_callback"] = self.fitness_processing.update_after_iteration

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

    def _objective_function_population(self, population):
        fitness_values, batch_indices = self.distribution_strategy.evaluate(population)

        for i, batch_idx in enumerate(batch_indices):
            fitness_values[i] = self.fitness_processing.update_batch(batch_idx, fitness_values[i])

        return fitness_values

    def _infer_for_individual_algorithm(self, weights):
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
            return self.fitness_processing.update_batch(batch_idx, loss)
        return loss

    # @profile
    def run(self, test_func=None):
        """Optimize model's weights wrt. the given criterion.

        Returns:
            Optimized model.
        """
        try:
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
                self._reweight_model(self.model, best_value)
                return self.model, fitnesses_iterations
        finally:
            self.cleanup_nodes()

    # @profile
    def _reweight_model(self, model, individual):
        for param, layer in zip(model.parameters(), self.tensor_model_converter.unzip_layers(individual)):
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


class RNNnDESOptimizer(NDESOptimizer):
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