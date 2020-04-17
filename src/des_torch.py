import gc
from math import floor, log, sqrt
from random import sample

import numpy as np
import torch
from torch.distributions import MultivariateNormal, Uniform

from timeit import default_timer as timer
from utils import seconds_to_human_readable


class DESOptimizer(object):

    """Interface for the DES optimizer for the neural networks optimization."""

    def __init__(self, model, criterion, X, Y, restarts=None, **kwargs):
        """TODO: to be defined1.

        Args:
            model: ``pytorch``'s model
            criterion: Loss function, must be minimizable.
            X: Training data.
            Y: Training ground truth for the data.
            restarts: Optional number of DES's restarts.
            **kwargs: Keyword arguments for DES optimizer
        """
        self._layers_offsets_shapes = []
        self.best_value = None
        self.model = model
        self.criterion = criterion
        self.X = X
        self.Y = Y
        self.kwargs = kwargs
        self.restarts = restarts
        self.start = timer()
        if restarts is not None and self.kwargs.get('budget') is not None:
            self.kwargs['budget'] //= restarts
        self.zip_layers(model.parameters())

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
        xavier_coeffs = []
        for param in layers_iter:
            shape = param.shape
            tmp = param.flatten()
            current_offset += len(tmp)
            self._layers_offsets_shapes.append((current_offset, shape))
            tensors.append(tmp)
            if len(shape) > 1:
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(param)
                xavier_coeffs.extend([sqrt(6 / (fan_in + fan_out))]*len(tmp))
            else:
                xavier_coeffs.extend([xavier_coeffs[-1]]*len(tmp))
        self.best_value = torch.cat(tensors, 0)
        self.xavier_coeffs = torch.tensor(xavier_coeffs)

    def unzip_layers(self, zipped_layers):
        """Iterator over 'unzipped' layers, with their proper shapes.

        Args:
            zipped_layers: Flattened representation of layers.
        """
        start = 0
        for offset, shape in self._layers_offsets_shapes:
            yield zipped_layers[start:offset].view(shape)
            start = offset

    def _objective_function(self, weights):
        """Custom objective function for the DES optimizer."""
        #  X = Variable(torch.Tensor(self.X).float())
        #  Y = Variable(torch.Tensor(self.Y).long())
        for param, layer in zip(self.model.parameters(),
                                self.unzip_layers(weights)):
            param.data = layer
        out = self.model(self.X)
        return self.criterion(out, self.Y).item()

    def run(self, test_func=None):
        """Optimize model's weights wrt. the given criterion.

        Returns:
            Optimized model.
        """
        with torch.no_grad():
            for param in self.model.parameters():
                param.requires_grad = False
            if self.restarts is not None:
                for i in range(self.restarts):
                    des = DES(
                        self.best_value, self._objective_function,
                        xavier_coeffs=self.xavier_coeffs, log_id=i,
                        **self.kwargs)
                    self.best_value = des.run()
                    del des
                    if test_func is not None:
                        end = timer()
                        model = self.model
                        for param, layer in zip(model.parameters(),
                                                self.unzip_layers(self.best_value)):
                            param.data = layer
                        print(f"\nPerf after {seconds_to_human_readable(end - self.start)}")
                        test_func(model)
                    gc.collect()
                    torch.cuda.empty_cache()
            else:
                des = DES(
                    self.best_value, self._objective_function,
                    xavier_coeffs=self.xavier_coeffs, **self.kwargs)
                self.best_value = des.run()
            for param, layer in zip(self.model.parameters(),
                                    self.unzip_layers(self.best_value)):
                param.data = layer
            return self.model


class DES(object):

    """Docstring for DES. """

    def __init__(self, initial_value, fn, lower, upper, **kwargs):
        self.initial_value = initial_value
        self.problem_size = int(len(initial_value))
        self.fn = fn
        self.lower = lower
        self.upper = upper

        self.device = kwargs.get("device", torch.device('cpu'))
        self.dtype = kwargs.get("dtype", torch.float32)
        self.xavier_coeffs = kwargs.get('xavier_coeffs', None)
        if self.xavier_coeffs is not None:
            self.xavier_coeffs = self.xavier_coeffs.to(self.device)

        if np.isscalar(lower):
            self.lower = torch.tensor(
                [lower]*self.problem_size, device=self.device,
                dtype=self.dtype)

        if np.isscalar(upper):
            self.upper = torch.tensor(
                [upper]*self.problem_size, device=self.device,
                dtype=self.dtype)

        # Neural networks training mode
        self.nn_train = kwargs.get("nn_train", False)

        # Scaling factor of difference vectors (a variable!)
        self.Ft = kwargs.get("Ft", 1)
        self.initFt = kwargs.get("initFt", 1)

        #  Fitness value after which the convergence is reached
        self.stopfitness = kwargs.get("stopfitness", -np.inf)

        # Strategy parameter setting:
        #  The maximum number of fitness function calls
        self.budget = kwargs.get("budget", 10000 * self.problem_size)
        #  Population starting size
        self.init_lambda = kwargs.get("lambda_", 4 * self.problem_size)
        #  print(f'Problem size: {self.problem_size} lambda: {self.init_lambda}')
        #  Population ending size
        self.minlambda = kwargs.get("minlambda", 4 * self.problem_size)
        #  Population size
        self.lambda_ = self.init_lambda
        #  Selection size
        self.mu = kwargs.get("mu", floor(self.lambda_ / 2))
        #  Weights to calculate mean from selected individuals
        self.weights = log(self.mu + 1) - torch.arange(1., self.mu + 1,
                                                       device=self.device,
                                                    dtype=self.dtype).log()
        #     \-> weights are normalized by the sum
        self.weights = self.weights / sum(self.weights)
        self.weights_pop = log(self.lambda_ + 1) - torch.arange(
            1., self.lambda_ + 1, device=self.device, dtype=self.dtype).log()
        self.weights_pop = self.weights_pop / sum(self.weights_pop)
        #  Variance effectiveness factor
        self.mueff = kwargs.get("mueff", sum(self.weights) ** 2 /
                                sum(self.weights ** 2))
        #  Evolution Path decay factor
        self.cc = kwargs.get("ccum", self.mu / (self.mu + 2))
        #  Size of evolution path
        self.path_length = kwargs.get("path_length",  6)
        #  Evolution Path decay factor
        self.cp = kwargs.get("cp", 1 / sqrt(self.problem_size))
        #  Maximum number of iterations after which algorithm stops
        self.max_iter = kwargs.get("maxit", floor(
            self.budget / (self.lambda_ + 1)))
        self.c_Ft = kwargs.get("c_Ft",  0)
        #  Path Length Control reference value
        self.path_ratio = kwargs.get("path_ratio", sqrt(self.path_length))
        #  Size of the window of history - the step length history
        self.hist_size = kwargs.get("history", 5)
        self.Ft_scale = kwargs.get(
            "Ft_scale",
            ((self.mueff + 2) / (self.problem_size + self.mueff + 3)) /
            (1 + 2 * max(0, sqrt((self.mueff - 1) / (self.problem_size + 1)) - 1) +
             (self.mueff + 2) / (self.problem_size + self.mueff + 3)))
        self.tol = kwargs.get("tol", 1e-12)
        #  Number of function evaluations
        self.count_eval = 0
        self.sqrt_N = sqrt(self.problem_size)

        self.log_all = kwargs.get("log_all", False)
        self.log_Ft = kwargs.get("log_Ft", self.log_all)
        self.log_value = kwargs.get("log_value", self.log_all)
        #  self.log_mean = kwargs.get("log_mean", self.log_all)
        self.log_pop = kwargs.get("log_pop", self.log_all)
        self.log_best_val = kwargs.get("log_best_val", self.log_all)
        self.log_worst_val = kwargs.get("log_worst_val", self.log_all)
        self.log_id = kwargs.get("log_id", 0)
        #  self.log_eigen = kwargs.get("log_eigen", self.log_all)

        # nonLamarckian approach allows individuals to violate boundaries.
        # Fitness value is estimeted by fitness of repaired individual.
        self.lamarckism = kwargs.get("lamarckism", False)
        self.worst_fitness = torch.finfo(self.dtype).max

        self.cpu = torch.device('cpu')

    def bounce_back_boundary_1d(self, x, lower, upper):
        """TODO

        Examples:
        >>> a = torch.tensor([-2429.4529, 10, 580.3583, -10, 1316.1814, 0, 0])
        >>> lower = torch.ones(7) * -2000
        >>> upper = torch.ones(7) * 500
        >>> bounce_back_boundary_1d(a, lower, upper)
        tensor([-1570.5471,    10.0000,   419.6417,   -10.0000,  -316.1814,
        0.0000, 0.0000])
        """
        is_lower_boundary_ok = (x >= lower)
        is_upper_boundary_ok = (x <= upper)
        if is_lower_boundary_ok.all() and is_upper_boundary_ok.all():
            return x
        delta = upper - lower
        x = torch.where(is_lower_boundary_ok, x, lower + ((lower - x) % delta))
        x = torch.where(is_upper_boundary_ok, x,
                        upper - (((upper - x) * -1) % delta))
        #  x = self.delete_infs_nans(x)
        self.delete_infs_nans(x)
        return x

    def bounce_back_boundary_2d(self, x, lower, upper):
        """TODO

        Examples:
        >>> a = torch.tensor([[-1386.0077,  3332.5007,  1055.5032,  -565.2601],
        [ 1038.6169, -1425.2521,  -847.1431,   628.4320],
        [ -108.6701,  1107.3376,  1553.3640,  -536.9139],
        [ -187.0682,  1417.4506,    77.6953, -1234.0394],
        [  401.2337,   141.4529,   563.3681,  1164.2892],
        [ 1775.2698,  1104.7098,  -551.6083,  1209.7771],
        [  961.3401,   186.5654,  -450.5545,   155.7374]])
        >>> lower = torch.ones(7) * -300
        >>> upper = torch.ones(7) * 500
        >>> bounce_back_boundary_2d(a, lower, upper)
        tensor([[ -13.9923,   67.4993,  -55.5032,  -34.7399],
                [ -38.6169,   25.2521,  247.1431,  371.5680],
                [-108.6701, -107.3376,  246.6360,  -63.0861],
                [-187.0682,  382.5494,   77.6953, -165.9606],
                [ 401.2337,  141.4529,  436.6319, -164.2892],
                [  24.7302, -104.7098,  -48.3917, -209.7771],
                [  38.6599,  186.5654, -149.4455,  155.7374]])
        """
        transposed = x.transpose(0, 1)
        is_lower_boundary_ok = (transposed >= lower)
        is_upper_boundary_ok = (transposed <= upper)
        if is_lower_boundary_ok.all() and is_upper_boundary_ok.all():
            return x
        delta = upper - lower
        transposed = torch.where(is_lower_boundary_ok, transposed,
                                 lower + ((lower - transposed) % delta))
        x = torch.where(is_upper_boundary_ok, transposed,
                        upper - (((upper - transposed) * -1) % delta)
                        ).transpose(0, 1)
        return x

    def delete_infs_nans(self, x):
        assert torch.isfinite(x).all()
        #  x[~infs] = self.worst_fitness
        #  return x

    def _fitness_wrapper(self, x):
        if (x >= self.lower).all() and (x <= self.upper).all():
            self.count_eval += 1
            return self.fn(x)
        else:
            return self.worst_fitness

    def sample_from_history(self, history, history_sample):
        ret = []
        for i in range(self.lambda_):
            n_cols = history[history_sample[i]].shape[1]
            ret.extend(sample(range(n_cols), 1))
        return torch.tensor(ret, device=self.cpu)

    def _fitness_lamarckian(self, x):
        if not np.isscalar(x):
            cols = 1 if len(x.shape) == 1 else x.shape[1]
            if self.count_eval + cols <= self.budget:
                ret = []
                if cols > 1:
                    for i in range(cols):
                        ret.append(self._fitness_wrapper(x[:, i]))
                else:
                    return self._fitness_wrapper(x)
                return torch.tensor(ret, device=self.device, dtype=self.dtype)
            else:
                ret = []
                budget_left = self.budget - self.count_eval
                for i in range(budget_left):
                    ret.append(self._fitness_wrapper(x[:, i]))
                return torch.tensor(
                    ret + [self.worst_fitness]*(cols - budget_left),
                    device=self.device, dtype=self.dtype)
        else:
            if self.count_eval < self.budget:
                return self._fitness_wrapper(x)
            else:
                return self.worst_fitness

    def _fitness_non_lamarckian(self, x, x_repaired, fitness):
        #  x = self.delete_infs_nans(x)
        self.delete_infs_nans(x)
        #  x_repaired = self.delete_infs_nans(x_repaired)
        self.delete_infs_nans(x_repaired)
        p_fit = fitness

        repaired_ind = (x != x_repaired).all(dim=0)
        vec_dist = sum((x - x_repaired) ** 2)

        p_fit[repaired_ind] = self.worst_fit + vec_dist[repaired_ind]
        #  p_fit = self.delete_infs_nans(p_fit)
        self.delete_infs_nans(p_fit)
        return p_fit

    def run(self):
        assert len(self.upper) == self.problem_size
        assert len(self.lower) == self.problem_size
        assert (self.lower < self.upper).all()

        # The best fitness found so far
        self.best_fit = self.worst_fitness
        # The best solution found so far
        self.best_par = None
        # The worst solution found so far
        self.worst_fit = None

        d_mean = torch.zeros((self.problem_size, self.hist_size),
                             device=self.device, dtype=self.dtype)
        ft_history = torch.zeros(self.hist_size, device=self.device, dtype=self.dtype)
        pc = torch.zeros((self.problem_size, self.hist_size),
                         device=self.device, dtype=self.dtype)

        uniform = Uniform(self.lower * 0.9, self.upper * 0.9)

        # XXX this thing doesn't work if we modify lambda on the fly
        #  mean = self.initial_value.unsqueeze(1).repeat(1, self.lambda_).cpu()
        #  sd = torch.tensor([sigma]).repeat(self.lambda_).cpu()
        #  normal_covariance_matrix = (torch.eye(self.lambda_, device=self.device,
                                              #  dtype=self.dtype) *
        #  (self.upper[0] / 4.5)).cpu()
        #  normal = MultivariateNormal(mean, normal_covariance_matrix)
        #  normal = Normal(self.initial_value.cpu(), torch.tensor([sigma]).cpu())

        #  mean = self.initial_value.unsqueeze(1).repeat(1, self.lambda_).cpu()
        #  normal_covariance_matrix = (torch.eye(self.lambda_, device=self.device) * sigma).cpu()
        #  normal = MultivariateNormal(mean, normal_covariance_matrix)
        #
        mean = torch.zeros_like(self.initial_value).unsqueeze(1).repeat(1, self.lambda_).cpu()
        sd = torch.eye(self.lambda_, device=self.device).cpu()
        normal = MultivariateNormal(mean, sd)

        #  start_from_uniform = (self.initial_value == 0).all()
        start_from_uniform = False
        if start_from_uniform:
            nn_uni = Uniform(-self.xavier_coeffs.cpu(), self.xavier_coeffs.cpu())



        best_val_log = []
        evaluation_times = []
        while self.count_eval < self.budget:  # and self.iter_ < self.max_iter:

            hist_head = -1
            self.iter_ = -1

            history = [None]*self.hist_size
            self.Ft = self.initFt
            population = None
            population_repaired = None

            gc.collect()
            torch.cuda.empty_cache()
            print(f"Before population: {torch.cuda.memory_allocated(self.device) / (1024**3)}")
            #  population = torch.empty((self.problem_size, self.lambda_),
                                     #  device=self.device, dtype=self.dtype)
            #  print(f"After population: {torch.cuda.memory_allocated(self.device) / (1024**3)}")
            #  population_repaired = torch.empty((self.problem_size,
                                               #  self.lambda_),
                                              #  device=self.device, dtype=self.dtype)
            #  print(f"After population repaired: {torch.cuda.memory_allocated(self.device) / (1024**3)}")
            cum_mean = (self.upper + self.lower) / 2

            if self.nn_train:
                #  population = normal.sample().to(self.device)
                #  population = mvn.sample().to(self.device)
                #  population = normal.sample((self.lambda_,)).t().to(self.device)

                if start_from_uniform:
                    population = nn_uni.sample((self.lambda_,)).transpose(0, 1).to(self.device)
                    start_from_uniform = False
                    del nn_uni
                else:
                    population = normal.sample().to(self.device)
                    population = population * self.xavier_coeffs[:, None] + self.initial_value[:, None]
                population[:, 0] = self.initial_value
            else:
                population = uniform.sample((self.lambda_,)).transpose(0, 1).to(self.device)
            population_repaired = self.bounce_back_boundary_2d(
                population, self.lower, self.upper)
            if self.lamarckism:
                population = population_repaired

            #  selection = torch.zeros(self.mu, device=self.device, dtype=self.dtype)
            #  selected_points = torch.zeros((self.problem_size, self.mu),
                                          #  device=self.device)

            start = timer()
            fitness = self._fitness_lamarckian(population)
            end = timer()
            evaluation_times.append(end - start)


            new_mean = self.initial_value
            limit = 0
            self.worst_fit = max(fitness)

            # Store population and selection means
            sorting_idx = fitness.argsort()
            pop_mean = population[:, sorting_idx].matmul(self.weights_pop)
            mu_mean = new_mean

            # Matrices for creating diffs
            diffs = torch.zeros((self.problem_size, self.lambda_),
                                device=self.device, dtype=self.dtype)
            x1_sample = self.lambda_
            x2_sample = self.lambda_

            chi_N = sqrt(self.problem_size)
            hist_norm = 1 / sqrt(2)

            stoptol = False

            while self.count_eval < self.budget and not stoptol:  # and self.iter_ < self.max_iter:

                torch.cuda.empty_cache()
                gc.collect()
                print(f"In loop: {torch.cuda.memory_allocated(self.device) / (1024**3)}")
                print(f"{self.count_eval} / {self.budget} ({self.count_eval * 100 / self.budget})")
                self.iter_ += 1
                hist_head = (hist_head + 1) % self.hist_size

                #  if self.log_Ft:
                    #  Ft_log.append(self.Ft)
                #  if self.log_value:
                    #  value_log = np.vstack((value_log, fitness))
                #  if self.log_mean:
                    #  mean_log.append(self._fitness_lamarckian(
                        #  self.bounce_back_boundary(new_mean)))
                #  if self.log_pop:
                    #  pop_log[:, :, self.iter_] = population
                if self.log_best_val:
                    best_val_log.append(self.best_fit)
                    #  np.save(f'log_best_val_{self.log_id}.npy', np.array(best_val_log))
                #  if self.log_worst_val:
                    #  if len(worst_val_log) > 0:
                        #  worst_val_log.append(max(max(worst_val_log), max(fitness)))
                    #  else:
                        #  worst_val_log.append(max(fitness))
                #  if self.log_eigen:
                    #  eigen_log = np.zeros((0, self.problem_size))

                # Select best 'mu' individuals of population
                selection = torch.argsort(fitness)[:self.mu]

                # Save selected population in the history buffer
                history[hist_head] = (population[:, selection] * hist_norm / self.Ft).cpu()

                # Calculate weighted mean of selected points
                old_mean = new_mean
                new_mean = population[:, selection].matmul(self.weights)

                # Write to buffers
                mu_mean = new_mean
                d_mean[:, hist_head] = (mu_mean - pop_mean) / self.Ft

                step = (new_mean - old_mean) / self.Ft
                #  steps.append(step)

                # Update Ft
                ft_history[hist_head] = self.Ft

                # Update parameters
                if hist_head == 0:
                    pc[:, hist_head] = \
                        sqrt(self.mu * self.cp * (2 - self.cp)) * step
                else:
                    pc[:, hist_head] = (1 - self.cp) * pc[:, hist_head - 1] + \
                        sqrt(self.mu * self.cp * (2 - self.cp)) * step

                #  print(f"|step|={sum(step**2)}")
                # Sample from history with uniform distribution
                limit = hist_head + 1 if self.iter_ <= self.hist_size else self.hist_size
                history_sample1 = torch.randint(0, limit, (self.lambda_,),
                                                device=self.cpu)
                history_sample2 = torch.randint(0, limit, (self.lambda_,),
                                                device=self.cpu)

                # XXX this samples from the same history sample
                x1_sample = self.sample_from_history(history, history_sample1)
                x2_sample = self.sample_from_history(history, history_sample1)

                # Make diffs
                for i in range(self.lambda_):
                    x1 = history[history_sample1[i]][:, x1_sample[i]]\
                        .to(self.device)
                    x2 = history[history_sample1[i]][:, x2_sample[i]]\
                        .to(self.device)

                    diffs[:, i] = \
                        sqrt(self.cc) * ((x1 - x2) + torch.randn(1, device=self.device,
                                                                 dtype=self.dtype) * d_mean[:, history_sample1[i]]) + \
                        sqrt(1 - self.cc) * torch.randn(1, device=self.device,
                                                        dtype=self.dtype) * pc[:, history_sample2[i]]

                # New population
                population = (
                    new_mean.unsqueeze(1) + self.Ft * diffs +
                    self.tol *
                    (1 - 2 / (self.problem_size**2)) ** (self.iter_ / 2) *
                    torch.randn(diffs.shape, device=self.device, dtype=self.dtype) / chi_N)
                #  population = self.delete_infs_nans(population)
                self.delete_infs_nans(population)

                # Check constraints violations
                # Repair the individual if necessary
                population_repaired = self.bounce_back_boundary_2d(
                    population, self.lower, self.upper)

                if self.lamarckism:
                    population = population_repaired

                # TODO maybe reuse sorting_idx
                sorting_idx = fitness.argsort()
                pop_mean = population[:, sorting_idx].matmul(self.weights_pop)

                gc.collect()
                torch.cuda.empty_cache()
                print(f"Before fitness: {torch.cuda.memory_allocated(self.device) / (1024**3)}")
                # Evaluation
                start = timer()
                fitness = self._fitness_lamarckian(population)
                end = timer()
                evaluation_times.append(end - start)
                if not self.lamarckism:
                    fitness_non_lamarckian = self._fitness_non_lamarckian(
                            population, population_repaired, fitness)

                # Break if fit
                # XXX doesn't break
                wb = fitness.argmin()
                print(f"best fitness: {fitness[wb]}")
                if fitness[wb] < self.best_fit:
                    self.best_fit = fitness[wb]
                    if not self.lamarckism:
                        self.best_par = population_repaired[:, wb]
                    else:
                        self.best_par = population[:, wb]

                # Check worst fit
                ww = fitness.argmax()
                if fitness[ww] > self.worst_fit:
                    self.worst_fit = fitness[ww]

                # Fitness with penalty for non-lamarckian approach
                if not self.lamarckism:
                    fitness = fitness_non_lamarckian

                # Check if the middle point is the best found so far
                cum_mean = 0.8 * cum_mean + 0.2 * new_mean
                cum_mean_repaired = self.bounce_back_boundary_1d(
                    cum_mean, self.lower, self.upper)

                fn_cum = self._fitness_lamarckian(cum_mean_repaired)
                #  print(f"fn_cum: {fn_cum}")
                if fn_cum < self.best_fit:
                    self.best_fit = fn_cum
                    self.best_par = cum_mean_repaired

                if fitness[0] <= self.stopfitness:
                    #  print("Stop fitness reached.")
                    break

                if abs(fitness.max() - fitness.min()) < self.tol and \
                        self.count_eval < 0.8 * self.budget:
                    stoptol = True
                print(f"iter={self.iter_} ,best={self.best_fit}")
        log_ = {}
        #  if self.log_Ft:
            #  log_["Ft"] = Ft_log
        #  if self.log_value:
            #  log_["value"] = value_log
        #  #  if self.log_mean:
            #  #  log_["mean"] = mean_log
        #  if self.log_pop:
            #  log_["pop"] = pop_log
        if self.log_best_val:
            log_["best_val"] = best_val_log
        #  if self.log_worst_val:
            #  log_["worst_val"] = worst_val_log

        #  np.save(f"times_{self.problem_size}.npy", np.array(evaluation_times))
        return self.best_par #, log_
