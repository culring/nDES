from ndes.distribution_strategy.distribution_strategy import DistributionStrategy


class IterativeLoadingStrategy(DistributionStrategy):
    def __init__(self, nodes, data_gen):
        self.nodes = nodes
        self.num_nodes = len(nodes)
        self.data_gen = data_gen
        self.max_individuals_per_node = 3

    def evaluate(self, population):
        fitness_values = []

        population_size = population.shape[1]
        current_individual_idx = 0
        total_batch_indices = []

        while current_individual_idx < population_size:
            individuals_left = population_size - current_individual_idx
            if individuals_left > self.max_individuals_per_node * self.num_nodes:
                for node in self.nodes:
                    end_individual_idx = current_individual_idx + self.max_individuals_per_node
                    individuals = population[current_individual_idx:end_individual_idx]
                    current_individual_idx = end_individual_idx
                    node.load_individuals(individuals)

                    batches = self._get_next_batches(self.max_individuals_per_node)
                    node.load_batches(batches)

                    batch_indices = [idx for idx, _ in batches]
                    total_batch_indices.extend(batch_indices)

                    batch_order = range(self.max_individuals_per_node)
                    node.evaluate(batch_order)
            else:
                num_individuals_per_node = population_size // self.num_nodes
                remaining_individuals = population_size - num_individuals_per_node * self.num_nodes
                for node in self.nodes:
                    num_individuals = num_individuals_per_node
                    if remaining_individuals > 0:
                        remaining_individuals -= 1
                        num_individuals += 1

                    end_individual_idx = current_individual_idx + num_individuals
                    individuals = population[:, current_individual_idx:end_individual_idx]
                    current_individual_idx = end_individual_idx
                    node.load_individuals(individuals)

                    batches = self._get_next_batches(num_individuals)
                    node.load_batches(batches)

                    batch_indices = [idx for idx, _ in batches]
                    total_batch_indices.extend(batch_indices)

                    batch_order = range(num_individuals)
                    node.evaluate(batch_order)

            for node in self.nodes:
                result = node.get_fitness()
                fitness_values.extend(result)

            return fitness_values, total_batch_indices

    def _get_next_batches(self, num_batches):
        return [next(self.data_gen) for _ in range(num_batches)]
