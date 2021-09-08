from ndes.distribution_strategy.distribution_strategy import DistributionStrategy


class CompleteLoadingStrategy(DistributionStrategy):
    def __init__(self, nodes, data_gen, batches):
        self.nodes = nodes
        self.data_gen = data_gen
        self.batches = batches
        self._send_batches_to_nodes()

    def _send_batches_to_nodes(self):
        for node in self.nodes:
            node.load_batches(self.batches)

    def evaluate(self, population):
        fitness_values = []

        population_size = population.shape[1]
        num_nodes = len(self.nodes)
        num_individuals_per_node = population_size // num_nodes
        remaining_individuals = population_size - num_individuals_per_node * num_nodes
        current_individual_idx = 0

        total_batch_indices = []

        for i in range(num_nodes):
            num_individuals = num_individuals_per_node
            if remaining_individuals > 0:
                remaining_individuals -= 1
                num_individuals += 1

            current_node = self.nodes[i]

            end_individual_idx = current_individual_idx + num_individuals
            individuals = population[:, current_individual_idx:end_individual_idx]
            current_individual_idx = end_individual_idx
            current_node.load_individuals(individuals)

            batch_indices = self._get_next_batches_indices(num_individuals)
            total_batch_indices.extend(batch_indices)

            batch_order = batch_indices
            current_node.evaluate(batch_order)

        for i in range(num_nodes):
            result = self.nodes[i].get_fitness()
            fitness_values.extend(result)

        return fitness_values, total_batch_indices

    def _get_next_batches_indices(self, num_batches):
        return [next(self.data_gen)[0] for _ in range(num_batches)]
