import abc


class Node(abc.ABC):
    @abc.abstractmethod
    def load_individuals(self, individuals):
        pass

    @abc.abstractmethod
    def load_batches(self, batches):
        pass

    @abc.abstractmethod
    def evaluate(self, batch_order):
        pass

    @abc.abstractmethod
    def get_fitness(self):
        pass

    @abc.abstractmethod
    def get_capacity(self):
        pass

    def cleanup(self):
        pass
