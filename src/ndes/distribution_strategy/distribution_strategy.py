import abc


class DistributionStrategy(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, population):
        pass
