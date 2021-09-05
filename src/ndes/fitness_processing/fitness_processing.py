import abc
from enum import Enum


class FitnesssProcessingType(Enum):
    IDENTITY = 1,
    EWMA = 2


class FitnessProcessing(abc.ABC):
    @abc.abstractmethod
    def update_batch(self, batch_idx, loss):
        pass

    @abc.abstractmethod
    def update_after_iteration(self):
        pass
