from enum import Enum


class Command:
    def __init__(self, command_type):
        self.command_type = command_type


class StopCommand(Command):
    def __init__(self):
        super().__init__(CommandType.Stop)


class LoadIndividualsCommand(Command):
    def __init__(self, individuals):
        super().__init__(CommandType.LoadIndividuals)
        self.individuals = individuals


class LoadBatchesCommand(Command):
    def __init__(self, batches):
        super().__init__(CommandType.LoadBatches)
        self.batches = batches


class EvaluateCommand(Command):
    def __init__(self, batch_order):
        super().__init__(CommandType.Evaluate)
        self.batch_order = batch_order


class CommandType(Enum):
    Stop = "stop"
    LoadIndividuals = "load_individuals"
    LoadBatches = "load_batches"
    Evaluate = "evaluate"
