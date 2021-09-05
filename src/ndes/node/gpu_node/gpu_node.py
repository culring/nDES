import copy

import torch

from ndes.node.gpu_node.command import LoadIndividualsCommand, LoadBatchesCommand, EvaluateCommand, StopCommand
from ndes.node.gpu_node.gpu_node_client import worker
from ndes.logger import Logger
from ndes.node.node import Node


class GPUNode(Node):
    def __init__(self, device, model, forward_backward_pass):
        self.device = device
        self.model = copy.deepcopy(model).to(device)
        self.queue_query = None
        self.queue_result = None
        self.logger = Logger(str(device))
        self.forward_backward_pass = forward_backward_pass
        self.job = None
        self._init_job()

    def _init_job(self):
        self.queue_query = torch.multiprocessing.Queue()
        self.queue_result = torch.multiprocessing.Queue()
        job = torch.multiprocessing.Process(target=worker,
                                            args=(self.model,
                                                  self.queue_query,
                                                  self.queue_result,
                                                  self.logger,
                                                  self.forward_backward_pass))
        job.start()
        self.job = job

    def load_individuals(self, individuals):
        command = LoadIndividualsCommand(individuals.to(self.device))
        self.queue_query.put(command)

    def load_batches(self, batches):
        batches_device = self._copy_batches_to_device(batches)
        command = LoadBatchesCommand(batches_device)
        self.queue_query.put(command)

    def _copy_batches_to_device(self, batches):
        batches_device = []
        for batch in batches:
            batch_idx, (x, y) = batch
            batch_device = batch_idx, (x.to(self.device), y.to(self.device))
            batches_device.append(batch_device)

        return batches_device

    def evaluate(self, batch_order):
        command = EvaluateCommand(batch_order)
        self.queue_query.put(command)

    def get_fitness(self):
        return self.queue_result.get()

    def get_capacity(self):
        pass

    def cleanup(self):
        command = StopCommand()
        self.queue_query.put(command)
        self.queue_query.close()
        self.queue_result.close()
        self.job.join()
