from ndes.node.gpu_node.command import CommandType


class GPUNodeClient:
    def __init__(self, model, queue_query, queue_result, logger, forward_backward_pass):
        self.model = model
        self.queue_query = queue_query
        self.queue_result = queue_result
        self.logger = logger
        self.forward_backward_pass = forward_backward_pass
        self.individuals = None
        self.batches = None

    def run(self):
        while True:
            self.logger.log("waiting for queue")
            command = self.queue_query.get()
            self.parse_command(command)

    def parse_command(self, command):
        command_type = command.command_type
        command_name = command_type.name
        self.logger.log(f"received {command_name} command")
        if command_type == CommandType.Stop:
            self.shutdown(command)
        elif command_type == CommandType.LoadIndividuals:
            self.load_individuals(command)
        elif command_type == CommandType.LoadBatches:
            self.load_batches(command)
        elif command_type == CommandType.Evaluate:
            self.evaluate(command)

    def shutdown(self, command):
        exit(0)

    def load_individuals(self, command):
        self.individuals = command.individuals

    def load_batches(self, command):
        self.batches = command.batches

    def evaluate(self, command):
        fitness_individuals = []
        for i, batch_order_idx in enumerate(command.batch_order):
            individual = self.individuals[:, i]
            batch = self.batches[batch_order_idx]
            fitness = self.forward_backward_pass.run(individual, self.model, batch)
            fitness_individuals.append(fitness)
        self.queue_result.put(fitness_individuals)


def worker(*args, **kwargs):
    gpu_node_client = GPUNodeClient(*args, **kwargs)
    gpu_node_client.run()
