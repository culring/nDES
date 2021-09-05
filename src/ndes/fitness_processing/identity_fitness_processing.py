from ndes.fitness_processing.fitness_processing import FitnessProcessing


class IdentityFitnessProcessing(FitnessProcessing):
    def update_batch(self, batch_idx, loss):
        return loss

    def update_after_iteration(self):
        pass
