class Cycler:
    def __init__(self, batches):
        self.batch_idx = 0
        self.batches = batches

    def __iter__(self):
        return self

    def __next__(self):
        batch_idx = self.batch_idx
        self.batch_idx = (self.batch_idx + 1) % len(self.batches)
        return self.batches[batch_idx]

    def __len__(self):
        return len(self.batches)
