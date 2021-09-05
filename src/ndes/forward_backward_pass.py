import torch

from ndes.ndes import SecondaryMutation


class ForwardBackwardPass:
    def __init__(self, secondary_mutation, criterion, lr, tensor_model_converter):
        self.secondary_mutation = secondary_mutation
        self.criterion = criterion
        self.lr = lr
        self.tensor_model_converter = tensor_model_converter

    def run(self, weights, model, batch):
        """Custom objective function for the DES optimizer."""
        batch_idx, (b_x, y) = batch
        self._reweight_model(model, weights)
        if self.secondary_mutation == SecondaryMutation.Gradient:
            gradient = []
            with torch.enable_grad():
                model.zero_grad()
                out = model(b_x)
                loss = self.criterion(out, y)
                loss.backward()
                for param in model.parameters():
                    gradient.append(param.grad.flatten())
                gradient = torch.cat(gradient, 0)
                # In-place mutation of the weights
                weights -= self.lr * gradient
        else:
            out = model(b_x)
            loss = self.criterion(out, y)
        loss = loss.item()
        return loss

    def _reweight_model(self, model, individual):
        for param, layer in zip(model.parameters(), self.tensor_model_converter.unzip_layers(individual)):
            param.data.copy_(layer)
