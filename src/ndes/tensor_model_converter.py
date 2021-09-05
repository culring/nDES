import torch


class TensorModelConverter:
    def __init__(self, layers_iter):
        self._layers_offsets_shapes = []
        self._init_layers_offets_shapes(layers_iter)

    def _init_layers_offets_shapes(self, layers_iter):
        current_offset = 0
        for param in layers_iter:
            shape = param.shape
            tmp = param.flatten()
            current_offset += len(tmp)
            self._layers_offsets_shapes.append((current_offset, shape))

    def zip_layers(self, layers_iter):
        """Concatenate flattened layers into a single 1-D tensor.
        This method also saves shapes of layers and their offsets in the final
        tensor, allowing for a fast unzip operation.

        Args:
            layers_iter: Iterator over model's layers.
        """
        tensors = []
        current_offset = 0
        for param in layers_iter:
            tmp = param.flatten()
            current_offset += len(tmp)
            tensors.append(tmp)
        return torch.cat(tensors, 0).contiguous()

    def unzip_layers(self, zipped_layers):
        """Iterator over 'unzipped' layers, with their proper shapes.

        Args:
            zipped_layers: Flattened representation of layers.
        """
        start = 0
        for offset, shape in self._layers_offsets_shapes:
            yield zipped_layers[start:offset].view(shape)
            start = offset