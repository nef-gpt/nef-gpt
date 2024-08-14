import numpy as np
import os
import torch
import torch.nn as nn
from vector_quantize_pytorch import VectorQuantize


# class ModelTransform3D(torch.nn.Module):
#     def __init__(self, weights_dict: dict = mlp_kwargs):
#         super().__init__()
#         self.weights_dict = weights_dict

#     def forward(self, state_dict, y=None):
#         if "state_dict" in state_dict:
#             model = MLP3D(**state_dict["model_config"])
#             model.load_state_dict(state_dict["state_dict"])
#         else:
#             model = MLP3D(**self.weights_dict)
#             model.load_state_dict(state_dict)
#         return model, y


class FlattenTransform3D(torch.nn.Module):
    def forward(self, state_dict, y):
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        weights = torch.cat([state_dict[key].flatten() for key in state_dict.keys()])
        return weights, y


class ImageTransform3D(nn.Module):
    """
    Transforms a model to a 2D image

    Padding is applied to the weights to make them square
    """

    def __init__(self):
        super().__init__()

    def forward(self, state_dict, y, dim=32):
        # Store reference to original state_dict (for inverse)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        self.original_state_dict = state_dict

        # random tensor from state_dict
        t = state_dict[list(state_dict.keys())[0]]

        cat = torch.cat([state_dict[key].view(-1) for key in state_dict], dim=0)
        if cat.shape[0] % dim != 0:
            cat = torch.cat([cat, torch.zeros(dim - cat.shape[0] % dim).to(t)])
        cat = cat.view(1, dim, -1)
        return cat, y

    def inverse(self, cat, model_dict=None):
        # flatten
        # cat = cat.view(-1)
        cat = cat.flatten()

        assert (
            model_dict is not None or self.original_state_dict is not None
        ), "[ImageTransform3D]-inverse No model_dict provided"
        if model_dict is None:
            model_dict = self.original_state_dict
        i = 0
        for key in model_dict:
            model_dict[key] = cat[i : i + model_dict[key].numel()].view(
                model_dict[key].shape
            )
            i += model_dict[key].numel()
        return model_dict


class TrainingTransform3D(nn.Module):
    """
    Takes a token sequence and returns a pair of the token sequence and the target token sequence
    (shifted by one position)
    """

    def __init__(self, sos: int):
        self.start_tokens = torch.tensor([sos]).to(torch.long)

    def forward(self, indices, y):
        sample = torch.cat([self.start_tokens, indices[:]])

        X = sample[:-1]  # exclude last token
        Y = sample[1:]  # exclude first token

        return X, Y


# Transform that uses vq, first flatten data, then use vq for quantization and return indices and label
class TokenTransform3D(nn.Module):
    def __init__(self, vq: VectorQuantize, sos: int, condition: torch.Tensor = None):
        super().__init__()
        self.flatten = ImageTransform3D()
        self.training_transform = TrainingTransform3D(sos)
        self.vq = vq
        self.condition = condition
        self.eval()

    def forward(self, weights_dict, y):
        # Apply min-max normalization
        weights, y = self.flatten(weights_dict, y)
        if self.condition is not None:
            weights = weights - self.condition.to(weights)
        self.target_shape = weights.shape
        with torch.no_grad():
            flattened_weights = weights.view(-1, self.vq.layers[0].dim)
            _x, indices, _commit_loss = self.vq(flattened_weights, freeze_codebook=True)

            return self.training_transform(indices, y)

    def inverse(self, indices):
        quantized = self.vq.get_codes_from_indices(indices)
        # quantized_reshaped = quantized.view(-1, self.vq.dim)

        quantized = quantized.reshape(self.target_shape)

        if self.condition is not None:
            return self.flatten.inverse(quantized + self.condition.to(quantized))
        return self.flatten.inverse(quantized)

    def backproject(self, indices):
        return self.vq.get_codes_from_indices(indices)
