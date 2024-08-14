from torch.utils.data import Dataset
import numpy as np
import os
import torch
import torch.nn as nn
from vector_quantize_pytorch import VectorQuantize

from animation.animation_util import backtransform_weights
from networks.mlp_models import MLP3D
from os.path import join

from utils import get_default_device

mlp_kwargs = {
    "out_size": 1,
    "hidden_neurons": [128, 128, 128],
    "use_leaky_relu": False,
    "input_dims": 3,
    "multires": 4,
    "include_input": True,
    "output_type": "occ",
}


class ShapeNetDataset(Dataset):
    def __init__(self, mlps_folder, transform=None, cpu=False):
        self.mlps_folder = mlps_folder

        if transform is None:
            self.transform = None
        elif hasattr(transform, "__iter__"):
            self.transform = transform
        else:
            self.transform = [transform]

        self.mlp_files = [file for file in list(os.listdir(mlps_folder))]
        self.mlp_files = [file for i, file in enumerate(self.mlp_files)]

        self.device = torch.device(get_default_device(cpu))
        self.mlp_kwargs = mlp_kwargs

    def __getitem__(self, index):
        file = join(self.mlps_folder, self.mlp_files[index])
        out = torch.load(file, map_location=self.device)
        y = "plane"

        if self.transform is not None:
            for t in self.transform:
                out, y = t(out, y)

        return out, y

    def __len__(self):
        return len(self.mlp_files)


class ModelTransform3D(torch.nn.Module):
    def __init__(self, weights_dict: dict = mlp_kwargs):
        super().__init__()
        self.weights_dict = weights_dict

    def forward(self, state_dict, y=None):
        if "state_dict" in state_dict:
            model = MLP3D(**state_dict["model_config"])
            model.load_state_dict(state_dict["state_dict"])
        else:
            model = MLP3D(**self.weights_dict)
            model.load_state_dict(state_dict)
        return model, y


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


# condition loading
dataset_model_unconditioned = ShapeNetDataset(
    os.path.join("./", "datasets", "shapenet_nef_2", "unconditioned"),
    transform=ImageTransform3D(),
)
condition = dataset_model_unconditioned[0][0]


# Transform that uses vq, first flatten data, then use vq for quantization and return indices and label
class TokenTransform3D(nn.Module):
    def __init__(self, vq: VectorQuantize):
        super().__init__()
        self.flatten = ImageTransform3D()
        self.vq = vq
        self.eval()

    def forward(self, weights_dict, y):
        # Apply min-max normalization
        weights, y = self.flatten(weights_dict, y)
        weights = weights - condition.to(weights)
        self.target_shape = weights.shape
        with torch.no_grad():
            flattened_weights = weights.view(-1, self.vq.layers[0].dim)
            _x, indices, _commit_loss = self.vq(flattened_weights, freeze_codebook=True)
            return indices, y

    def inverse(self, indices):
        quantized = self.vq.get_codes_from_indices(indices)
        # quantized_reshaped = quantized.view(-1, self.vq.dim)

        quantized = quantized.reshape(self.target_shape)

        return self.flatten.inverse(quantized + condition.to(quantized))

    def backproject(self, indices):
        return self.vq.get_codes_from_indices(indices)


class ModelTransform3DFromTokens(torch.nn.Module):
    def __init__(self, vq: VectorQuantize, weights_dict: dict = mlp_kwargs):
        super().__init__()
        self.weights_dict = weights_dict
        self.vq = vq
        self.token_transform = TokenTransform3D(vq)

    def forward(self, indices, y=None):
        weights = self.token_transform.backproject(indices)
        model = MLP3D(**self.weights_dict)

        prototyp = model.state_dict()

        model.load_state_dict(
            backtransform_weights(weights.flatten().unsqueeze(0), prototyp)
        )

        return model, y


# Transform that uses vq, first flatten data, then use vq for quantization and return indices and label
class WeightTransform3D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, state_dict, y):
        # Apply min-max normalization
        state_dict[f"layers.0.weight"]
        weights = torch.stack(
            state_dict[f"layers.0.weight"],
            state_dict[f"layers.1.weight"],
            state_dict[f"layers.2.weight"],
            torch.transpose(state_dict[f"layers.3.weight"], 0, 1),
        )
        return weights, y

    def backproject(self, indices):
        return self.vq.get_codes_from_indices(indices)


def get_neuron_mean_n_std(dataset: ShapeNetDataset):
    all_weights = torch.stack([sample[0] for sample in dataset])
    neuron_count = all_weights.shape[1]
    means = torch.stack([all_weights[:, i, :].mean(dim=0) for i in range(neuron_count)])
    stds = torch.stack([all_weights[:, i, :].std(dim=0) for i in range(neuron_count)])
    return means, stds


def get_total_mean_n_std(dataset: ShapeNetDataset, dim=128, norm_over_dim=None):
    all_weights = torch.stack([sample[0] for sample in dataset])
    if norm_over_dim:
        return all_weights.view(-1, dim).mean(dim=norm_over_dim), all_weights.view(
            -1, dim
        ).std(dim=norm_over_dim)
    else:
        return all_weights.view(-1, dim).mean(), all_weights.view(-1, dim).std()


class ZScore3D(nn.Module):
    def __init__(self, neuron_means, neuron_stds):
        super().__init__()
        self.neuron_means = neuron_means
        self.neuron_stds = neuron_stds

    def forward(self, neuron_weights, last_bias):
        # Apply min-max normalization
        normalized_neurons = (neuron_weights - self.neuron_means) / self.neuron_stds
        return normalized_neurons, last_bias

    def reverse(self, neuron_weights, last_bias):
        # Revers min-max normalization
        normalized_neurons = neuron_weights * self.neuron_stds + self.neuron_means
        return normalized_neurons, last_bias


# Transform that uses vq, first flatten data, then use vq for quantization and return indices and label
class AllWeights3D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, state_dict, y):
        # Apply min-max normalization
        all_weights = torch.cat(
            (
                state_dict[f"layers.0.weight"].T,
                state_dict[f"layers.1.weight"].T,
                state_dict[f"layers.2.weight"].T,
                state_dict[f"layers.3.weight"],
                state_dict[f"layers.0.bias"].unsqueeze(0),
                state_dict[f"layers.1.bias"].unsqueeze(0),
                state_dict[f"layers.2.bias"].unsqueeze(0),
            )
        )
        return all_weights, state_dict[f"layers.3.bias"]
