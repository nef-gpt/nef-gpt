import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import distributions as dist
from torch import nn


# https://github.com/yenchenlin/nerf-pytorch/blob/63a5a630c9abd62b0f21c08703d0ac2ea7d4b9dd/run_nerf_helpers.py#L48
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)
        # 1 0 yap
        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class MLP(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        hidden_neurons,
        use_tanh=True,
        over_param=False,
        use_bias=True,
    ):
        super().__init__()
        multires = 1
        self.over_param = over_param
        if not over_param:
            self.embedder = Embedder(
                include_input=True,
                input_dims=2,
                max_freq_log2=multires - 1,
                num_freqs=multires,
                log_sampling=True,
                periodic_fns=[torch.sin, torch.cos],
            )
        self.layers = nn.ModuleList([])

        self.layers.append(nn.Linear(in_size, hidden_neurons[0], bias=use_bias))
        for i, _ in enumerate(hidden_neurons[:-1]):
            self.layers.append(
                nn.Linear(hidden_neurons[i], hidden_neurons[i + 1], bias=use_bias)
            )
        self.layers.append(nn.Linear(hidden_neurons[-1], out_size, bias=use_bias))
        self.use_tanh = use_tanh

    def forward(self, x):
        if not self.over_param:
            x = self.embedder.embed(x)
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = F.relu(x)
        x = self.layers[-1](x)
        if self.use_tanh:
            x = torch.tanh(x)
        return x, None


class MLP3D(nn.Module):
    def __init__(
        self,
        out_size,
        hidden_neurons,
        use_gelu=False,
        use_bias=True,
        multires=10,
        output_type=None,
        input_dims=2,
        weight_init=None,
        include_input=True,
        **kwargs,
    ):
        super().__init__()
        self.embedder = Embedder(
            include_input=include_input,
            input_dims=input_dims,
            max_freq_log2=multires - 1,
            num_freqs=multires,
            log_sampling=True,
            periodic_fns=[torch.sin, torch.cos],
        )
        self.layers = nn.ModuleList([])
        self.output_type = output_type
        self.use_gelu = use_gelu
        in_size = self.embedder.out_dim
        first_layer = nn.Linear(in_size, hidden_neurons[0], bias=use_bias)
        if weight_init:
            weight_init(first_layer.weight)
        self.layers.append(first_layer)
        for i, _ in enumerate(hidden_neurons[:-1]):
                hidden_layer = nn.Linear(hidden_neurons[i], hidden_neurons[i + 1], bias=use_bias)
                if weight_init:
                    weight_init(hidden_layer.weight)
                self.layers.append(hidden_layer)

        output_layer = nn.Linear(hidden_neurons[-1], out_size, bias=False)
        if weight_init:
            weight_init(output_layer.weight)
        self.layers.append(output_layer)

    def forward(self, model_input):
        coords_org = model_input.clone().detach().requires_grad_(True)
        x = coords_org
        x = self.embedder.embed(x)
        # using a relu in the last hidden layer improves the performance
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = F.gelu(x) if self.use_gelu else F.relu(x)
            
        x = self.layers[-1](x)

        if self.output_type == "occ":
            # x = torch.sigmoid(x)
            pass
        elif self.output_type == "sdf":
            x = torch.tanh(x)
        elif self.output_type == "logits":
            x = x
        else:
            raise f"This self.output_type ({self.output_type}) not implemented"
        x = dist.Bernoulli(logits=x).logits

        return x
