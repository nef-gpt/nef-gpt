from typing import Optional
from vector_quantize_pytorch import VectorQuantize
import os
from PIL import Image
import torch

from models.mlp_models import MLP3D
from models.nano_gpt import NanoGPT
from utils import backtransform_weights


def generate_neural_field(
    model: NanoGPT,
    vq: VectorQuantize,
    sos: int,
    condition: list[int],
    device,
    config: dict,
    template_state_dict: Optional[dict] = None,
    top_k=None,
    temperature=1.0,
) -> MLP3D:
    seed = torch.zeros((len(condition), 2)).long()
    seed[:, 0] = sos
    seed[:, 1] = torch.Tensor(condition).long()

    novel_tokens = model.generate(
        seed.to(device), model.config.block_size, temperature=temperature, top_k=top_k
    )[:, 2:]
    novel_tokens = novel_tokens.to("cpu")
    novel_weights = vq.get_codes_from_indices(
        (novel_tokens.clamp(0, vq.codebook_size - 1))
    )

    def make_model(i):
        mlp3d = MLP3D(**config)
        template_state_dict = template_state_dict or mlp3d.state_dict()
        reconstructed_dict = backtransform_weights(
            novel_weights[i].unsqueeze(0), template_state_dict
        )

        mlp3d.load_state_dict(reconstructed_dict)
        return mlp3d

    return [make_model(i) for i in range(len(condition))]
