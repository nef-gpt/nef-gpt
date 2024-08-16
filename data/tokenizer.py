import torch
import abc
from vector_quantize_pytorch import ResidualVQ


# abstract base class for tokenizers (using abc)
class Tokenizer(abc.ABC):
    @abc.abstractmethod
    def special_tokens(self) -> dict:
        pass

    @abc.abstractmethod
    def vocab_size(self) -> int:
        pass

    @abc.abstractmethod
    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def decode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        pass


class ScalarTokenizer(Tokenizer):
    @staticmethod
    def load_from_file(file_path: str, device):
        rq_dict = torch.load(file_path, map_location=device, weights_only=False)

        state_dict = rq_dict["state_dict"]
        rq_config = rq_dict["rq_config"]

        rvq = ResidualVQ(**rq_config)
        rvq.load_state_dict(state_dict)
        rvq.to(device)
        rvq.eval()

        return ScalarTokenizer(rvq, rq_config)

    def __init__(self, quantizer: ResidualVQ, config: dict):
        self.quantizer = quantizer
        self.config = config

    def special_tokens(self):
        return {"sos": self.config["codebook_size"]}

    def _special_token_mask(self, x: torch.Tensor):
        mask = torch.zeros_like(x)
        for token in self.special_tokens().values():
            mask = mask | (x == token)
        return mask.to(bool)

    def vocab_size(self):
        return self.config["codebook_size"] + len(self.special_tokens().keys())

    @torch.no_grad()
    def encode_sequence(self, x: torch.Tensor):
        flattened = x.view(-1, self.quantizer.layers[0].dim)
        _x, indices, _commit_loss = self.quantizer(flattened)
        # flatten indices and add special token
        indices = indices.view(-1)
        start_tokens = (
            torch.tensor([self.special_tokens()["sos"]]).to(torch.long).to(indices)
        )
        return torch.cat([start_tokens, indices])

    @torch.no_grad()
    def decode_sequence(self, x: torch.Tensor):
        # filter out *all* special tokens
        x = x[~self._special_token_mask(x)]
        x = x.view(-1, self.quantizer.layers[0].dim)
        quantized = self.quantizer.get_codes_from_indices(x)
        return quantized.reshape(x.shape)
