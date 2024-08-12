from dataclasses import dataclass
import inspect
import math
from typing import Optional
from lightning.pytorch import LightningModule
from torch import nn
import torch.nn.functional as F
import torch
import tqdm
from components import Block, LayerNorm, PositionalEncoding


@dataclass
class NanoGPTConfig:
    block_size: int = 1024
    vocab_size: int = (
        1024  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    )
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = (
        True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    )

    max_len: Optional[int] = (
        None  # 36737  # maximum number of tokens in a sequence (for global positional encoding)
    )


class NanoGPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        if self.config.max_len is not None:
            self.global_pos_emb = PositionalEncoding(
                config.n_embd, max_len=config.max_len
            )

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                # wts=nn.Embedding(1 + 3 * 2 + 1, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def entropy_loss(self, logits, reduction="mean"):
        """
        Computes the entropy loss for a batch of logits.

        Enforcing high entropy in the model's predictions across a batch can help to prevent overfitting

        Args:
            logits (torch.Tensor): The input logits with shape (batch_size, num_classes).

        Returns:
            torch.Tensor: The entropy loss.
        """

        # compute crossentropy for each item in the batch, against all others
        b, s, v = logits.shape

        if True:
            return 0

        if b == 1:
            return 0

        entropy = torch.stack(
            [
                torch.nn.functional.cross_entropy(
                    logits[i].repeat((3, 1, 1)),
                    torch.softmax(logits[[j for j in range(b) if j != i]], dim=2),
                )
                for i in range(b)
            ]
        ).mean()

        # get residual loss between entropy and target
        target = 140

        return (entropy - target) ** 2

    def forward(self, idx, targets=None, offset=None, reduction="mean"):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        # pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        pos_emb = (
            self.global_pos_emb.get_pe(self.config.block_size, offset)[:, 0:t].squeeze(
                -2
            )
            if self.config.max_len is not None
            else self.transformer.wpe(
                torch.arange(0, t, dtype=torch.long, device=device)
            )
        )

        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            if reduction == "none":
                b, s, l = logits.size()
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-1,
                    reduction=reduction,
                )

                # calculate entropy loss across the batch
                # entropy_loss = self.entropy_loss(logits)
                loss = loss.view(b, -1).mean(dim=-1)
                mean_loss = loss.mean()  # + entropy_loss
            else:
                mean_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
                ) + self.entropy_loss(logits)
                loss = None
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None
            mean_loss = None

        if reduction == "none":
            return logits, mean_loss, loss

        return logits, mean_loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    # @classmethod
    # def from_pretrained(cls, model_type, override_args=None):
    #     assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
    #     override_args = override_args or {}  # default to empty dict
    #     # only dropout can be overridden see more notes below
    #     assert all(k == "dropout" for k in override_args)
    #     from transformers import GPT2LMHeadModel
    #     print("loading weights from pretrained gpt: %s" % model_type)
    #     # n_layer, n_head and n_embd are determined from model_type
    #     config_args = {
    #         "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
    #         "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
    #         "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
    #         "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
    #     }[model_type]
    #     print("forcing vocab_size=50257, block_size=1024, bias=True")
    #     config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
    #     config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
    #     config_args["bias"] = True  # always True for GPT model checkpoints
    #     # we can override the dropout rate, if desired
    #     if "dropout" in override_args:
    #         print(f"overriding dropout rate to {override_args['dropout']}")
    #         config_args["dropout"] = override_args["dropout"]
    #     # create a from-scratch initialized minGPT model
    #     config = NanoGPTConfig(**config_args)
    #     model = NanoGPT(config)
    #     sd = model.state_dict()
    #     sd_keys = sd.keys()
    #     sd_keys = [
    #         k for k in sd_keys if not k.endswith(".attn.bias")
    #     ]  # discard this mask / buffer, not a param
    #     # init a huggingface/transformers model
    #     model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    #     sd_hf = model_hf.state_dict()
    #     # copy while ensuring all of the parameters are aligned and match in names and shapes
    #     sd_keys_hf = sd_hf.keys()
    #     sd_keys_hf = [
    #         k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
    #     ]  # ignore these, just a buffer
    #     sd_keys_hf = [
    #         k for k in sd_keys_hf if not k.endswith(".attn.bias")
    #     ]  # same, just the mask (buffer)
    #     transposed = [
    #         "attn.c_attn.weight",
    #         "attn.c_proj.weight",
    #         "mlp.c_fc.weight",
    #         "mlp.c_proj.weight",
    #     ]
    #     # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
    #     # this means that we have to transpose these weights when we import them
    #     assert len(sd_keys_hf) == len(
    #         sd_keys
    #     ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
    #     for k in sd_keys_hf:
    #         if any(k.endswith(w) for w in transposed):
    #             # special treatment for the Conv1D weights we need to transpose
    #             assert sd_hf[k].shape[::-1] == sd[k].shape
    #             with torch.no_grad():
    #                 sd[k].copy_(sd_hf[k].t())
    #         else:
    #             # vanilla copy over the other parameters
    #             assert sd_hf[k].shape == sd[k].shape
    #             with torch.no_grad():
    #                 sd[k].copy_(sd_hf[k])
    #     return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate_logits(self, idx):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and predict the next token.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        idx_cond = (
            idx
            if idx.size(1) <= self.config.block_size
            else idx[:, -self.config.block_size :]
        )
        logits, _ = self(idx_cond)
        return logits[:, -1, :]

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in tqdm(range(max_new_tokens), desc="Autoregressive Generation"):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            # offset = torch.arange(idx.size(1))[-self.config.block_size :].to(idx.device).int()
            offset = (
                torch.Tensor([max(0, idx.size(1) - self.config.block_size)])
                .int()
                .unsqueeze(0)
                .to(idx.device)
            )
            # TODO: Offset is currently not passed down
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

            # free everything but index
            del probs
            del logits
            del idx_next

        return idx
