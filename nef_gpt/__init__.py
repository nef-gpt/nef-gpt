# Start the experiment

from dataclasses import asdict, dataclass
from typing import Any, Tuple
import lightning as pl
import torch

from models.nano_gpt import NanoGPT, NanoGPTConfig


@dataclass
class OptimizerConfig:
    learning_rate: float = 2e-3
    weight_decay: float = 0.05
    betas: Tuple[float, float] = (0.9, 0.95)


@dataclass
class LRConfig:
    T_max: int = 10


class NefGPT(pl.LightningModule):
    """
    Implementation of NefGPT from adl4cv as a lightning module

    * Current caveats (in comparison to original training_script)
    - LR Decay is implemented as CosineAnnealingLR
    """

    def __init__(
        self,
        config: NanoGPTConfig,
        optimizer_config: OptimizerConfig,
        lr_config: LRConfig,
    ):
        super().__init__()
        self.save_hyperparameters()

        # model
        self.model = NanoGPT(self.hparams.config)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        X, Y = batch
        _, loss = self.model(X, Y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        X, Y = batch
        _, loss = self.model(X, Y)
        self.log("val/loss", loss)

        # TODO: Metrics as proposed in report
        return loss

    def configure_optimizers(self):
        optimizer_cfg = self.hparams.optimizer_config
        optimizer = self.model.configure_optimizers(
            device_type=self.device.type, **asdict(optimizer_cfg)
        )

        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, **asdict(self.hparams.lr_config)
            ),
            "interval": "step",
            "name": "CosineAnnealingLR - Scheduler",
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
