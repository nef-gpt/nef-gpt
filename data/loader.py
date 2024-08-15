from dataclasses import dataclass
from typing import Any
import lightning as pl
import os
import torch
from torch.utils.data import random_split

from data.dataset import ShapeNetDataset
from vector_quantize_pytorch import ResidualVQ
from einops import rearrange

from data.tokenizer import ScalarTokenizer
from data.transforms import TokenTransform3D

# Implement a pytorch lightning datamodule for the ShapeNetDataset class in data/loader.py
# This module functions as following:


@dataclass
class ShapeNetDataConfig:
    mlps_folder: str
    vector_quantize_pth: str
    tmp_folder: str = "tmp"
    batch_size: int = 2
    force_recompute: bool = False


class ShapeNetData(pl.LightningModule):
    def __init__(self, data_config: ShapeNetDataConfig) -> None:
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self):

        # Token transform
        self.tokenizer = ScalarTokenizer.load_from_file(
            self.hparams.data_config.vector_quantize_pth, self.device
        )
        self.token_transform = TokenTransform3D(self.tokenizer)

        dataset = ShapeNetDataset(
            self.hparams.data_config.mlps_folder,
            transform=self.token_transform,
            device=self.device,
        )

        # load the whole dataset to memory
        # take only the tensors, concatenate them and save them to disk
        # this is a one-time operation
        # TODO: Get the hash for the dataset (hash of the mlps_folder)
        # NOTE currently this is handled by the force_recompute flag
        def do_dataset(dataset, stage: str):
            if not os.path.exists(self.hparams.data_config.tmp_folder):
                os.makedirs(self.hparams.data_config.tmp_folder)
            # only save if it doesn't exist
            if self.hparams.data_config.force_recompute or not os.path.exists(
                f"{self.hparams.data_config.tmp_folder}/{stage}.pt"
            ):
                path = f"{self.hparams.data_config.tmp_folder}/{stage}.pt"
                loaded_dataset = torch.stack(
                    # TODO: this is bulsshit
                    [torch.stack(dataset[i]) for i in range(len(dataset))],
                    dim=0,
                )
                # rearrange the data, currently it is [N, (X, Y), L] where L is the length of the sequence, N is the number of samples, X is the input and Y is the output
                # we want to have [(X, Y), N, L]
                loaded_dataset = rearrange(loaded_dataset, "N T L -> T N L")
                torch.save(loaded_dataset, path)

        generator = torch.Generator().manual_seed(42)
        splits = random_split(dataset, [0.8, 0.15, 0.05], generator)
        for i, stage in enumerate(["train", "val", "test"]):
            do_dataset(splits[i], stage)

    def setup(self, stage: str):
        # load it back here
        # stage is {fit,validate,test,predict}
        if stage == "fit":
            data = torch.load(
                f"{self.hparams.data_config.tmp_folder}/train.pt", weights_only=True
            )
            self.train_data = torch.utils.data.TensorDataset(data[0], data[1])

            data = torch.load(
                f"{self.hparams.data_config.tmp_folder}/val.pt", weights_only=True
            )
            self.val_data = torch.utils.data.TensorDataset(data[0], data[1])
        elif stage == "test":
            data = torch.load(
                f"{self.hparams.data_config.tmp_folder}/test.pt", weights_only=True
            )
            self.test_data = torch.utils.data.TensorDataset(data[0], data[1])
        elif stage == "validate":
            data = torch.load(
                f"{self.hparams.data_config.tmp_folder}/val.pt", weights_only=True
            )
            self.val_data = torch.utils.data.TensorDataset(data[0], data[1])
        else:
            raise ValueError(f"Stage {stage} not recognized")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.hparams.data_config.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data, batch_size=self.hparams.data_config.batch_size, shuffle=False
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_data,
            batch_size=self.hparams.data_config.batch_size,
            shuffle=False,
        )

    def predict_dataloader(self):
        raise NotImplementedError
