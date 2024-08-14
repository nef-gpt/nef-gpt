from dataclasses import dataclass
from typing import Any
import lightning as pl
import os
import torch
from torch.utils.data import random_split

from data.dataset import ShapeNetDataset
from vector_quantize_pytorch import ResidualVQ

from data.tokenization import TokenTransform3D

# Implement a pytorch lightning datamodule for the ShapeNetDataset class in data/loader.py
# This module functions as following:


@dataclass
class ShapeNetDataConfig:
    mlps_folder: str
    vector_quantize_pth: str
    tmp_folder: str = "tmp"
    batch_size: int = 32


class ShapeNetData(pl.LightningModule):
    def __init__(self, data_config: ShapeNetDataConfig) -> None:
        super().__init__()
        self.save_hyperparameters()

    def load_rvq(self):
        rq_dict = torch.load(
            self.hparams.data_config.vector_quantize_pth, map_location=self.device
        )
        rq_dict.keys()

        state_dict = rq_dict["state_dict"]
        rq_config = rq_dict["rq_config"]

        rvq = ResidualVQ(**rq_config)
        rvq.load_state_dict(state_dict)

        return rvq

    def prepare_data(self):

        # Token transform
        rvq = self.load_rvq()
        self.token_transform = TokenTransform3D(rvq)

        dataset = ShapeNetDataset(
            self.hparams.data_config.mlps_folder,
            transform=self.token_transform,
            device=self.device,
        )

        # load the whole dataset to memory
        # take only the tensors, concatenate them and save them to disk
        # this is a one-time operation
        # TODO: Get the hash for the dataset (hash of the mlps_folder)

        def do_dataset(dataset, stage: str):
            loaded_dataset_y = torch.stack(
                # TODO: this is bulsshit
                [dataset[i].view(-1) for i in range(len(dataset))],
                dim=0,
            )
            if not os.path.exists(self.hparams.data_config.tmp_folder):
                os.makedirs(self.hparams.data_config.tmp_folder)
            path = f"{self.hparams.data_config.tmp_folder}/{stage}.pt"
            torch.save(loaded_dataset, path)

        generator = torch.Generator().manual_seed(42)
        splits = random_split(dataset, [0.8, 0.15, 0.05], generator)
        for i, stage in enumerate(["train", "val", "test"]):
            do_dataset(splits[i], stage)

    def setup(self, stage: str):
        # load it back here
        # stage is {fit,validate,test,predict}
        if stage == "fit":
            self.train_data = torch.load(
                f"{self.hparams.data_config.tmp_folder}/train.pt"
            )
            self.val_data = torch.load(f"{self.hparams.data_config.tmp_folder}/val.pt")
        elif stage == "test":
            self.test_data = torch.load(
                f"{self.hparams.data_config.tmp_folder}/test.pt"
            )
        elif stage == "validate":
            self.val_data = torch.load(f"{self.hparams.data_config.tmp_folder}/val.pt")
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
