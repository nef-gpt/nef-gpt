from torch.utils.data import Dataset
import torch
import os
from os.path import join


class ShapeNetDataset(Dataset):
    def __init__(self, mlps_folder, transform=None, device=None):
        self.mlps_folder = mlps_folder

        if transform is None:
            self.transform = None
        elif hasattr(transform, "__iter__"):
            self.transform = transform
        else:
            self.transform = [transform]

        self.mlp_files = [file for file in list(os.listdir(mlps_folder))]
        self.mlp_files = [file for i, file in enumerate(self.mlp_files)]

        self.device = device
        # self.device = torch.device(get_default_device(cpu))
        # self.mlp_kwargs = mlp_kwargs

    def __getitem__(self, index):
        file = join(self.mlps_folder, self.mlp_files[index])
        # NOTE weights_only is due to a security issue reported from torch, it is not necessary
        out = torch.load(file, map_location=self.device, weights_only=True)
        y = "plane"

        if self.transform is not None:
            for t in self.transform:
                out, y = t(out, y)

        return out, y

    def __len__(self):
        return len(self.mlp_files)
