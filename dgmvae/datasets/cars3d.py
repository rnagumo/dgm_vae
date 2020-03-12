
"""Dataset class for cars3d"""

import pathlib
from PIL import Image

import numpy as np
import scipy.io as sio

import torch


class Cars3dDataset(torch.utils.data.Dataset):

    def __init__(self, root, subdir=None):
        super().__init__()

        # Configure path
        subdir = "data/cars/" if subdir is None else subdir
        root = pathlib.Path(root, subdir)

        # Load pre-downloaded data
        data = []
        targets = []
        for i, path in enumerate(root.glob("*.mat")):
            # Data
            data.append(torch.tensor(_load_mesh(path)))

            # Factor label
            targets.append(torch.tensor(_load_factor(i)))

        # Unsqueeze minibatch
        data = torch.stack(data).view(-1, 64, 64, 3)
        targets = torch.stack(targets).view(-1, 3)

        # Reshape dataset (batch, channel, height, width)
        self.data = data.permute(0, 3, 1, 2)
        self.targets = targets

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.data.size(0)


def _load_mesh(path):
    with open(path, "rb") as f:
        mesh = np.einsum("abcde->deabc", sio.loadmat(f)["im"])
    flattened_mesh = mesh.reshape((-1,) + mesh.shape[2:])
    rescaled_mesh = np.zeros((flattened_mesh.shape[0], 64, 64, 3))
    for i in range(flattened_mesh.shape[0]):
        pic = Image.fromarray(flattened_mesh[i])
        pic.thumbnail((64, 64, 3), Image.ANTIALIAS)
        rescaled_mesh[i] = np.array(pic)
    return rescaled_mesh * 1. / 255


def _load_factor(idx):
    factor1 = np.arange(4)
    factor2 = np.arange(24)
    all_factors = np.transpose([
        np.tile(factor1, len(factor2)),
        np.repeat(factor2, len(factor1)),
        np.tile(idx, len(factor1) * len(factor2)),
    ])
    return all_factors
