
"""Dataset class for cars3d

ref)
https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/data/ground_truth/cars3d.py
"""

import pathlib
from PIL import Image

import numpy as np
import scipy.io as sio
import torch

from .base_data import BaseDataset


class Cars3dDataset(BaseDataset):
    """Cars3D data set.

    The data set was first used in the paper "Deep Visual Analogy-Making"
    (https://papers.nips.cc/paper/5845-deep-visual-analogy-making) and can be
    downloaded from http://www.scottreed.info/. The images are rescaled to
    64x64.

    The ground-truth factors of variation are:
    0 - elevation (4 different values)
    1 - azimuth (24 different values)
    2 - object type (183 different values)
    """

    def __init__(self, root):
        super().__init__()

        # Load pre-downloaded data
        data = []
        targets = []
        for i, path in enumerate(pathlib.Path(root).glob("*.mat")):
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
        # Change dtype uint8 -> float32
        return self.data[index].float(), self.targets[index]

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
