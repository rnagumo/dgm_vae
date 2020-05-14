
"""Dataset class for 3D-shapes

ref)
https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/data/ground_truth/shapes3d.py
"""

import pathlib

import numpy as np
import torch

from .base_data import BaseDataset


class Shapes3D(BaseDataset):
    """Shapes3D dataset.

    The data set was originally introduced in "Disentangling by Factorising".

    The ground-truth factors of variation are:
    0 - floor color (10 different values)
    1 - wall color (10 different values)
    2 - object color (10 different values)
    3 - object size (8 different values)
    4 - object type (4 different values)
    5 - azimuth (15 different values)

    Args:
        root (str): Path to root directory of data.
        filename (str, optional): File name of dataset.
    """

    def __init__(self, root, filename=None):
        super().__init__()

        # Load pre-downloaded dataset
        filename = "3dshapes.npz" if filename is None else filename
        path = pathlib.Path(root, filename)

        with np.load(path) as dataset:
            # Load data and permute dims NHWC -> NCHW
            self.data = torch.tensor(dataset["images"]).permute(0, 3, 1, 2)

        self.factor_sizes = [10, 10, 10, 8, 4, 15]
        self.targets = torch.cartesian_prod(
            *[torch.arange(v) for v in self.factor_sizes])

    def __getitem__(self, index):
        # Change data dtype and normalize
        return self.data[index].float() / 255, self.targets[index]

    def __len__(self):
        return self.data.size(0)
