
"""Dataset class for dSprites

ref)
https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/data/ground_truth/dsprites.py
"""

import pathlib

import numpy as np
import torch


class DSpritesDataset(torch.utils.data.Dataset):

    def __init__(self, root, filename=None):
        super().__init__()

        # Load pre-downloaded dataset
        filename = "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz" \
            if filename is None else filename
        path = pathlib.Path(root, filename)
        with np.load(path, encoding="latin1", allow_pickle=True) as dataset:
            data = torch.tensor(dataset["imgs"])
            targets = torch.tensor(dataset["latents_classes"])

        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        # Reshape dataset (channel, height, width)
        return self.data[index].unsqueeze(0).float(), self.targets[index]

    def __len__(self):
        return self.data.size(0)
