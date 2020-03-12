
"""Dataset class for dSprites"""

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
        with open(path, "rb") as f:
            dataset = np.load(f, encoding="latin1", allow_pickle=True)
            data = torch.tensor(dataset["imgs"])
            targets = torch.tensor(dataset["latents_classes"])

        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        # Image size (channel, height, width)
        img = self.data[index].unsqueeze(0)
        target = self.targets[index]
        return img, target

    def __len__(self):
        return len(self.data)
