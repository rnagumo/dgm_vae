
"""VAE experiment with PyTorchLightning"""


import torch
from torchvision import datasets, transforms
import pytorch_lightning as pl

from ..datasets.cars3d import Cars3dDataset
from ..datasets.dsprites import DSpritesDataset


class VAEUpdater(pl.LightningModule):

    def __init__(self, model, hparams, dataset, root, batch_size, **kwargs):
        super().__init__()

        self.model = model
        self.hparams = hparams
        self.dataset = dataset
        self.root = root
        self.batch_size = batch_size

        # Dataset parameters
        self.train_size = 0

    def forward(self, inputs, **kwargs):
        return self.model(inputs, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        x, y = batch
        outputs = self.model.loss_func(x, optimizer_idx=optimizer_idx,
                                       dataset_size=self.train_size)

        loss_dict = {}
        for key in outputs:
            loss_dict[f"train/{key}"] = outputs[key]

        if optimizer_idx == 0:
            results = {
                "loss": loss_dict["train/loss"],
                "progress_bar": {"training_loss": loss_dict["train/loss"]},
                "log": loss_dict,
            }
        else:
            results = {
                "loss": loss_dict["train/adv_loss"],
                "log": loss_dict,
            }

        return results

    def configure_optimizers(self):
        optims = [torch.optim.Adam(self.model.parameters())]
        if self.model.second_optim is not None:
            optims.append(self.model.second_optim)
        return optims

    def prepare_data(self):
        """Download dataset"""
        if self.dataset == "mnist":
            datasets.MNIST(root=self.root, train=True, download=True)
            datasets.MNIST(root=self.root, train=False, download=True)

    def train_dataloader(self):
        """Loads train data loader.

        The specified dataset name should be same in disentanglement_lib.
        https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/data/ground_truth/named_data.py
        """

        # Dataset
        if self.dataset == "mnist":
            _transform = transforms.Compose([
                transforms.Resize(64), transforms.ToTensor()])
            dataset = datasets.MNIST(root=self.root, train=True,
                                     transform=_transform)
        elif self.dataset == "dsprites_full":
            dataset = DSpritesDataset(root=self.root)
        elif self.dataset == "cars3d":
            dataset = Cars3dDataset(root=self.root)
        else:
            raise KeyError(f"Unexpected dataset is specified: {self.dataset}")

        # Params for data loader
        params = {"batch_size": self.batch_size}

        # Loader
        loader = torch.utils.data.DataLoader(dataset, shuffle=True, **params)
        self.train_size = len(loader)

        return loader
