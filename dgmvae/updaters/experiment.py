
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
        self.device = None
        self.x_org = None
        self.train_size = 0
        self.val_size = 0

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

    # def validation_step(self, batch, batch_idx, optimizer_idx=0):
    #     x, y = batch

    #     # Set device
    #     if self.device is None:
    #         self.device = x.device

    #     return self.model.loss_func(x, optimizer_idx=optimizer_idx,
    #                                 dataset_size=self.val_size)

    # def validation_epoch_end(self, outputs):
    #     # Accumulate val loss
    #     val_loss = torch.stack([x["loss"] for x in outputs]).mean()
    #     results = {
    #         "val_loss": val_loss,
    #         "log": {"val/loss": val_loss}
    #     }
    #     return results

    def configure_optimizers(self):
        optims = [torch.optim.Adam(self.model.parameters())]
        if self.model.second_optim is not None:
            optims.append(self.model.second_optim)
        return optims

    def reconstruct_images(self):
        pass

    def prepare_data(self):
        """Download dataset"""
        if self.dataset == "mnist":
            datasets.MNIST(root=self.root, train=True, download=True)
            datasets.MNIST(root=self.root, train=False, download=True)

    def train_dataloader(self):

        # Dataset
        if self.dataset == "mnist":
            _transform = self.data_transform()
            dataset = datasets.MNIST(root=self.root, train=True,
                                     transform=_transform)
        elif self.dataset == "dsprites":
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

    def val_dataloader(self):
        # Dataset
        _transform = self.data_transform()

        if self.dataset == "mnist":
            dataset = datasets.MNIST(root=self.root, train=False,
                                     transform=_transform)

        # Params for data loader
        params = {"batch_size": self.batch_size}

        # Loader
        loader = torch.utils.data.DataLoader(dataset, shuffle=False, **params)
        self.val_size = len(loader)

        # Sample image
        x_org, _ = iter(loader).next()
        self.x_org = x_org[:8]

        return loader

    @staticmethod
    def data_transform():
        _transform = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
        ])

        return _transform
