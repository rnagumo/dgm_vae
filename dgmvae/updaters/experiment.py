
"""VAE experiment with PyTorchLightning"""


import torch
from torchvision import datasets, transforms
import pytorch_lightning as pl


class VAEUpdater(pl.LightningModule):

    def __init__(self, model, params):
        super().__init__()

        self.model = model
        self.params = params
        self.device = None

        # Dataset parameter
        self.x_org = None
        self.train_size = 0
        self.test_size = 0

    def forward(self, inputs, **kwargs):
        return self.model(inputs, **kwargs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss_dict = self.model.loss_func({"x": x})

        for key in loss_dict:
            loss_dict[key] *= x.size(0)
        return loss_dict

    def training_end(self, outputs):
        loss_dict = {}

        # Accumulate 1-epoch loss
        for key in outputs[0]:
            loss_dict[f"train/{key}"] = \
                torch.stack([x[key] for x in outputs]).sum()

        # Standardize by dataset size
        for key in loss_dict:
            loss_dict[key] /= self.train_size

        results = {
            "loss": loss_dict["train/loss"],
            "progress_bar": {"training_loss": loss_dict["train/loss"]},
            "log": loss_dict
        }

        return results

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.device = x.device
        loss_dict = self.model.loss_func({"x": x})

        for key in loss_dict:
            loss_dict[key] *= x.size(0)
        return loss_dict

    def validation_end(self, outputs):
        loss_dict = {}

        # Accumulate 1-epoch loss
        for key in outputs[0]:
            loss_dict[f"val/{key}"] = \
                torch.stack([x[key] for x in outputs]).sum()

        # Standardize by dataset size
        for key in loss_dict:
            loss_dict[key] /= self.train_size

        results = {
            "val_loss": loss_dict["val/loss"],
            "progress_bar": {"val_loss": loss_dict["val/loss"]},
            "log": loss_dict
        }

        return results

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters())

    def reconstruct_images(self):
        pass

    def prepare_data(self):
        """Download dataset"""
        datasets.MNIST(root=self.params["root"], train=True, download=True)
        datasets.MNIST(root=self.params["root"], train=False, download=True)

    @pl.data_loader
    def train_dataloader(self):
        # Dataset
        _transform = self.data_transform()
        dataset = datasets.MNIST(root=self.params["root"], train=True,
                                 transform=_transform)

        # Params for data loader
        params = {"batch_size": self.params["batch_size"]}

        # Loader
        loader = torch.utils.data.DataLoader(dataset, shuffle=True, **params)
        self.train_size = len(loader)

        return loader

    @pl.data_loader
    def val_dataloader(self):
        # Dataset
        _transform = self.data_transform()
        dataset = datasets.MNIST(root=self.params["root"], train=False,
                                 transform=_transform)

        # Params for data loader
        params = {"batch_size": self.params["batch_size"]}

        # Loader
        loader = torch.utils.data.DataLoader(dataset, shuffle=False, **params)
        self.test_size = len(loader)

        # Sample image
        x_org, _ = iter(loader).next()
        self.x_org = x_org[:8]

        return loader

    def data_transform(self):
        _transform = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
        ])

        return _transform
