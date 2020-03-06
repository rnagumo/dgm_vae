
"""VAE experiment with PyTorchLightning"""


import torch
from torchvision import datasets, transforms
import pytorch_lightning as pl


class VAEUpdater(pl.LightningModule):

    def __init__(self, model, hparams):
        super().__init__()

        self.model = model
        self.hparams = hparams

        # Dataset parameters
        self.device = None
        self.x_org = None
        self.train_size = 0
        self.test_size = 0

    def forward(self, inputs, **kwargs):
        return self.model(inputs, **kwargs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss_dict = self.model.loss_func({"x": x})

        results = {}
        for key in loss_dict:
            results[f"train/{key}"] = loss_dict[key]

        output = {
            "loss": results["train/loss"],
            "progress_bar": {"training_loss": results["train/loss"]},
            "log": results
        }

        return output

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # Set device
        if self.device is None:
            self.device = x.device

        return self.model.loss_func({"x": x})

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        results = {
            "val_loss": avg_loss,
            "log": {"val/loss": avg_loss}
        }
        return results

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters())

    def reconstruct_images(self):
        pass

    def prepare_data(self):
        """Download dataset"""
        datasets.MNIST(root=self.hparams.root, train=True, download=True)
        datasets.MNIST(root=self.hparams.root, train=False, download=True)

    @pl.data_loader
    def train_dataloader(self):
        # Dataset
        _transform = self.data_transform()
        dataset = datasets.MNIST(root=self.hparams.root, train=True,
                                 transform=_transform)

        # Params for data loader
        params = {"batch_size": self.hparams.batch_size}

        # Loader
        loader = torch.utils.data.DataLoader(dataset, shuffle=True, **params)
        self.train_size = len(loader)

        return loader

    @pl.data_loader
    def val_dataloader(self):
        # Dataset
        _transform = self.data_transform()
        dataset = datasets.MNIST(root=self.hparams["root"], train=False,
                                 transform=_transform)

        # Params for data loader
        params = {"batch_size": self.hparams.batch_size}

        # Loader
        loader = torch.utils.data.DataLoader(dataset, shuffle=False, **params)
        self.test_size = len(loader)

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
