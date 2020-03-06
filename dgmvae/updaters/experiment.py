
"""VAE experiment with PyTorchLightning"""


import torch
from torchvision import datasets, transforms
import pytorch_lightning as pl


class VAEUpdater(pl.LightningModule):

    def __init__(self, model, root, batch_size, **kwargs):
        super().__init__()

        self.model = model
        self.root = root
        self.batch_size = batch_size

        # Dataset parameters
        self.device = None
        self.x_org = None
        self.train_size = 0
        self.test_size = 0

    def forward(self, inputs, **kwargs):
        return self.model(inputs, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        x, y = batch
        return self.model.loss_func({"x": x}, optimizer_idx=optimizer_idx)

    def training_end(self, outputs):
        """Training step end"""

        loss_dict = {}
        for key in outputs:
            loss_dict[f"train/{key}"] = outputs[key]

        results = {
            "loss": loss_dict["train/loss"],
            "progress_bar": {"training_loss": loss_dict["train/loss"]},
            "log": loss_dict
        }

        return results

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        x, y = batch

        # Set device
        if self.device is None:
            self.device = x.device

        return self.model.loss_func({"x": x}, optimizer_idx=optimizer_idx)

    def validation_end(self, outputs):
        """Validation epoch end"""

        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        results = {
            "val_loss": avg_loss,
            "log": {"val/loss": avg_loss}
        }
        return results

    def configure_optimizers(self):
        optims = [torch.optim.Adam(self.model.parameters())]
        if self.model.second_optim is not None:
            optims.append(self.model.second_optim)
        return optims

    def reconstruct_images(self):
        pass

    def prepare_data(self):
        """Download dataset"""
        datasets.MNIST(root=self.root, train=True, download=True)
        datasets.MNIST(root=self.root, train=False, download=True)

    @pl.data_loader
    def train_dataloader(self):
        # Dataset
        _transform = self.data_transform()
        dataset = datasets.MNIST(root=self.root, train=True,
                                 transform=_transform)

        # Params for data loader
        params = {"batch_size": self.batch_size}

        # Loader
        loader = torch.utils.data.DataLoader(dataset, shuffle=True, **params)
        self.train_size = len(loader)

        return loader

    @pl.data_loader
    def val_dataloader(self):
        # Dataset
        _transform = self.data_transform()
        dataset = datasets.MNIST(root=self.root, train=False,
                                 transform=_transform)

        # Params for data loader
        params = {"batch_size": self.batch_size}

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
