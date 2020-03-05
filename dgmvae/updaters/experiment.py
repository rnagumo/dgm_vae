
"""VAE experiment with PyTorchLightning"""


import torch
from torchvision import datasets, transforms
import pytorch_lightning as pl


class VAEUpdater(pl.LightningModule):

    def __init__(self, model, params):
        super().__init__()

        self.model = model
        self.params = params
        self.device = params["device"]
        self.loader = None

    def forward(self, inputs, **kwargs):
        return self.model(inputs, **kwargs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss_dict = self.model.loss_function({"x": x})

        for key in loss_dict:
            loss_dict[key] *= x.size(0)

        return loss_dict

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss_dict = self.model.loss_function({"x": x})

        for key in loss_dict:
            loss_dict[key] *= x.size(0)

        return loss_dict

    def validation_end(self, outputs):

        # Calculate average loss
        avg_loss = torch.stack([x["loss"] for x in outputs]).sum()
        avg_loss /= len(outputs)

        # Returned doct
        results = {
            "progress_bar": avg_loss,
            "log": {"validataion/loss": avg_loss},
        }

        return results

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters())

    def reconstruct_images(self):
        x_org, _ = iter(self.loader).next()

    def prepare_data(self):
        """Download dataset"""
        datasets.MNIST(root=self.params["root"], train=True, download=True)
        datasets.MNIST(root=self.params["root"], train=False, download=True)

    @pl.data_loader
    def train_dataloader(self):
        # Dataset
        _transform = self.data_transform()
        _dataset = datasets.MNIST(root=self.params["root"], train=True,
                                  transform=_transform)

        # Params for data loader
        params = {"batch_size": self.params["batch_size"]}
        if self.params["cuda"]:
            params.update({"num_workers": 1, "pin_memory": True})

        return torch.utils.data.DataLoader(_dataset, shuffle=True, **params)

    @pl.data_loader
    def val_dataloader(self):
        # Dataset
        _transform = self.data_transform()
        _dataset = datasets.MNIST(root=self.params["root"], train=False,
                                  transform=_transform)

        # Params for data loader
        _params = {"batch_size": self.params["batch_size"]}
        if self.params["cuda"]:
            _params.update({"num_workers": 1, "pin_memory": True})

        # Instantiate for sampling image at reconstruction
        self.loader = torch.utils.data.DataLoader(
            _dataset, shuffle=False, **_params)

        return self.loader

    def data_transform(self):
        _transform = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
        ])

        return _transform
