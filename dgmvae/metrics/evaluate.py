
"""Evaluation method.

ref)
https://github.com/AIcrowd/neurips2019_disentanglement_challenge_starter_kit/blob/master/evaluate.py
"""

import torch
from torchvision import datasets, transforms

from ..datasets.cars3d import Cars3dDataset
from ..datasets.dsprites import DSpritesDataset

from ..metrics.beta_vae_metric import beta_vae_metric
from ..metrics.factor_vae_metric import factor_vae_metric
from ..metrics.irs import irs
from ..metrics.mig import mig
from ..metrics.sap_score import sap_score
from ..metrics.dci import dci


class MetricsEvaluator:

    def __init__(self, config=None):
        self.dataset = None
        self.model = None
        self.config = config

        self.metric_dict = {
            "beta_vae_metric": beta_vae_metric,
            "factor_vae_metric": factor_vae_metric,
            "irs": irs,
            "mig": mig,
            "sap_score": sap_score,
            "dci": dci,
        }

    def load_dataset(self, dataset_name, root):
        """Loads dataset of specified name."""

        if dataset_name == "mnist":
            _transform = transforms.Compose([
                transforms.Resize(64), transforms.ToTensor()])
            self.dataset = datasets.MNIST(root=root, train=True,
                                          transform=_transform)
        elif dataset_name == "dsprites_full":
            self.dataset = DSpritesDataset(root=root)
        elif dataset_name == "cars3d":
            self.dataset = Cars3dDataset(root=root)
        else:
            raise KeyError(f"Unexpected dataset is specified: {dataset_name}")

    def load_model(self, path):
        """Loads pre-trained model.

        Args:
            path (str or pathlib.Path): Path to dataset.
        """

        # Load a model (as torch.jit.ScriptModule or torch.nn.Module)
        try:
            model = torch.jit.load(str(path))
        except RuntimeError:
            try:
                model = torch.load(str(path))
            except Exception as e:
                raise IOError("Could not load file") from e

        self.model = model.cpu()

    def repr_fn(self, x):
        """Representation function that takes observation as input and outputs
        a representation.

        Args:
            x (torch.tensor): Observations tensor.

        Returns:
            reprs (torch.tensor): Representations tensor.
        """
        with torch.no_grad():
            return self.model(x)

    def compute_metric(self, metric_name):
        """Computes metric.

        Args:
            metric_name (str): Metric name in metric_dict.
        """
        if self.dataset is None or self.model is None:
            raise ValueError("Load dataset and model before computing metric")

        if (self.config is not None) and (metric_name in self.config):
            kwargs = self.config[metric_name]
        else:
            kwargs = {}

        return self.metric_dict[metric_name](
                   self.dataset, self.repr_fn, **kwargs)
