
import unittest
import torch

from dgmvae.datasets.base_data import BaseDataset


class TestMetricBase(unittest.TestCase):

    def setUp(self):
        factor_sizes = [5] * 5
        batch_size = torch.prod(torch.tensor(factor_sizes))

        # Dataset
        self.dataset = BaseDataset()
        self.dataset.data = torch.rand(batch_size, 1, 64, 64)
        self.dataset.targets = torch.cartesian_prod(
            *[torch.arange(v) for v in factor_sizes])
        self.dataset.factor_sizes = factor_sizes

        # Representation fn
        self.repr_fn = lambda x: torch.ones(x.size(0), 10) * x.mean()
