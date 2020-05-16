
import unittest

import torch

import dgmvae.metrics as dgm
from dgmvae.datasets.base_data import BaseDataset


class TestMetricsEvaluator(unittest.TestCase):

    def setUp(self):
        self.evaluator = dgm.MetricsEvaluator()

    def test_compute_metric(self):
        # Dataset
        factor_sizes = [5] * 5
        batch_size = torch.prod(torch.tensor(factor_sizes))

        dataset = BaseDataset()
        dataset.data = torch.rand(batch_size, 1, 64, 64)
        dataset.targets = torch.cartesian_prod(
            *[torch.arange(v) for v in factor_sizes])
        dataset.factor_sizes = factor_sizes
        self.evaluator.dataset = dataset

        # Model
        self.evaluator.model = TmpModel()

        # Compute metric
        scores_dict = self.evaluator.compute_metric("mig")

        self.assertIsInstance(scores_dict, dict)
        self.assertTrue(0 <= scores_dict["discrete_mig"] <= 1)


class TmpModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(64 * 64, 10)

    def forward(self, x):
        x = x.view(-1, 64 * 64)
        return self.fc(x)


if __name__ == "__main__":
    unittest.main()
