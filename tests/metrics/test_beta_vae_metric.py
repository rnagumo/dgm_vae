

import unittest

import dgmvae.metrics as dgm
from .shared import TestMetricBase


class TestBetavaeMetric(TestMetricBase):

    def test_metric(self):
        scores = dgm.beta_vae_metric(self.dataset, self.repr_fn,
                                     num_train=100, num_eval=100)
        self.assertLessEqual(scores["train_accuracy"], 1.0)
        self.assertLessEqual(scores["eval_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
