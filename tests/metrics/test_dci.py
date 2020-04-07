

import dgmvae.metrics as dgm
from .shared import TestMetricBase


class TestDciMetric(TestMetricBase):

    def test_metric(self):
        scores = dgm.dci(self.dataset, self.repr_fn, num_train=100,
                         num_test=100)

        self.assertTrue(0.0 <= scores["informativeness_train"] <= 1.0)
        self.assertTrue(0.0 <= scores["informativeness_test"] <= 1.0)
        self.assertTrue(0.0 <= scores["disentanglement"] <= 1.0)
        self.assertTrue(0.0 <= scores["completeness"] <= 1.0)
