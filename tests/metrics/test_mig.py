

import dgmvae.metrics as dgm
from .shared import TestMetricBase


class TestMIG(TestMetricBase):

    def test_metric(self):
        scores = dgm.mig(self.dataset, self.repr_fn, num_train=100)
        self.assertTrue(0 <= scores["discrete_mig"] <= 1)
