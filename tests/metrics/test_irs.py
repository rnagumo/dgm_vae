
import unittest

import dgmvae.metrics as dgm
from .shared import TestMetricBase


class TestIRSMetric(TestMetricBase):

    def test_metric(self):
        scores = dgm.irs(self.dataset, self.repr_fn, num_points=100)
        self.assertTrue(0.0 < scores["irs"] < 1.0)
        self.assertGreaterEqual(scores["num_active_dims"], 0)


if __name__ == "__main__":
    unittest.main()
