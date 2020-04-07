
import numpy as np

import dgmvae.metrics as dgm
from .shared import TestMetricBase


class TestUtilFuncs(TestMetricBase):

    def test_generate_repr_factor_batch(self):
        reprs, targets = dgm.generate_repr_factor_batch(
            self.dataset, self.repr_fn, batch_size=64, num_points=100)

        self.assertTupleEqual(reprs.shape, (100, 10))
        self.assertTupleEqual(targets.shape, (100, 5))

    def test_discretize_target(self):
        target = np.vstack([np.arange(10)] * 4)
        discretized = dgm.discretize_target(target, 10)

        self.assertTupleEqual(discretized.shape, (4, 10))
        self.assertTrue(all(target[0] + 1 == discretized[0]))
