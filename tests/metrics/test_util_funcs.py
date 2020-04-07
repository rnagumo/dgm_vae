

import dgmvae.metrics as dgm
from .shared import TestMetricBase


class TestUtilFuncs(TestMetricBase):

    def test_generate_repr_factor_batch(self):
        reprs, targets = dgm.generate_repr_factor_batch(
            self.dataset, self.repr_fn, batch_size=64, num_points=100)

        self.assertTupleEqual(reprs.shape, (100, 10))
        self.assertTupleEqual(targets.shape, (100, 5))
