

import dgmvae.metrics as dgm
from .shared import TestMetricBase


class TestFactorVaeMetric(TestMetricBase):

    def test_metric(self):
        scores = dgm.factor_vae_metric(
            self.dataset, self.repr_fn, num_train=100, num_eval=100,
            num_var=100, th_var=0.)
        self.assertTrue(0 < scores["train_accuracy"] <= 1.0)
        self.assertTrue(0 < scores["eval_accuracy"] <= 1.0)
        self.assertGreaterEqual(scores["num_active_dims"], 0)
