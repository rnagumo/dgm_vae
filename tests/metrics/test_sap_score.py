

import dgmvae.metrics as dgm
from .shared import TestMetricBase


class TestSapScoreMetric(TestMetricBase):

    def test_metric(self):
        scores = dgm.sap_score(self.dataset, self.repr_fn, num_points=100)
        self.assertTrue(0.0 <= scores["SAP_score"] <= 1.0)
