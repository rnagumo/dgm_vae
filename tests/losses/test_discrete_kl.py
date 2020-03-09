
import unittest
import torch
import pixyz.distributions as pxd

import dgmvae.losses as dvl


class TestCategoricalKullbackLeibler(unittest.TestCase):

    def test_kl(self):
        p = pxd.Categorical(probs=torch.ones(2, dtype=torch.float32) / 2)
        q = pxd.RelaxedCategorical(
            probs=torch.ones(2, dtype=torch.float32) / 2)
        loss_cls = dvl.CategoricalKullbackLeibler(p, q)

        # Check symbol
        print(loss_cls)

        # Evaluate
        self.assertGreaterEqual(loss_cls.eval(), 0)
