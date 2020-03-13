
import unittest
import torch
import pixyz.distributions as pxd

import dgmvae.losses as dvl


class TestDipLoss(unittest.TestCase):

    def test_diploss_i(self):
        p = pxd.Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
                       features_shape=[2])
        loss_cls = dvl.DipLoss(p, 10, 10, dip_type="i")

        # Check symbol
        print(loss_cls)

        # Evaluate
        self.assertGreaterEqual(loss_cls.eval(), 0)

    def test_diploss_ii(self):
        p = pxd.Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
                       features_shape=[2])
        loss_cls = dvl.DipLoss(p, 10, 10, dip_type="ii")

        # Check symbol
        print(loss_cls)

        # Evaluate
        self.assertGreaterEqual(loss_cls.eval(), 0)
