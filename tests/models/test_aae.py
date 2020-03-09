
import unittest

import torch

import dgmvae.models as dgm


class TestAAE(unittest.TestCase):

    def setUp(self):
        self.batch_n = 5
        self.z_dim = 2
        self.c_dim = 2

        params = {
            "channel_num": 1,
            "z_dim": self.z_dim,
            "c_dim": self.c_dim,
            "beta": 1
        }
        self.model = dgm.AAE(**params)

    def test_encode(self):
        x = torch.randn(self.batch_n, 1, 64, 64)

        latent = self.model.encode(x)
        self.assertIsInstance(latent, dict)
        self.assertEqual(
            latent["z"].size(), torch.Size([self.batch_n, self.z_dim]))
        self.assertEqual(
            latent["c"].size(), torch.Size([self.batch_n, self.c_dim]))

        z, c = self.model.encode(x, mean=True)
        self.assertEqual(z.size(), torch.Size([self.batch_n, self.z_dim]))
        self.assertEqual(c.size(), torch.Size([self.batch_n, self.c_dim]))

    def test_decode(self):
        z = torch.randn(self.batch_n, self.z_dim)
        c = torch.randn(self.batch_n, self.c_dim)
        latent = {"z": z, "c": c}

        obs = self.model.decode(latent)
        self.assertIsInstance(obs, dict)
        self.assertEqual(
            obs["x"].size(), torch.Size([self.batch_n, 1, 64, 64]))

        obs = self.model.decode(latent, mean=True)
        self.assertIsInstance(obs, torch.Tensor)
        self.assertEqual(obs.size(), torch.Size([self.batch_n, 1, 64, 64]))

        # Input without dict
        obs = self.model.decode(None, z=z, c=c)
        self.assertIsInstance(obs, dict)
        self.assertEqual(
            obs["x"].size(), torch.Size([self.batch_n, 1, 64, 64]))

    @unittest.expectedFailure
    def test_decode_with_none(self):
        # Wrong input
        z = torch.randn(self.batch_n, self.z_dim)
        c = torch.randn(self.batch_n, self.c_dim)
        _ = self.model.decode(z=z, c=c)

    def test_sample(self):
        batch_n = 2
        obs = self.model.sample(batch_n=batch_n)

        self.assertIsInstance(obs, torch.Tensor)
        self.assertEqual(obs.size(), torch.Size([batch_n, 1, 64, 64]))

    def test_forward(self):
        x = torch.randn(self.batch_n, 1, 64, 64)
        z, c = self.model(x)
        self.assertEqual(z.size(), torch.Size([self.batch_n, self.z_dim]))
        self.assertEqual(c.size(), torch.Size([self.batch_n, self.c_dim]))

    def test_reconstruct(self):
        x = torch.randn(self.batch_n, 1, 64, 64)

        # 1. reconstruct without latent
        obs = self.model.reconstruct(x)
        self.assertIsInstance(obs, torch.Tensor)
        self.assertEqual(obs.size(), torch.Size([self.batch_n, 1, 64, 64]))

        # 2. reconstruct with latent
        sample = self.model.reconstruct(x, return_latent=True)
        self.assertIsInstance(sample, dict)
        self.assertEqual(
            sample["x"].size(), torch.Size([self.batch_n, 1, 64, 64]))
        self.assertEqual(
            sample["z"].size(), torch.Size([self.batch_n, self.z_dim]))
        self.assertEqual(
            sample["c"].size(), torch.Size([self.batch_n, self.c_dim]))

    def test_loss_func(self):
        x = torch.randn(self.batch_n, 1, 64, 64)

        loss_dict = self.model.loss_func(x, optimizer_idx=0)
        self.assertGreaterEqual(loss_dict["loss"], 0)
        self.assertGreaterEqual(loss_dict["ce_loss"], 0)
        self.assertGreaterEqual(loss_dict["js_loss"], 0)

        adv_dict = self.model.loss_func(x, optimizer_idx=1)
        self.assertGreaterEqual(adv_dict["adv_loss"], 0)

    def test_loss_str(self):
        self.assertIsInstance(self.model.loss_str, str)

    def test_second_optim(self):
        self.assertIsInstance(self.model.second_optim, torch.optim.Adam)


if __name__ == "__main__":
    unittest.main()
