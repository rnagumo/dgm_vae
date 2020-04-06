
import unittest
import pathlib

import torch

import dgmvae.datasets as dvd


class TestCars3dDataset(unittest.TestCase):

    def test_dataset(self):
        # Need pre-downloaded dataset
        path = pathlib.Path(__file__).parent.parent.parent
        dataset = dvd.Cars3dDataset(path.joinpath("data/cars/"))

        x, y = dataset[0]
        self.assertTupleEqual(x.size(), (3, 64, 64))
        self.assertTupleEqual(y.size(), (3,))

        data_len = len(dataset)
        self.assertEqual(data_len, 24 * 4 * 183)

        # Test sample_fixed_batch
        batch_data, batch_targets = dataset.sample_fixed_batch(32)
        self.assertTupleEqual(batch_data.size(), (32, 3, 64, 64))
        self.assertTupleEqual(batch_targets.size(), (32, 3))

        # Check that only one column in targets has the same value
        cnt = 0
        for i in range(3):
            tmp = batch_targets[:, i].float()
            cnt += int(torch.all(tmp == tmp.mean()))
        self.assertEqual(cnt, 1)
