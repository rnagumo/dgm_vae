
import unittest
import pathlib

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
