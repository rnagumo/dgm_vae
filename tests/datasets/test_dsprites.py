
import unittest
import pathlib

import dgmvae.datasets as dvd


class TestDSpritesDataset(unittest.TestCase):

    def test_dataset(self):
        # Need pre-downloaded dataset
        path = pathlib.Path(__file__).parent.parent.parent
        dataset = dvd.DSpritesDataset(path.joinpath("data/dsprites"))

        x, y = dataset[0]
        self.assertTupleEqual(x.size(), (1, 64, 64))
        self.assertTupleEqual(y.size(), (6,))

        data_len = len(dataset)
        self.assertEqual(data_len, 737280)
