
import unittest
import pathlib

import dgmvae.datasets as dvd
from .base_dataset_case import BaseDatasetTestCase


class TestCars3dDataset(unittest.TestCase, BaseDatasetTestCase):

    def setUp(self):
        # Need pre-downloaded dataset
        path = pathlib.Path(__file__).parent.parent.parent
        self.dataset = dvd.Cars3dDataset(path.joinpath("data/cars/"))

        self.channel = 3
        self.latents = 3
        self.all_factors = 24 * 4 * 183
