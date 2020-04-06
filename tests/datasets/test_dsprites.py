
import unittest
import pathlib

import dgmvae.datasets as dvd
from .base_case import BaseTestCase


class TestDSpritesDataset(unittest.TestCase, BaseTestCase):

    def setUp(self):
        # Need pre-downloaded dataset
        path = pathlib.Path(__file__).parent.parent.parent
        self.dataset = dvd.DSpritesDataset(path.joinpath("data/dsprites"))

        self.channel = 1
        self.latents = 5
        self.all_factors = 737280
