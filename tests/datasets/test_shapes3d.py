
import unittest
import pathlib

import dgmvae.datasets as dvd
from .base_dataset_case import BaseDatasetTestCase


class TestShapes3D(unittest.TestCase, BaseDatasetTestCase):

    def setUp(self):
        # Need pre-downloaded dataset
        path = pathlib.Path(__file__).parent.parent.parent
        self.dataset = dvd.Shapes3D(path.joinpath("data/3dshapes"))

        self.channel = 3
        self.latents = 6
        self.all_factors = 480000
