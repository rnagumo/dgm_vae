
import unittest

import dgmvae.models as dgm


class TestBaseVAE(unittest.TestCase):

    def setUp(self):
        self.model = dgm.BaseVAE()

    def test_init(self):
        self.assertFalse(self.model.distributions)


if __name__ == "__main__":
    unittest.main()
