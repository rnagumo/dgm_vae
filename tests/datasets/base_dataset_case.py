
"""Base test case for dataset class"""

import torch


class BaseDatasetTestCase:

    def test_dataset(self):
        x, y = self.dataset[0]
        self.assertTupleEqual(x.size(), (self.channel, 64, 64))
        self.assertTupleEqual(y.size(), (self.latents,))

        data_len = len(self.dataset)
        self.assertEqual(data_len, self.all_factors)

        # Test sample_batch
        batch_data, batch_targets = self.dataset.sample_fixed_batch(32)
        self.assertTupleEqual(batch_data.size(), (32, self.channel, 64, 64))
        self.assertTupleEqual(batch_targets.size(), (32, self.latents))

        # Test sample_fixed_batch
        factor_index = self.dataset.sample_factor_index()
        batch_data, batch_targets = self.dataset.sample_fixed_batch(
            32, factor_index)
        self.assertTupleEqual(batch_data.size(), (32, self.channel, 64, 64))
        self.assertTupleEqual(batch_targets.size(), (32, self.latents))

        # Check that only one column in targets has the same value
        for i in range(self.latents):
            tmp = batch_targets[:, i].float()
            self.assertFalse(
                (i == factor_index) ^ torch.all(tmp == tmp.mean()))

        # Test sample_paired_batch
        data1, data2, targets1, targets2 = self.dataset.sample_paired_batch(
            32, factor_index)
        self.assertTupleEqual(data1.size(), (32, self.channel, 64, 64))
        self.assertTupleEqual(data2.size(), (32, self.channel, 64, 64))
        self.assertTupleEqual(targets1.size(), (32, self.latents))
        self.assertTupleEqual(targets2.size(), (32, self.latents))

        for i in range(self.latents):
            flg = torch.all(targets1[:, i] == targets2[:, i])
            self.assertFalse((i == factor_index) ^ flg)
