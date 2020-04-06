
"""Dataset class for disentanglement score

BaseDataset
https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/data/ground_truth/ground_truth_data.py

sample fixed batch
https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/evaluation/metrics/factor_vae.py#L137
"""

import torch


class BaseDataset(torch.utils.data.Dataset):
    """Base Dataset class for disentanglement scoreing."""

    def __init__(self):
        super().__init__()

        self.data = None
        self.targets = None
        self.factor_sizes = []

    def sample_batch(self, batch_size):
        # Sample data with replacement
        batch_index = torch.randint(self.targets.size(0), (batch_size,))
        batch_data = self.data[batch_index].float()
        batch_targets = self.targets[batch_index]

        # Expand channel dim
        if batch_data.dim() == 3:
            batch_data = batch_data.unsqueeze(1)

        return batch_data, batch_targets

    def sample_factor_index(self):
        """Samples fixed factor column."""
        return torch.randint(len(self.factor_sizes), (1,))

    def sample_fixed_batch(self, batch_size, factor_index=None):
        """Samples batch observations and factors with fixed coordinate,
        mainly used for calculating Factor-VAE metrics.
        """

        if factor_index is None:
            factor_index = self.sample_factor_index()

        # Fixed factor value
        factor_value = torch.randint(self.factor_sizes[factor_index], (1,))

        # Selected dataset
        mask = (self.targets[:, factor_index] == factor_value).squeeze(1)
        tmp_data = self.data[mask]
        tmp_targets = self.targets[mask]

        # Sample batch
        batch_index = torch.randint(tmp_targets.size(0), (batch_size,))
        batch_data = tmp_data[batch_index].float()
        batch_targets = tmp_targets[batch_index]

        # Expand channel dim
        if batch_data.dim() == 3:
            batch_data = batch_data.unsqueeze(1)

        return batch_data, batch_targets

    def sample_paired_batch(self, batch_size, factor_index=None):
        """Samples paired batch observations and factors with fixed coordinate,
        mainly used for calculating beta-VAE metrics.
        """

        if factor_index is None:
            factor_index = self.sample_factor_index()

        data = []
        targets = []
        for _ in range(batch_size):
            # Sample paired observations which share the same factor at only
            # one column
            _data, _label = self.sample_fixed_batch(2, factor_index)
            data.append(_data)
            targets.append(_label)

        # Sample size of (batch_size, 2, c, w, h), (batch_size, 2, latents)
        data = torch.stack(data)
        targets = torch.stack(targets)

        return data[:, 0], data[:, 1], targets[:, 0], targets[:, 1]
