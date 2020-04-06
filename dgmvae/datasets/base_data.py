
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

    def sample_fixed_batch(self, batch_size):
        """Samples batch observations and factors with fixed coordinate."""

        # Select random coordinate to keep fixed
        factor_index = torch.randint(len(self.factor_sizes), (1,))

        # Fixed factor value
        factor_value = torch.randint(self.factor_sizes[factor_index], (1,))

        # Selected dataset
        mask = (self.targets[:, factor_index] == factor_value).squeeze(1)
        tmp_data = self.data[mask]
        tmp_targets = self.targets[mask]

        # Sample batch
        batch_index = torch.randint(tmp_targets.size(0), (batch_size,))
        return tmp_data[batch_index], tmp_targets[batch_index]
