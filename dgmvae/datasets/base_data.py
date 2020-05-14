
"""Dataset class for disentanglement score

BaseDataset
https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/data/ground_truth/ground_truth_data.py

sample fixed batch
https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/evaluation/metrics/factor_vae.py#L137
"""

from typing import Tuple, Optional

import torch
from torch import Tensor


class BaseDataset(torch.utils.data.Dataset):
    """Base Dataset class for disentanglement scoreing.

    Attrutes:
        data (torch.Tensor): Dataset.
        targets (torch.Tensor): Target labels.
        factor_sizes (list): List of size of each factor.
    """

    def __init__(self):
        super().__init__()

        self.data = None
        self.targets = None
        self.factor_sizes = []

    @property
    def num_factors(self):
        return len(self.factor_sizes)

    def sample_batch(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        """Samples batch data.

        Args:
            batch_size (int): Batch size.

        Returns:
            batch_data (torch.Tensor): Sampled batch data.
            batch_targets (torch.Tensor): Sampled targets.
        """

        # Sample data with replacement
        batch_index = torch.randint(self.targets.size(0), (batch_size,))
        batch_data = self.data[batch_index].float()
        batch_targets = self.targets[batch_index]

        # Expand channel dim
        if batch_data.dim() == 3:
            batch_data = batch_data.unsqueeze(1)

        return batch_data, batch_targets

    def sample_factor_index(self) -> int:
        """Samples fixed factor column.

        Returns:
            factor_index (int): Index number of selected factor.
        """

        return torch.randint(len(self.factor_sizes), (1,)).item()

    def sample_fixed_batch(self, batch_size: int,
                           factor_index: Optional[int] = None
                           ) -> Tuple[Tensor, Tensor]:
        """Samples batch observations and factors with fixed coordinate,
        mainly used for calculating Factor-VAE metrics.

        Args:
            batch_size (int): Batch size.
            factor_index (int, optional): Number of fixed factor index.

        Returns:
            batch_data (torch.Tensor): Sampled batch data.
            batch_targets (torch.Tensor): Sampled targets.
        """

        if factor_index is None:
            factor_index = self.sample_factor_index()

        # Fixed factor value
        factor_value = torch.randint(self.factor_sizes[factor_index], (1,))

        # Selected dataset
        mask = self.targets[:, factor_index] == factor_value
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

    def sample_paired_batch(self, batch_size: int,
                            factor_index: Optional[int] = None
                            ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Samples paired batch observations and factors with fixed coordinate,
        mainly used for calculating beta-VAE metrics.

        Args:
            batch_size (int): Batch size.
            factor_index (int, optional): Number of fixed factor index.

        Returns:
            data0 (torch.Tensor): Sampled data.
            data1 (torch.Tensor): Sampled data.
            targets0 (torch.Tensor): Sampled targets.
            targets1 (torch.Tensor): Sampled targets.
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

        # Samples of size (batch_size, 2, c, w, h), (batch_size, 2, latents)
        data = torch.stack(data)
        targets = torch.stack(targets)

        return data[:, 0], data[:, 1], targets[:, 0], targets[:, 1]
