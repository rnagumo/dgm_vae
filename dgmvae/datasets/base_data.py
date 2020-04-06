
"""Dataset class for disentanglement score

BaseDataset
https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/data/ground_truth/ground_truth_data.py

SplitDiscreteStateSpace, StateSpaceAtomIndex
https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/data/ground_truth/util.py
"""

import torch


class BaseDataset(torch.utils.data.Dataset):
    """Base Dataset class for disentanglement scoreing."""

    @property
    def num_factors(self):
        raise NotImplementedError

    @property
    def factors_num_values(self):
        raise NotImplementedError

    @property
    def observation_shape(self):
        raise NotImplementedError

    def sample_factors(self, num):
        """Sample a batch of factors Y."""
        raise NotImplementedError

    def sample_observations_from_factors(self, factors):
        """Sample a batch of observations X given a batch of factors Y."""
        raise NotImplementedError

    def sample(self, num):
        """Sample a batch of factors Y and observations X."""
        factors = self.sample_factors(num)
        obs = self.sample_observations_from_factors(factors)
        return factors, obs

    def sample_observations(self, num):
        """Sample a batch of observations X."""
        return self.sample(num)[1]
