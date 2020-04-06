
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


class SplitDiscreteStateSpace:
    def __init__(self, factor_sizes, latent_factor_indices):
        self.factor_sizes = factor_sizes
        self.num_factors = len(factor_sizes)
        self.latent_factor_indices = latent_factor_indices
        self.observation_factor_indices = [
            i for i in range(self.num_factors)
            if i not in self.latent_factor_indices
        ]

    @property
    def num_latent_factors(self):
        return len(self.latent_factor_indices)

    def sample_latnet_factors(self, num):
        """Samples a batch of the latent factors."""
        factors = torch.zeros(num, self.num_factors)
        for pos, i in enumerate(self.latent_factor_indices):
            factors[:, pos] = self._sample_factor(i, num)
        return factors

    def sample_all_factors(self, latent_factors):
        """Samples the remaining factors based on the latent factors."""
        num_samples = latent_factors.shape[0]
        all_factors = torch.zeros(num_samples, self.num_factors)
        all_factors[:, self.latent_factor_indices] = latent_factors

        # Complete all other factors
        for i in self.observation_factor_indices:
            all_factors[:, i] = self._sample_factor(i, num_samples)
        return all_factors

    def _sample_factor(self, i, num):
        return torch.randint(self.factor_sizes[i], size=(num,))


class StateSpaceAtomIndex:
    """Index mapping from features to positions of state space atoms."""

    def __init__(self, factor_sizes, features):
        num_total_atoms = torch.prod(factor_sizes)

        self.factor_sizes = factor_sizes
        self.factor_bases = num_total_atoms / torch.cumprod(factor_sizes, 0)

        ftr_ss_index = self._features_to_state_space_index(features)
        if torch.unique(ftr_ss_index).size(0) != num_total_atoms:
            raise ValueError("Features matrix does not cover the whole "
                             "state space, {} != {}".format(
                                 torch.unique(ftr_ss_index).size(0),
                                 num_total_atoms
                             ))

        lookup_table = torch.zeros(num_total_atoms)
        lookup_table[ftr_ss_index] = torch.arange(num_total_atoms).float()
        self.state_space_to_save_space_index = lookup_table

    def features_to_index(self, features):
        state_space_index = self._features_to_state_space_index(features)
        return self.state_space_to_save_space_index[state_space_index]

    def _features_to_state_space_index(self, features):
        """Returns the indices in the atom space for given factors
        configurations.
        """
        if (torch.any(features > self.factor_sizes.unsqueeze(0))
                or torch.any(features < 0)):
            raise ValueError("Feature indices have to be within "
                             "(0, factor_size-1)")
        return torch.matmul(features, self.factor_bases)
