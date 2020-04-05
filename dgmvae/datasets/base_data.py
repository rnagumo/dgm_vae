
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

    def sample_factors(self, num, random_state):
        """Sample a batch of factors Y."""
        raise NotImplementedError

    def sample_observations_from_factors(self, factors, random_state):
        """Sample a batch of observations X given a batch of factors Y."""
        raise NotImplementedError

    def sample(self, num, random_state):
        """Sample a batch of factors Y and observations X."""
        factors = self.sample_factors(num, random_state)
        obs = self.sample_observations_from_factors(factors, random_state)
        return factors, obs

    def sample_observations(self, num, random_state):
        """Sample a batch of observations X."""
        return self.sample(num, random_state)[1]
