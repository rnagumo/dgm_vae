
"""Dataset class for disentanglement score

BaseDataset
https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/data/ground_truth/ground_truth_data.py
"""

import torch


class BaseDataset(torch.utils.data.Dataset):
    """Base Dataset class for disentanglement scoreing."""
