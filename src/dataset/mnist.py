
"""MNIST dataset"""

import torch.utils.data
from torchvision import datasets, transforms


def init_mnist_dataloader(root, batch_size, cuda=False):

    # Kwargs for data loader
    kwargs = {"num_workers": 1, "pin_memory": True} if cuda else {}

    # Instantiate data loader
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=root, train=True, transform=transforms.ToTensor(),
                       download=True),
        shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=root, train=False,
                       transform=transforms.ToTensor()),
        shuffle=False, **kwargs)

    return train_loader, test_loader
