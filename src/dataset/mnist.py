
"""MNIST dataset"""

import torch.utils.data
from torchvision import datasets, transforms


def init_mnist_dataloader(root, batch_size, cuda=False):

    # Kwargs for data loader
    kwargs = {"batch_size": batch_size}
    if cuda:
        kwargs.update({"num_workers": 1, "pin_memory": True})

    # Transform
    transform = transforms.Compose(
        [transforms.Resize(64), transforms.ToTensor()])

    # Instantiate data loader
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=root, train=True, transform=transform,
                       download=True),
        shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=root, train=False, transform=transform),
        shuffle=False, **kwargs)

    return train_loader, test_loader
