
"""MNIST dataset"""

import torch.utils.data
from torchvision import datasets, transforms


def init_mnist_dataloader(root, batch_size, cuda=False):

    # Kwargs for data loader
    kwargs = {"num_workers": 1, "pin_memory": True} if cuda else {}

    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Lambda(lambd=lambda x: x.view(-1))])

    # Instantiate data loader
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=root, train=True, transform=transform,
                       download=True),
        shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=root, train=False, transform=transform),
        shuffle=True, **kwargs)

    return train_loader, test_loader
