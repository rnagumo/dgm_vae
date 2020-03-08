
from setuptools import setup, find_packages


install_requires = [
    "pytorch-lightning>=0.7.0",
    "torch>=1.4.0",
    "torchvision>=0.5.0",
    "pixyz>=0.1.4",
]

setup(
    name="dgmvae",
    version="0.2",
    packages=find_packages(),
    install_requires=install_requires,
)
