
from setuptools import setup, find_packages


install_requires = [
    "torch>=1.0",
    "torchvision",
    "pixyz>=0.1.3",
]

setup(
    name="dgmvae",
    version="0.1",
    packages=find_packages(),
    install_requires=install_requires,
)
