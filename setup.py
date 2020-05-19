
from setuptools import setup, find_packages


install_requires = [
    "torch==1.4.0",
    "torchvision==0.5.0",
    "pixyz==0.1.4",
    "numpy==1.18.1",
    "scikit-learn==0.22.2",
]


setup(
    name="dgmvae",
    version="0.5",
    description="VAE model packages for disentanglement experiment",
    packages=find_packages(),
    install_requires=install_requires,
)
