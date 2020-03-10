
from setuptools import setup, find_packages


install_requires = [
    "torch==1.4.0",
    "torchvision==0.5.0",
    "pytorch-lightning==0.7.1",
    "pixyz==0.1.4",
    "disentanglement-lib==1.4",
    "numpy==1.18.1",
    "tensorflow-probability==0.7.0",
]

extras_require = {
    "tf": "tensorflow==1.14.0",
    "tf_gpu": "tensorflow-gpu==1.14.0",
}


setup(
    name="dgmvae",
    version="0.2",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require=extras_require,
)
