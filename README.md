
# dgmvae

Deep Generative Model: Variational Auto-Encoder

# Requirements

* Python == 3.7
* PyTorch == 1.4.0 ([Official](https://pytorch.org/))
* Torchvision == 0.5.0 ([GitHub](https://github.com/pytorch/vision))
* PyTorch Lightning == 0.7.1 ([GitHub](https://github.com/PyTorchLightning/pytorch-lightning))
* Pixyz == 0.1.4 ([GitHub](https://github.com/masa-su/pixyz))
* disentanglement_lib == 1.4 ([GitHub](https://github.com/google-research/disentanglement_lib))
* numpy == 1.18.1
* tensorflow == 1.14.0
* tensorflow-probability == 0.7.0
* scikit-learn == 0.22.2
* pandas == 1.0.1

# How to use

## Set up environments

Clone repository and make environments.

```bash
git clone https://github.com/rnagumo/dgmvae.git
cd dgmvae
```

Install package in virtual env.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

Or run docker file. This Dockerfile creates a very large docker image (>4GB), so please check the upper limit of the docker memory size on your computer.

```bash
# Option
docker build -t dgmvae .
docker run -it dgmvae bash
```

## Prepare dataset

Download dataset ([dSprites](https://github.com/deepmind/dsprites-dataset/), [mpi3d_toy](https://github.com/rr-learning/disentanglement_dataset), or [cars3d](http://www.scottreed.info/)).

```bash
bash bin/download.sh
```

## Run experiment

Training and evaluation. Shell scripts in `bin` folder contains the necessary settings for building the environment.

```bash
bash bin/run_cars3d.sh
```

Evaluation with original metrics (run on CPU).

```bash
bash bin/eval_cars3d.sh
```

# Experimental Results

VAE model included in disentanglement_lib are trained with the same setteings. Pre-trained models (TF models) are downloaded from disentanglement_lib. Although pre-trained models are tested with 50 different random seeds, my experiments are tested with 10 or 5 random seeds due to the lack of GPUs.

|Model|Ours|disentanglement_lib|
|:-:|:-:|:-:|
|BetaVAE|![beta_vae](./images/betavae.png)|![beta_vae_tf](./images/betavae_tf.png)|
|FactorVAE|![factor_vae](./images/factorvae.png)|![factor_vae_tf](./images/factorvae_tf.png)|
|DIPVAE-1|![dip_vae1](./images/dipvae1.png)|![dip_vae1_tf](./images/dipvae1_tf.png)|
|DIPVAE-2|![dip_vae2](./images/dipvae2.png)|![dip_vae2_tf](./images/dipvae2_tf.png)|
|TC-VAE|![tc_vae](./images/tcvae.png)|![tc_vae_tf](./images/tcvae_tf.png)|

Other models not in disentanglement_lib are also trained and evaluated.

|Model|Quantitative evaluation|
|:-:|:-:|
|JointVAE|![joint_vae](./images/jointvae.png)|
|AAE|![aae](./images/aae.png)|
|AVB|![avb](./images/avb.png)|

I tested original codes for disentanglement metrics. The following figure shows the comparison of (tf-model, disentanglement_lib metrics), (pixyz-model, disentanglement_lib metrics), and (pixyz-model, original metrics). This figure shows that my implementation seems correct.

|Model|Original metrics|disentanglement_lib|
|:-:|:-:|:-:|
|BetaVAE|![metrics_org](./images/metrics_org.png)|![metrics_dlib](./images/metrics_dlib.png)|

# Reference

* Preferred Networks: [Disentangled な表現の教師なし学習手法の検証](https://tech.preferred.jp/ja/blog/disentangled-represetation/)
* PyTorch VAE by AntixK: [GitHub](https://github.com/AntixK/PyTorch-VAE)
* NeurIPS 2019 : Disentanglement Challenge Starter Kit: [GitHub](https://github.com/AIcrowd/neurips2019_disentanglement_challenge_starter_kit)
