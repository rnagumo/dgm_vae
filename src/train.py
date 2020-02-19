
"""Training method"""

import argparse

import torch
from torch.utils import tensorboard

from dataset.mnist import init_mnist_dataloader
from utils.utils import init_logger, load_config, check_logdir
from model.base_vae import BaseVAE


def train(args, logger, config):

    # -------------------------------------------------------------------------
    # 1. Settings
    # -------------------------------------------------------------------------

    # Cuda setting
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f"Device: {device}")

    # Random seed
    torch.manual_seed(args.seed)

    # Tensorboard writer
    writer = tensorboard.SummaryWriter(args.logdir)

    # -------------------------------------------------------------------------
    # 2. Data
    # -------------------------------------------------------------------------

    # Loader
    train_loader, test_loader = init_mnist_dataloader(
        args.root, args.batch_size, use_cuda)

    # Data dimension, (batch_size, channel_num, height, width)
    channel_num = iter(train_loader).next()[0].size(1)

    # Sample data for comparison
    x_org, _ = iter(test_loader).next()

    # Log
    logger.info(f"Train data size: {train_loader.dataset.data.size()}")
    logger.info(f"Test data size: {test_loader.dataset.data.size()}")

    # -------------------------------------------------------------------------
    # 3. Model
    # -------------------------------------------------------------------------

    params = {"channel_num": channel_num, "device": device}
    model = BaseVAE(**config["vae_params"], **params)

    # -------------------------------------------------------------------------
    # 4. Training
    # -------------------------------------------------------------------------

    for epoch in range(1, args.epochs + 1):
        logger.info(f"--- Epoch {epoch} ---")

        # Training
        train_loss = model.run(train_loader, training=True)
        test_loss = model.run(test_loader, training=False)

        # Reconstruction data
        recon = model.reconstruction(x_org)

        # Sample data
        sample = model.sample(8)

        # Log
        writer.add_scalar("loss/train_loss", train_loss["loss"], epoch)
        writer.add_scalar("loss/test_loss", test_loss["loss"], epoch)
        writer.add_scalar("training/cross_entropy",
                          train_loss["ce_loss"], epoch)
        writer.add_scalar("training/kl_divergence",
                          train_loss["kl_loss"], epoch)
        writer.add_images("image_reconstruction", recon, epoch)
        writer.add_images("image_from_latent", sample, epoch)

        logger.info(f"Train loss = {train_loss['loss']}")
        logger.info(f"Test loss = {test_loss['loss']}")

    # -------------------------------------------------------------------------
    # 5. Summary
    # -------------------------------------------------------------------------

    # Log hyper-parameters
    hparam_dict = vars(args)
    hparam_dict.update(config[f"{args.model}_params"])
    metric_dict = {"train_loss": train_loss["loss"],
                   "test_loss": test_loss["loss"]}
    writer.add_hparams(hparam_dict, metric_dict)

    writer.close()


def init_args():
    parser = argparse.ArgumentParser(description="VAE training")
    parser.add_argument("--logdir", type=str, default="../logs/vae/tmp/")
    parser.add_argument("--root", type=str, default="../data/mnist/")
    parser.add_argument("--config", type=str, default="./config.json")
    parser.add_argument("--model", type=str, default="vae")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)

    return parser.parse_args()


def main():
    # Args
    args = init_args()

    # Make logdir
    check_logdir(args.logdir)

    # Logger
    logger = init_logger(args.logdir)
    logger.info("Start logger")
    logger.info(f"Commant line args: {args}")

    # Config
    config = load_config(args.config)
    logger.info(f"Configs: {config}")

    try:
        train(args, logger, config)
    except Exception as e:
        logger.exception(f"Run function error: {e}")

    logger.info("End logger")


if __name__ == "__main__":
    main()
