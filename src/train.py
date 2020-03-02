
"""Training method"""

import argparse
import datetime
import pathlib

import torch
import tensorboardX as tb

import dgmvae.dataset as dvd
import dgmvae.model as dvm
import dgmvae.utils as dvu


def train(args, logger, config):

    # -------------------------------------------------------------------------
    # 1. Settings
    # -------------------------------------------------------------------------

    # Cuda setting
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{args.cuda_num}" if use_cuda else "cpu")
    logger.info(f"Device: {device}")

    # Random seed
    torch.manual_seed(args.seed)

    # Tensorboard writer
    writer = tb.SummaryWriter(args.logdir)

    # Timer
    timer = datetime.datetime.now()

    # -------------------------------------------------------------------------
    # 2. Data
    # -------------------------------------------------------------------------

    # Loader
    train_loader, test_loader = dvd.init_mnist_dataloader(
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

    if args.model == "vae":
        model = dvm.BaseVAE(**config["vae_params"], **params)
    elif args.model == "beta":
        model = dvm.BetaVAE(**config["beta_params"], **params)
    elif args.model == "factor":
        model = dvm.FactorVAE(**config["factor_params"], **params)
    elif args.model == "dip-i":
        model = dvm.DIPVAE(**config["dip-i_params"], **params)
    elif args.model == "dip-ii":
        model = dvm.DIPVAE(**config["dip-ii_params"], **params)
    elif args.model == "joint":
        model = dvm.JointVAE(**config["joint_params"], **params)
    else:
        raise KeyError(f"Not implemented model is specified, {args.model}")

    # -------------------------------------------------------------------------
    # 4. Training
    # -------------------------------------------------------------------------

    for epoch in range(1, args.epochs + 1):
        logger.info(f"--- Epoch {epoch} ---")

        # Training
        train_loss = model.run(train_loader, training=True)
        logger.info(f"Train loss = {train_loss['loss']}")
        for key, value in train_loss.items():
            writer.add_scalar(f"train/{key}", value, epoch)

        # Test
        if epoch % args.test_interval == 0:
            test_loss = model.run(test_loader, training=False)
            logger.info(f"Test loss = {test_loss['loss']}")
            for key, value in test_loss.items():
                writer.add_scalar(f"test/{key}", value, epoch)

        # Sample data
        if epoch % args.plot_interval == 0:
            # Sample data
            logger.info("Sample data")
            z_sample, x_sample = model.sample(32)
            writer.add_images(
                "sample/latnet", z_sample.view(32, 1, -1, 1), epoch)
            writer.add_images("sample/observable", x_sample, epoch)

            # Reconstruction data
            logger.info("Reconstruct data")
            z, x_recon = model.reconstruction(x_org[:8])
            img = torch.cat([x_org[:8], x_recon])
            writer.add_images("reconstruct/latent", z.view(8, 1, -1, 1), epoch)
            writer.add_images("reconstruct/observable", img, epoch)

        # Save model
        if epoch % args.save_interval == 0:
            logger.info(f"Save model at epoch {epoch}")
            t = timer.strftime("%Y%m%d%H%M%S")
            filename = f"{args.model}_{t}_epoch_{epoch}.pt"
            torch.save({"distributions_dict": model.distributions.state_dict(),
                        "optimizer_dict": model.optimizer.state_dict()},
                       pathlib.Path(args.logdir, filename))

    # -------------------------------------------------------------------------
    # 5. Summary
    # -------------------------------------------------------------------------

    # Log hyper-parameters
    hparam_dict = vars(args)
    hparam_dict.update(config[f"{args.model}_params"])
    metric_dict = {"summary/train_loss": train_loss["loss"]}
    if "test_loss" in locals():
        metric_dict.update({"summary/test_loss": test_loss["loss"]})
    writer.add_hparams(hparam_dict, metric_dict)

    writer.close()


def init_args():
    parser = argparse.ArgumentParser(description="VAE training")
    parser.add_argument("--logdir", type=str, default="../logs/tmp/")
    parser.add_argument("--root", type=str, default="../data/mnist/")
    parser.add_argument("--config", type=str, default="./config.json")
    parser.add_argument("--model", type=str, default="vae")
    parser.add_argument("--cuda-num", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--test-interval", type=int, default=10)
    parser.add_argument("--plot-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=100)

    return parser.parse_args()


def main():
    # Args
    args = init_args()

    # Make logdir
    dvu.check_logdir(args.logdir)

    # Logger
    logger = dvu.init_logger(args.logdir)
    logger.info("Start logger")
    logger.info(f"Commant line args: {args}")

    # Config
    config = dvu.load_config(args.config)
    logger.info(f"Configs: {config}")

    try:
        train(args, logger, config)
    except Exception as e:
        logger.exception(f"Run function error: {e}")
    finally:
        logger.info("End logger")


if __name__ == "__main__":
    main()
