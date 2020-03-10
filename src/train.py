
"""Training method"""

import argparse
import json
import pathlib

import torch
import pytorch_lightning as pl

import dgmvae.models as dvm
import dgmvae.updaters as dvu


def main():

    # -------------------------------------------------------------------------
    # 1. Settings
    # -------------------------------------------------------------------------

    # Kwargs
    args = init_args()

    # Configs
    with pathlib.Path(args.config).open() as f:
        config = json.load(f)

    # Cuda setting
    use_cuda = torch.cuda.is_available() and args.cuda != "null"
    gpus = args.cuda if use_cuda else None

    # Random seed
    torch.manual_seed(args.seed)

    # -------------------------------------------------------------------------
    # 2. Training
    # -------------------------------------------------------------------------

    # VAE model
    if args.model == "beta":
        model = dvm.BetaVAE(**config["beta_params"])
    elif args.model == "factor":
        model = dvm.FactorVAE(**config["factor_params"])
    elif args.model == "dip-i":
        model = dvm.DIPVAE(**config["dip-i_params"])
    elif args.model == "dip-ii":
        model = dvm.DIPVAE(**config["dip-ii_params"])
    elif args.model == "joint":
        model = dvm.JointVAE(**config["joint_params"])
    elif args.model == "tcvae":
        model = dvm.TCVAE(**config["tcvae_params"])
    elif args.model == "aae":
        model = dvm.AAE(**config["aae_params"])
    elif args.model == "avb":
        model = dvm.AVB(**config["avb_params"])
    else:
        raise KeyError(f"Not implemented model is specified, {args.model}")

    # Updater
    updater = dvu.VAEUpdater(model, args, **config["updater_params"])

    # Trainer
    params = {
        "default_save_path": args.logdir,
        "gpus": gpus,
        "early_stop_callback": None,
        "max_epochs": args.epochs,
        "check_val_every_n_epoch": args.val_interval,
        "log_save_interval": args.log_save_interval,
    }
    trainer = pl.Trainer(**params)

    # Run
    trainer.fit(updater)


def init_args():
    parser = argparse.ArgumentParser(description="VAE training")
    parser.add_argument("--logdir", type=str, default="./logs/")
    parser.add_argument("--config", type=str, default="./src/config.json")
    parser.add_argument("--model", type=str, default="beta")
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--log-save-interval", type=int, default=100)

    return parser.parse_args()


if __name__ == "__main__":
    main()
