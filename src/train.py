
"""Training method"""

import argparse
import json
import os
import pathlib

import numpy as np
import torch
from torch.backends import cudnn
import pytorch_lightning as pl

import dgmvae.models as dvm
import dgmvae.updaters as dvu

import utils_pytorch as utils


def main():

    # -------------------------------------------------------------------------
    # 1. Settings
    # -------------------------------------------------------------------------

    # Kwargs
    args = init_args()

    # Configs
    condig_path = os.getenv("CONFIG_PATH", "./train/config.json")
    with pathlib.Path(condig_path).open() as f:
        config = json.load(f)

    # Cuda setting
    use_cuda = torch.cuda.is_available() and args.cuda != "null"
    gpus = args.cuda if use_cuda else None

    # Random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    # -------------------------------------------------------------------------
    # 2. Training
    # -------------------------------------------------------------------------

    # VAE model
    model_dict = {
        "beta": dvm.BetaVAE,
        "factor": dvm.FactorVAE,
        "dipi": dvm.DIPVAE,
        "dipii": dvm.DIPVAE,
        "joint": dvm.JointVAE,
        "tcvae": dvm.TCVAE,
        "aae": dvm.AAE,
        "avb": dvm.AVB,
    }
    model = model_dict[args.model](**config[f"{args.model}_params"])

    # Updater
    root = os.getenv("DATA_ROOT", "./data/mnist/")
    updater = dvu.VAEUpdater(model, args, root, args.batch_size)

    # Trainer
    params = {
        "default_save_path": os.getenv("SAVE_PATH", "./logs/"),
        "gpus": gpus,
        "early_stop_callback": None,
        "max_epochs": args.epochs,
        "check_val_every_n_epoch": args.val_interval,
        "log_save_interval": args.log_save_interval,
    }
    trainer = pl.Trainer(**params)

    # Run
    trainer.fit(updater)

    # Deep copy
    trained_model = model_dict[args.model](**config[f"{args.model}_params"])
    trained_model.load_state_dict(updater.model.state_dict())

    # Export model
    ch_num = config[f"{args.model}_params"]["channel_num"]
    utils.export_model(updater.model, input_shape=(1, ch_num, 64, 64))


def init_args():
    parser = argparse.ArgumentParser(description="VAE training")
    parser.add_argument("--model", type=str, default="beta")
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--log-save-interval", type=int, default=5)

    return parser.parse_args()


if __name__ == "__main__":
    main()
