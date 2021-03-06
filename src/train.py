
"""Training method"""

import argparse
import json
import os
import pathlib
from typing import Union

import numpy as np
import torch
from torch.backends import cudnn
import pytorch_lightning as pl

import dgmvae.models as dvm

from experiment import VAEUpdater


def main():

    # -------------------------------------------------------------------------
    # 1. Settings
    # -------------------------------------------------------------------------

    # Kwargs
    args = init_args()

    # Configs
    condig_path = os.getenv("CONFIG_PATH", "./src/config_ch1.json")
    with pathlib.Path(condig_path).open() as f:
        config = json.load(f)

    # Path
    root = pathlib.Path(os.getenv("DATA_ROOT", "./data/mnist/"))
    save_path = pathlib.Path(os.getenv("SAVE_PATH", "./logs/"),
                             os.getenv("EVALUATION_NAME", "dev"))
    model_path = save_path / "representation"
    dataset = os.getenv("DATASET_NAME", "mnist")

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
    updater = VAEUpdater(model, args, dataset, root, args.batch_size)

    # Trainer
    params = {
        "default_save_path": save_path,
        "gpus": gpus,
        "early_stop_callback": None,
        "max_steps": args.steps,
        "log_save_interval": args.log_save_interval,
    }
    trainer = pl.Trainer(**params)

    # Run
    trainer.fit(updater)

    # Export model
    model_path.mkdir()
    ch_num = config[f"{args.model}_params"]["channel_num"]
    export_model(updater.model, str(model_path / "pytorch_model.pt"),
                 input_shape=(1, ch_num, 64, 64))


def export_model(model: Union[torch.nn.Module, torch.jit.ScriptModule],
                 path: Union[str, pathlib.Path],
                 input_shape: tuple = (1, 3, 64, 64),
                 use_script_module: bool = True
                 ) -> Union[str, pathlib.Path]:
    """Exports model.

    Args:
        model (torch.nn.Module or torch.jit.ScriptModule): Saved model.
        path (str or pathlib.Path): Path to file.
        input_shape (tuple, optional): Tuple of input data shape.
        use_script_module (bool, optional): Boolean flag for using script
            module.

    Returns:
        path (str or pathlib.Path): Path to saved file.
    """

    model = model.cpu().eval()
    if isinstance(model, torch.jit.ScriptModule):
        assert use_script_module, \
            "Provided model is a ScriptModule, set use_script_module to True."
    if use_script_module:
        if not isinstance(model, torch.jit.ScriptModule):
            assert input_shape is not None
            traced_model = torch.jit.trace(model, torch.zeros(*input_shape))
        else:
            traced_model = model
        torch.jit.save(traced_model, path)
    else:
        torch.save(model, path)  # saves model as a nn.Module
    return path


def init_args():
    parser = argparse.ArgumentParser(description="VAE training")
    parser.add_argument("--model", type=str, default="beta")
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--log-save-interval", type=int, default=100)

    return parser.parse_args()


if __name__ == "__main__":
    main()
