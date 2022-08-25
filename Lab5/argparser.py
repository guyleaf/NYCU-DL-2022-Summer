from typing import Literal

import torch
from tap import Tap


class ArgumentParser(Tap):
    network_type: Literal["acgan", "cgan"] = "acgan"
    generator_lr: float = 2e-4  # learning rate for generator
    discriminator_lr: float = 2e-4  # learning rate for discriminator
    batch_size: int = 64  # batch size
    epoch_size: int = 100  # number of epochs to train for
    momentum: float = 0.5  # momentum of Adam optimizer
    z_dim: int = 100  # dimensionality of z noise
    label_smoothing_ratio: float = 0  # the ratio for label smoothing
    model_file: str = ""  # path of model file
    save_model_for_every_epoch: bool = False
    data_dir: str = "./data"  # path of data folder
    log_dir: str = "./logs"
    custom_log_dir_name: str = ""
    comments: str = ""
    max_objects: int = 3  # maximum objects in one generated image
    seed: int = 1  # manual seed
    num_workers: int = 4  # number of workers for dataloader
    device: str = "cuda"  # device


def parse_args() -> ArgumentParser:
    args = ArgumentParser().parse_args()
    assert args.device == "cpu" or (
        args.device == "cuda" and torch.cuda.is_available()
    )
    assert 0 <= args.label_smoothing_ratio and args.label_smoothing_ratio <= 1

    return args
