import json

import numpy as np
import torch
from torchvision.utils import make_grid, save_image


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_images(path: str, images: torch.Tensor):
    grid = make_grid(images, normalize=True)
    save_image(grid, path)


# custom weights initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)
