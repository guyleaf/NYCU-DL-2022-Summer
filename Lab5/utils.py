import json
from typing import Dict, List

import matplotlib.pyplot as plt
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


def calculate_accuracy_of_classifier(
    pred: torch.Tensor, gt: torch.Tensor
) -> float:
    batch_size = pred.size(0)
    acc = 0
    total = 0
    for i in range(batch_size):
        k = int(gt[i].sum().item())
        total += k
        outv, outi = pred[i].topk(k)
        lv, li = gt[i].topk(k)
        for j in outi:
            if j in li:
                acc += 1
    return acc / total


def show_curves(saved_path: str, metrics: Dict[str, List[float]]):
    epochs = list(range(len(metrics["train_d_loss"])))
    _, ax1 = plt.subplots()
    plt.title("train loss")

    ax1.set_xlabel("epoch")
    ax1.set_ylabel("D loss", color="b")
    ax1.tick_params("y", colors="b")
    ax1.plot(
        epochs,
        metrics["train_d_loss"],
        linestyle="dashdot",
        color="b",
        label="D loss",
    )

    ax2 = ax1.twinx()
    ax2.set_ylabel("G loss", color="g")
    ax2.tick_params("y", colors="g")
    ax2.plot(
        epochs,
        metrics["train_g_loss"],
        linestyle="dashdot",
        color="g",
        label="G loss",
    )

    handles, _ = [
        (a + b)
        for a, b in zip(
            ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels()
        )
    ]
    plt.legend(handles=handles)
    plt.savefig(f"{saved_path}/train_loss.png")

    plt.figure()
    plt.title("test accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.plot(epochs, metrics["test_accuracy"])
    plt.savefig(f"{saved_path}/test_accuracy.png")


# custom weights initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)
