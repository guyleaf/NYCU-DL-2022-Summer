import math
import imageio
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.init as init
from PIL import Image, ImageDraw
from scipy import signal
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric


def kl_criterion(mu: torch.Tensor, logvar: torch.Tensor):
    # derivation: https://stats.stackexchange.com/a/7443
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)


def eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            origin = gt[t][i]
            predict = pred[t][i]
            for c in range(origin.shape[0]):
                ssim[i, t] += ssim_metric(origin[c], predict[c])
                psnr[i, t] += psnr_metric(origin[c], predict[c])
            ssim[i, t] /= origin.shape[0]
            psnr[i, t] /= origin.shape[0]
            mse[i, t] = mse_metric(origin, predict)

    return mse, ssim, psnr


def mse_metric(x1, x2):
    err = np.sum((x1 - x2) ** 2)
    err /= float(x1.shape[0] * x1.shape[1] * x1.shape[2])
    return err


# ssim function used in Babaeizadeh et al. (2017), Fin et al. (2016), etc.
def finn_eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            origin = gt[t][i]
            predict = pred[t][i]
            for c in range(origin.shape[0]):
                res = finn_ssim(origin[c], predict[c]).mean()
                if math.isnan(res):
                    ssim[i, t] += -1
                else:
                    ssim[i, t] += res
                psnr[i, t] += psnr_metric(
                    origin[c], predict[c], data_range=1.0
                )
            ssim[i, t] /= origin.shape[0]
            psnr[i, t] /= origin.shape[0]
            mse[i, t] = mse_metric(origin, predict)

    return mse, ssim, psnr


def finn_psnr(x, y, data_range=1.0):
    mse = ((x - y) ** 2).mean()
    return 20 * math.log10(data_range) - 10 * math.log10(mse)


def fspecial_gauss(size, sigma):
    x, y = np.mgrid[
        -size // 2 + 1 : size // 2 + 1, -size // 2 + 1 : size // 2 + 1
    ]
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return g / g.sum()


def finn_ssim(img1, img2, data_range=1.0, cs_map=False):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)

    K1 = 0.01
    K2 = 0.03

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    mu1 = signal.fftconvolve(img1, window, mode="valid")
    mu2 = signal.fftconvolve(img2, window, mode="valid")
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = signal.fftconvolve(img1 * img1, window, mode="valid") - mu1_sq
    sigma2_sq = signal.fftconvolve(img2 * img2, window, mode="valid") - mu2_sq
    sigma12 = signal.fftconvolve(img1 * img2, window, mode="valid") - mu1_mu2

    if cs_map:
        return (
            ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
            / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)),
            (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2),
        )
    else:
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )


def init_weights(m: torch.nn.Module):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("Linear") != -1:
        init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        init.normal_(m.weight, 1.0, 0.02)
        init.constant_(m.bias, 0.0)


def show_curves(saved_path: str, metrics: Dict[str, List[float]]):
    epochs = list(range(len(metrics["kl_weights"])))
    _, ax1 = plt.subplots()
    plt.title("train loss")

    ax1.set_xlabel("epoch")
    ax1.set_ylabel("ratio/weight", color="b")
    ax1.tick_params("y", colors="b")
    ax1.plot(
        epochs,
        metrics["kl_weights"],
        linestyle="dashed",
        color="g",
        label="KL weight",
    )
    ax1.plot(
        epochs, metrics["tfrs"], linestyle="dashed", color="b", label="tfr"
    )

    ax2 = ax1.twinx()
    ax2.set_ylabel("loss")
    ax2.plot(epochs, metrics["train_losses"], color="r", label="loss")

    handles, _ = [
        (a + b)
        for a, b in zip(
            ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels()
        )
    ]
    plt.legend(handles=handles)
    plt.savefig(f"{saved_path}/train_loss.png")

    _, ax1 = plt.subplots()
    plt.title("Average PSNR")

    ax1.set_xlabel("epoch")
    ax1.set_ylabel("ratio/weight", color="b")
    ax1.tick_params("y", colors="b")
    ax1.plot(
        epochs,
        metrics["kl_weights"],
        linestyle="dashed",
        color="g",
        label="KL weight",
    )
    ax1.plot(
        epochs, metrics["tfrs"], linestyle="dashed", color="b", label="tfr"
    )

    ax2 = ax1.twinx()
    ax2.set_ylabel("psnr score")
    ax2.plot(
        epochs,
        metrics["average_psnrs"],
        linestyle="dashdot",
        color="r",
        label="psnr",
    )

    handles, _ = [
        (a + b)
        for a, b in zip(
            ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels()
        )
    ]
    plt.legend(handles=handles)
    plt.savefig(f"{saved_path}/psnr.png")


def add_border(x: np.ndarray, color: str, pad=1):
    w = x.shape[1]
    nc = x.shape[0]
    px = np.zeros(3, w + 2 * pad + 30, w + 2 * pad)
    if color == "red":
        px[0] = 0.7
    elif color == "green":
        px[1] = 0.7

    right_pad = w + pad
    if nc == 1:
        for c in range(3):
            px[c, pad:right_pad, pad:right_pad] = x
    else:
        px[:, pad:right_pad, pad:right_pad] = x
    return px


def image_tensor(inputs, padding=1):
    assert len(inputs) > 0

    # if this is a list of lists, unpack them all and grid them up
    if is_sequence(inputs[0]) or (hasattr(inputs, "dim") and inputs.dim() > 4):
        images = [image_tensor(x) for x in inputs]
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(
            c_dim, x_dim * len(images) + padding * (len(images) - 1), y_dim
        )
        for i, image in enumerate(images):
            result[
                :, i * x_dim + i * padding : (i + 1) * x_dim + i * padding, :
            ].copy_(image)

        return result

    # if this is just a list, make a stacked image
    else:
        images = [
            x.data if isinstance(x, torch.autograd.Variable) else x
            for x in inputs
        ]
        # print(images)
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(
            c_dim, x_dim, y_dim * len(images) + padding * (len(images) - 1)
        )
        for i, image in enumerate(images):
            result[
                :, :, i * y_dim + i * padding : (i + 1) * y_dim + i * padding
            ].copy_(image)
        return result


def make_gif(
    origin: np.ndarray,
    posterior: np.ndarray,
    best: np.ndarray,
    random_samples: np.ndarray,
    n_past: int,
    path: str,
    duration: float = 0.25,
):
    length_of_sequence = origin.shape[0]
    gifs = np.empty((length_of_sequence, 1))
    texts = np.empty_like(gifs)
    for t in range(length_of_sequence):
        # gt
        gifs[t].append(add_border(origin[t], "green"))
        texts[t].append("Ground\ntruth")
        # posterior
        if t < n_past:
            color = "green"
        else:
            color = "red"
        gifs[t].append(add_border(posterior[t], color))
        texts[t].append("Approx.\nposterior")
        # best
        if t < n_past:
            color = "green"
        else:
            color = "red"
        gifs[t].append(add_border(best[t], color))
        texts[t].append("Best SSIM")
        # random 3
        for i, sequence in enumerate(random_samples):
            gifs[t].append(add_border(sequence[t], color))
            texts[t].append("Random\nsample %d" % (i + 1))

    images = []
    for gif, _ in zip(gifs, texts):
        gif = np.clip(gif, 0, 1)
        images.append(gif)
    imageio.mimsave(path, images, duration=duration)


def export_sequence(sequence: np.ndarray, path: str):
    imgs = np.empty(sequence.shape[0])
    for t, img in enumerate(sequence):
        imgs[t] = add_border(img, "black")

    Image.fromarray()


if __name__ == "__main__":
    a = np.random.rand(50)
    losses = np.random.rand(50)
    psnrs = np.random.rand(10)
    show_curves("./", np.random.rand(50), a[::-1], losses, psnrs)
