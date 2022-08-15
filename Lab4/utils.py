import math
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.init as init
from PIL import Image, ImageDraw
from scipy import signal
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from torchvision.utils import save_image


def kl_criterion(mu: torch.Tensor, logvar: torch.Tensor):
    # derivation: https://stats.stackexchange.com/a/7443
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


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
                psnr[i, t] += finn_psnr(origin[c], predict[c])
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


def show_curves(
    saved_path: str,
    kl_weights: List[float],
    tfrs: List[float],
    losses: List[float],
    average_psnrs: List[float],
):
    epochs = list(range(1, len(kl_weights) + 1))
    plt.figure()
    plt.title("train loss")
    plt.xlabel("epoch")
    plt.ylabel("score/weight")
    plt.plot(
        epochs, kl_weights, linestyle="dashed", color="g", label="KL weight"
    )
    plt.plot(epochs, tfrs, linestyle="dashed", color="b", label="tfr")
    plt.plot(epochs, losses, color="r", label="loss")
    plt.legend()
    plt.savefig(f"{saved_path}/train_loss.png")

    plt.figure()
    plt.title("train loss")
    plt.xlabel("epoch")
    plt.ylabel("score/weight")
    plt.plot(
        epochs, kl_weights, linestyle="dashed", color="g", label="KL weight"
    )
    plt.plot(epochs, tfrs, linestyle="dashed", color="b", label="tfr")
    plt.plot(
        epochs[::5],
        average_psnrs,
        linestyle="dashdot",
        color="r",
        label="psnr",
    )
    plt.legend()
    plt.savefig(f"{saved_path}/psnr.png")


if __name__ == "__main__":
    a = np.random.rand(50)
    losses = np.random.rand(50)
    psnrs = np.random.rand(10)
    show_curves("./", np.random.rand(50), a[::-1], losses, psnrs)
