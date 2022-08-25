import torch
import torch.nn as nn
import torch.optim as optim

from acgan import ACGANDiscriminator, ACGANGenerator


def train_acgan(
    images: torch.Tensor,
    labels: torch.Tensor,
    generator: ACGANGenerator,
    discriminator: ACGANDiscriminator,
    d_optimizer: optim.Optimizer,
    g_optimizer: optim.Optimizer,
    real: torch.Tensor,
    fake: torch.Tensor,
    z: torch.Tensor,
    c: torch.Tensor,
    label_smoothing_ratio: float = 0,
):
    loss_fn = nn.BCEWithLogitsLoss()

    generator.train()
    discriminator.train()

    # -----------------
    #  Train Discriminator
    # -----------------

    d_optimizer.zero_grad()

    fake_images = generator(z, c)

    # train with real images
    pred_real_validity, pred_real_labels = discriminator(images)
    d_real_adversarial_loss = loss_fn(
        pred_real_validity, real - label_smoothing_ratio
    )
    d_real_auxiliary_loss = loss_fn(pred_real_labels, labels)
    d_real_loss = d_real_adversarial_loss + d_real_auxiliary_loss

    # train with fake images
    pred_fake_validity, pred_sampled_labels = discriminator(
        fake_images.detach()
    )
    d_fake_adversarial_loss = loss_fn(pred_fake_validity, fake)
    d_fake_auxiliary_loss = loss_fn(pred_sampled_labels, c)
    d_fake_loss = d_fake_adversarial_loss + d_fake_auxiliary_loss

    d_loss = d_real_loss + d_fake_loss
    d_loss.backward()
    d_optimizer.step()

    # -----------------
    #  Train Generator
    # -----------------

    g_optimizer.zero_grad()

    pred_validity, pred_labels = discriminator(fake_images)
    g_adversarial_loss = loss_fn(pred_validity, real)
    g_auxiliary_loss = loss_fn(pred_labels, c)
    g_loss = g_adversarial_loss + g_auxiliary_loss

    g_loss.backward()
    g_optimizer.step()

    return (
        d_loss.item(),
        d_real_loss.item(),
        d_fake_loss.item(),
        g_loss.item(),
    )


def train_cgan(
    images: torch.Tensor,
    labels: torch.Tensor,
    generator: ACGANGenerator,
    discriminator: ACGANDiscriminator,
    d_optimizer: optim.Optimizer,
    g_optimizer: optim.Optimizer,
    real: torch.Tensor,
    fake: torch.Tensor,
    z: torch.Tensor,
    c: torch.Tensor,
    label_smoothing_ratio: float = 0,
):
    loss_fn = nn.BCEWithLogitsLoss()

    generator.train()
    discriminator.train()

    # -----------------
    #  Train Discriminator
    # -----------------

    d_optimizer.zero_grad()

    fake_images = generator(z, c)

    # train with real images
    pred_real_validity = discriminator(images, labels)
    d_real_loss = loss_fn(pred_real_validity, real - label_smoothing_ratio)

    # train with fake images
    pred_fake_validity = discriminator(fake_images.detach(), c)
    d_fake_loss = loss_fn(pred_fake_validity, fake)

    d_loss = d_real_loss + d_fake_loss
    d_loss.backward()
    d_optimizer.step()

    # -----------------
    #  Train Generator
    # -----------------

    g_optimizer.zero_grad()

    pred_validity = discriminator(fake_images, c)
    g_loss = loss_fn(pred_validity, real)

    g_loss.backward()
    g_optimizer.step()

    return (
        d_loss.item(),
        d_real_loss.item(),
        d_fake_loss.item(),
        g_loss.item(),
    )
