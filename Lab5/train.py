import os
from datetime import datetime
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tap import Tap
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from datasets import CLEVRDataset
from evaluator import EvaluationModel
from models import Discriminator, Generator
from utils import save_images, set_seeds, weights_init


class ArgumentParser(Tap):
    lr: float = 2e-3  # learning rate
    batch_size: int = 64  # batch size
    epoch_size: int = 25  # number of epochs to train for
    momentum: float = 0.9  # momentum of Adam optimizer
    z_dim: int = 100  # dimensionality of z noise
    model_file: str = ""  # path of model file
    save_model_for_every_epoch: bool = False
    data_dir: str = "./data"  # path of data folder
    log_dir: str = "./logs"
    custom_log_dir_name: str = ""
    max_objects: int = 3  # maximum objects in one generated image
    seed: int = 1  # manual seed
    num_workers: int = 4  # number of workers for dataloader
    device: str = "cuda"  # device


@torch.no_grad()
def evaluate(
    args: ArgumentParser,
    generator: nn.Module,
    test_dataloader: DataLoader,
) -> Tuple[float, torch.Tensor]:
    evaluator = EvaluationModel()

    generator.eval()

    total_accuracy = 0
    generated_images = []
    for _, labels in test_dataloader:
        labels = labels.to(args.device)
        batch_size = labels.shape[0]

        # sample noises z
        z = torch.randn((batch_size, args.z_dim, 1, 1), device=args.device)

        sampled_images = generator(z, labels.unsqueeze(-1).unsqueeze(-1))
        total_accuracy += evaluator.eval(sampled_images, labels)

        generated_images.append(sampled_images.cpu())

    generated_images = torch.concat(generated_images, dim=0)
    return total_accuracy / len(test_dataloader.dataset), generated_images


def save_model(
    path: str,
    epoch: int,
    generator: Generator,
    discriminator: Discriminator,
    args: ArgumentParser,
    metrics={},
):
    torch.save(
        {
            "networks": {
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
            },
            "args": args,
            "last_epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def main():
    args = ArgumentParser().parse_args()
    assert args.device == "cpu" or (
        args.device == "cuda" and torch.cuda.is_available()
    )

    if args.model_file != "":
        # load model and continue training from checkpoint
        saved_model = torch.load(args.model_file)
        saved_args: ArgumentParser = saved_model["args"]
        saved_args.data_dir = args.data_dir
        saved_args.model_file = args.model_file
        saved_args.epoch_size = args.epoch_size
        saved_args.batch_size = args.batch_size
        saved_args.save_model_for_every_epoch = args.save_model_for_every_epoch

        args = saved_args
        args.log_dir = "%s/continued" % args.log_dir
        start_epoch = saved_model["last_epoch"] + 1
        networks = saved_model["networks"]
        metrics = saved_model["metrics"]
        best_test_accuracy = max(metrics["test_accuracy"])
    else:
        name = (
            args.custom_log_dir_name
            if args.custom_log_dir_name != ""
            else (
                "%s-lr=%.5f-momentum=%.5f-z_dim=%d-epoch_size=%d-max_objects=%d"
                % (
                    datetime.now().strftime("%Y-%m-%d_%H%M%S"),
                    args.lr,
                    args.momentum,
                    args.z_dim,
                    args.epoch_size,
                    args.max_objects,
                )
            )
        )

        args.log_dir = f"{args.log_dir}/{name}"
        start_epoch = 0
        metrics = {"test_accuracy": []}
        best_test_accuracy = 0

    print("Random seed:", args.seed)
    set_seeds(args.seed)

    print(args)

    # --------- setup logging ----------------
    train_record_path = f"{args.log_dir}/train_record.log"
    models_dir = f"{args.log_dir}/models"
    test_images_dir = f"{args.log_dir}/test"

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(test_images_dir, exist_ok=True)

    if os.path.exists(train_record_path):
        os.remove(train_record_path)

    with open(train_record_path, "a") as train_record:
        train_record.write(f"args: {args}\n")

    # --------- load a dataset ----------------
    train_data = CLEVRDataset(args.data_dir, mode="train")
    test_data = CLEVRDataset(args.data_dir, mode="test")
    train_dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print("Train data samples: ", len(train_data))
    print("Test data samples: ", len(test_data))

    # --------- build models ------------------
    generator = Generator(args.z_dim + 24, 3)
    discriminator = Discriminator(3, 24)
    if args.model_file != "":
        generator.load_state_dict(networks["generator"])
        discriminator.load_state_dict(networks["discriminator"])
    else:
        generator.apply(weights_init)
        discriminator.apply(weights_init)

    generator.to(args.device)
    discriminator.to(args.device)

    # --------- build optimizers --------------
    generator_optimizer = optim.Adam(
        generator.parameters(), lr=args.lr, betas=(args.momentum, 0.999)
    )
    discriminator_optimizer = optim.Adam(
        discriminator.parameters(), lr=args.lr, betas=(args.momentum, 0.999)
    )

    loss_fn = nn.BCEWithLogitsLoss()
    number_of_batches = len(train_dataloader)
    # --------- train loop --------------------
    for epoch in trange(
        start_epoch, args.epoch_size, initial=start_epoch, desc="Current epoch"
    ):
        test_accuracy_list = []
        for batch, (real_images, real_labels) in enumerate(
            tqdm(train_dataloader, desc="Current batch")
        ):
            real_images = real_images.to(args.device)
            real_labels = real_labels.to(args.device)

            batch_size = real_images.shape[0]

            real = torch.randn(
                (batch_size, 1), requires_grad=False, device=args.device
            )
            fake = torch.randn(
                (batch_size, 1), requires_grad=False, device=args.device
            )

            generator.train()

            # -----------------
            #  Train Discriminator
            # -----------------

            discriminator_optimizer.zero_grad()

            # train with real images
            real_or_fake, pred_labels = discriminator(real_images)
            adversarial_loss = loss_fn(real_or_fake, real)
            auxiliary_loss = loss_fn(pred_labels, real_labels)

            d_real_loss = adversarial_loss + auxiliary_loss

            # train with fake images
            # sample noises z
            z = torch.randn(
                (batch_size, args.z_dim, 1, 1),
                requires_grad=False,
                device=args.device,
            )

            # sample one-hot labels c
            sampled_labels = torch.randint(
                low=0,
                high=24,
                size=(batch_size, args.max_objects),
                requires_grad=False,
                device=args.device,
            )
            sampled_labels = torch.sum(
                F.one_hot(sampled_labels, num_classes=24), dim=1
            ).float()

            fake_images = generator(
                z, sampled_labels.unsqueeze(-1).unsqueeze(-1)
            )

            real_or_fake, pred_labels = discriminator(fake_images.detach())
            adversarial_loss = loss_fn(real_or_fake, fake)
            auxiliary_loss = loss_fn(pred_labels, sampled_labels)

            d_fake_loss = adversarial_loss + auxiliary_loss

            d_total_loss = d_real_loss + d_fake_loss
            d_total_loss.backward()
            discriminator_optimizer.step()

            # -----------------
            #  Train Generator
            # -----------------

            generator_optimizer.zero_grad()

            real_or_fake, pred_labels = discriminator(fake_images)
            adversarial_loss = loss_fn(real_or_fake, real)
            auxiliary_loss = loss_fn(pred_labels, sampled_labels)

            g_loss = adversarial_loss + auxiliary_loss
            g_loss.backward()
            generator_optimizer.step()

            # -----------------
            #  Evaluate batches
            # -----------------
            test_accuracy, generated_images = evaluate(
                args, generator, test_dataloader
            )
            test_accuracy_list.append(test_accuracy)
            save_images(
                f"{test_images_dir}/epoch={epoch}-batch={batch}.jpg",
                generated_images,
            )

            with open(train_record_path, "a") as train_record:
                train_record.write(
                    (
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %.4f] [G loss: %.4f] [Test accuracy: %.4f]\n"
                        % (
                            epoch,
                            args.epoch_size,
                            batch,
                            number_of_batches,
                            d_total_loss.item(),
                            g_loss.item(),
                            test_accuracy,
                        )
                    )
                )

        test_accuracy = np.array(test_accuracy_list).mean()
        metrics["test_accuracy"].append(test_accuracy)

        with open(train_record_path, "a") as train_record:
            train_record.write(
                (
                    "[Epoch %d/%d] [Average test accuracy: %.4f]\n"
                    % (
                        epoch,
                        args.epoch_size,
                        test_accuracy,
                    )
                )
            )

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            save_model(
                args.log_dir,
                epoch,
                generator,
                discriminator,
                args,
                metrics,
            )

        if args.save_model_for_every_epoch:
            save_model(
                models_dir, epoch, generator, discriminator, args, metrics
            )


if __name__ == "__main__":
    main()
