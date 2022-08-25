import os
from datetime import datetime
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from acgan import ACGANDiscriminator, ACGANGenerator
from cgan import CGANDiscriminator, CGANGenerator
from argparser import ArgumentParser, parse_args
from datasets import CLEVRDataset
from evaluator import EvaluationModel
from train import train_acgan, train_cgan
from utils import (
    sample_noises,
    sample_z,
    save_images,
    set_seeds,
    show_curves,
    weights_init,
)


def save_model(
    path: str,
    epoch: int,
    generator: nn.Module,
    discriminator: nn.Module,
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


@torch.no_grad()
def evaluate(
    args: ArgumentParser, generator: nn.Module, test_dataloader: DataLoader
) -> Tuple[float, torch.Tensor]:
    evaluator = EvaluationModel()

    generator.eval()

    total_accuracy = 0
    generated_images = []
    for _, labels in test_dataloader:
        labels = labels.to(args.device)
        batch_size = labels.shape[0]

        # sample noises z
        z = sample_z(args.z_dim, batch_size, args.device)

        sampled_images = generator(z, labels)
        total_accuracy += evaluator.eval(sampled_images, labels)

        generated_images.append(sampled_images.cpu())

    generated_images = torch.concat(generated_images, dim=0)
    return total_accuracy, generated_images


def train(
    args: ArgumentParser,
    train_fn,
    metrics: dict,
    start_epoch: int,
    n_classes: int,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    new_test_dataloader: DataLoader,
    generator: nn.Module,
    discriminator: nn.Module,
    g_optimizer: optim.Optimizer,
    d_optimizer: optim.Optimizer,
):
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

    n_batches = len(train_dataloader)
    for epoch in trange(
        start_epoch,
        args.epoch_size,
        initial=start_epoch,
        total=args.epoch_size,
        desc="Current epoch",
    ):
        train_d_loss_list = []
        train_g_loss_list = []
        test_accuracy_list = []
        for batch, (images, labels) in enumerate(
            tqdm(train_dataloader, leave=False, desc="Current batch")
        ):
            images = images.to(args.device)
            labels = labels.to(args.device)

            batch_size = images.shape[0]

            real = torch.ones(
                (batch_size, 1),
                requires_grad=False,
                device=args.device,
                dtype=torch.float32,
            )
            fake = torch.zeros(
                (batch_size, 1),
                requires_grad=False,
                device=args.device,
                dtype=torch.float32,
            )

            z, c = sample_noises(
                args.z_dim,
                batch_size,
                n_classes,
                args.max_objects,
                args.device,
            )

            # -----------------
            #  Train
            # -----------------

            d_loss, d_real_loss, d_fake_loss, g_loss = train_fn(
                images,
                labels,
                generator,
                discriminator,
                d_optimizer,
                g_optimizer,
                real,
                fake,
                z,
                c,
                args.label_smoothing_ratio,
            )

            train_d_loss_list.append(d_loss)
            train_g_loss_list.append(g_loss)

            with open(train_record_path, "a") as train_record:
                train_record.write(
                    (
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %.4f, real loss: %.4f, fake loss: %.4f]\n"
                        % (
                            epoch,
                            args.epoch_size,
                            batch,
                            n_batches,
                            d_loss,
                            d_real_loss,
                            d_fake_loss,
                        )
                    )
                )
                train_record.write(
                    (
                        "[Epoch %d/%d] [Batch %d/%d] [G loss: %.4f]\n"
                        % (
                            epoch,
                            args.epoch_size,
                            batch,
                            n_batches,
                            g_loss,
                        )
                    )
                )

            # -----------------
            #  Evaluate batches
            # -----------------

            # evaluator
            test_accuracy, generated_images = evaluate(
                args, generator, test_dataloader
            )
            new_test_accuracy, new_generated_images = evaluate(
                args, generator, new_test_dataloader
            )
            total_test_accuracy = (test_accuracy + new_test_accuracy) / 2

            with open(train_record_path, "a") as train_record:
                train_record.write(
                    (
                        "[Epoch %d/%d] [Batch %d/%d] [Test accuracy: %.4f%%] [New Test accuracy: %.4f%%]\n"
                        % (
                            epoch,
                            args.epoch_size,
                            batch,
                            n_batches,
                            test_accuracy * 100,
                            new_test_accuracy * 100,
                        )
                    )
                )

            test_accuracy_list.append(total_test_accuracy)

            if total_test_accuracy > metrics["best_batch_test_accuracy"]:
                metrics["best_batch_test_accuracy"] = total_test_accuracy
                save_images(
                    f"{test_images_dir}/epoch={epoch}-batch={batch}.jpg",
                    generated_images,
                )
                save_images(
                    f"{test_images_dir}/epoch={epoch}-batch={batch}-new.jpg",
                    new_generated_images,
                )
                save_model(
                    f"{args.log_dir}/best_batch_model_epoch={epoch}-batch={batch}.pth",
                    epoch,
                    generator,
                    discriminator,
                    args,
                    metrics,
                )

        # -----------------
        #  Evaluate epochs
        # -----------------

        train_d_loss = np.array(train_d_loss_list).mean()
        train_g_loss = np.array(train_g_loss_list).mean()
        test_accuracy = np.array(test_accuracy_list).mean()

        with open(train_record_path, "a") as train_record:
            train_record.write(
                (
                    "[Epoch %d/%d] [D loss: %.4f] [G loss: %.4f] [Eval accuracy: %.4f%%]\n"
                    % (
                        epoch,
                        args.epoch_size,
                        train_d_loss,
                        train_g_loss,
                        test_accuracy * 100,
                    )
                )
            )

        metrics["train_d_loss"].append(train_d_loss)
        metrics["train_g_loss"].append(train_g_loss)
        metrics["test_accuracy"].append(test_accuracy)

        if test_accuracy > metrics["best_test_accuracy"]:
            metrics["best_test_accuracy"] = test_accuracy
            save_model(
                f"{args.log_dir}/best_model.pth",
                epoch,
                generator,
                discriminator,
                args,
                metrics,
            )

        if args.save_model_for_every_epoch:
            save_model(
                f"{models_dir}/{epoch}.pth",
                epoch,
                generator,
                discriminator,
                args,
                metrics,
            )


def main():
    args = parse_args()

    if args.model_file != "":
        # load model and continue training from checkpoint
        saved_model = torch.load(args.model_file)
        saved_args: ArgumentParser = saved_model["args"]
        saved_args.data_dir = args.data_dir
        saved_args.model_file = args.model_file
        saved_args.epoch_size = args.epoch_size
        saved_args.batch_size = args.batch_size
        saved_args.save_model_for_every_epoch = args.save_model_for_every_epoch
        saved_args.comments = args.comments

        args = saved_args
        args.log_dir = "%s/continued" % args.log_dir
        start_epoch = saved_model["last_epoch"] + 1
        networks = saved_model["networks"]
        metrics = saved_model["metrics"]
    else:
        name = (
            args.custom_log_dir_name
            if args.custom_log_dir_name != ""
            else (
                "%s-g_lr=%.5f-d_lr=%.5f-momentum=%.5f-z_dim=%d-epoch_size=%d-max_objects=%d"
                % (
                    datetime.now().strftime("%Y-%m-%d_%H%M%S"),
                    args.generator_lr,
                    args.discriminator_lr,
                    args.momentum,
                    args.z_dim,
                    args.epoch_size,
                    args.max_objects,
                )
            )
        )

        args.log_dir = f"{args.log_dir}/{name}"
        start_epoch = 0
        metrics = {
            "train_d_loss": [],
            "train_g_loss": [],
            "test_accuracy": [],
            "best_test_accuracy": 0,
            "best_batch_test_accuracy": 0,
        }

    print("Random seed:", args.seed)
    set_seeds(args.seed)

    print(args)

    os.makedirs(args.log_dir, exist_ok=True)

    # --------- load a dataset ----------------
    train_data = CLEVRDataset(args.data_dir, mode="train")
    test_data = CLEVRDataset(args.data_dir, mode="test")
    new_test_data = CLEVRDataset(args.data_dir, mode="new_test")
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
    new_test_dataloader = DataLoader(
        new_test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print("Train data samples: ", len(train_data))
    print("Test data samples: ", len(test_data))
    print("New Test data samples: ", len(new_test_data))

    # --------- build models ------------------
    n_classes = train_data.get_n_classes()
    if args.network_type == "acgan":
        generator = ACGANGenerator(n_classes, args.z_dim, 3)
        discriminator = ACGANDiscriminator(3, n_classes)
        train_fn = train_acgan
    elif args.network_type == "cgan":
        generator = CGANGenerator(n_classes, args.z_dim, 3)
        discriminator = CGANDiscriminator(3, n_classes)
        train_fn = train_cgan
    else:
        raise NotImplementedError(
            f"The network type {args.network_type} is not implemented."
        )

    if args.model_file != "":
        generator.load_state_dict(networks["generator"])
        discriminator.load_state_dict(networks["discriminator"])
    else:
        generator.apply(weights_init)
        discriminator.apply(weights_init)

    generator.to(args.device)
    discriminator.to(args.device)

    # --------- build optimizers --------------
    g_optimizer = optim.Adam(
        generator.parameters(),
        lr=args.generator_lr,
        betas=(args.momentum, 0.999),
    )
    d_optimizer = optim.Adam(
        discriminator.parameters(),
        lr=args.discriminator_lr,
        betas=(args.momentum, 0.999),
    )

    try:
        train(
            args,
            train_fn,
            metrics,
            start_epoch,
            n_classes,
            train_dataloader,
            test_dataloader,
            new_test_dataloader,
            generator,
            discriminator,
            g_optimizer,
            d_optimizer,
        )
    finally:
        show_curves(args.log_dir, metrics)


if __name__ == "__main__":
    main()
