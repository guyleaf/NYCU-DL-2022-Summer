import os
from datetime import datetime
from typing import Literal, Tuple

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
from utils import (
    calculate_accuracy_of_classifier,
    save_images,
    set_seeds,
    show_curves,
    weights_init,
)


class ArgumentParser(Tap):
    network_type: Literal["acgan+aux", "acgan+proj"] = "acgan+aux"
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


def sample_noises(args: ArgumentParser, batch_size: int, n_classes: int):
    # sample noises z
    z = torch.randn(
        (batch_size, args.z_dim),
        requires_grad=False,
        device=args.device,
    )

    # sample one-hot labels c
    # use multinomial to avoid generating duplicated labels
    # discussion: https://discuss.pytorch.org/t/how-to-generate-non-overlap-random-integer-tuple-efficiently/40869/4
    numbers_of_objects = torch.randint(
        1, args.max_objects + 1, size=(batch_size,)
    )
    # uniformly sampled
    weights = torch.ones(n_classes)
    indices = torch.multinomial(
        weights, num_samples=torch.sum(numbers_of_objects), replacement=True
    )

    sampled_labels = []
    groups = torch.split(indices, numbers_of_objects.tolist())
    for group in groups:
        label = torch.sum(
            F.one_hot(group, num_classes=n_classes),
            dim=0,
            dtype=torch.float32,
        )
        label = label.clamp_max(1)
        sampled_labels.append(label)
    sampled_labels = torch.stack(sampled_labels, dim=0).to(args.device)

    return (z, sampled_labels)


@torch.no_grad()
def evaluate(
    args: ArgumentParser,
    generator: nn.Module,
    test_dataloader: DataLoader,
    n_classes: int,
) -> Tuple[float, torch.Tensor]:
    evaluator = EvaluationModel()

    generator.eval()

    total_accuracy = 0
    generated_images = []
    for _, labels in test_dataloader:
        labels = labels.to(args.device)
        batch_size = labels.shape[0]

        # sample noises z
        z, _ = sample_noises(args, batch_size, n_classes)

        sampled_images = generator(z, labels)
        total_accuracy += evaluator.eval(sampled_images, labels)

        generated_images.append(sampled_images.cpu())

    generated_images = torch.concat(generated_images, dim=0)
    return total_accuracy, generated_images


def main():
    args = ArgumentParser().parse_args()
    assert args.device == "cpu" or (
        args.device == "cuda" and torch.cuda.is_available()
    )
    assert 0 <= args.label_smoothing_ratio and args.label_smoothing_ratio <= 1

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
        best_test_accuracy = max(metrics["test_accuracy"])
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
            "train_d_label_accuracy": [],
            "test_accuracy": [],
        }
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
    # new_test_data = CLEVRDataset(args.data_dir, mode="new_test")
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
    # new_test_dataloader = DataLoader(
    #     new_test_data,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    # )

    print("Train data samples: ", len(train_data))
    print("Test data samples: ", len(test_data))
    # print("New Test data samples: ", len(new_test_data))

    # --------- build models ------------------
    n_classes = train_data.get_n_classes()
    generator = Generator(n_classes, args.z_dim, 3)
    if args.network_type == "acgan+aux":
        discriminator = Discriminator(3, n_classes)
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
    generator_optimizer = optim.Adam(
        generator.parameters(),
        lr=args.generator_lr,
        betas=(args.momentum, 0.999),
    )
    discriminator_optimizer = optim.Adam(
        discriminator.parameters(),
        lr=args.discriminator_lr,
        betas=(args.momentum, 0.999),
    )

    try:
        loss_fn = nn.BCEWithLogitsLoss()
        number_of_batches = len(train_dataloader)
        # --------- train loop --------------------
        for epoch in trange(
            start_epoch,
            args.epoch_size,
            initial=start_epoch,
            total=args.epoch_size,
            desc="Current epoch",
        ):
            train_d_loss_list = []
            train_g_loss_list = []
            train_d_label_accuracy_list = []
            test_accuracy_list = []
            for batch, (real_images, real_labels) in enumerate(
                tqdm(train_dataloader, leave=False, desc="Current batch")
            ):
                real_images = real_images.to(args.device)
                real_labels = real_labels.to(args.device)

                batch_size = real_images.shape[0]

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

                generator.train()

                # -----------------
                #  Train Discriminator
                # -----------------

                discriminator_optimizer.zero_grad()

                z, sampled_labels = sample_noises(args, batch_size, n_classes)

                fake_images = generator(z, sampled_labels)

                # train with real images
                pred_real_validity, pred_real_labels = discriminator(
                    real_images
                )
                d_real_adversarial_loss = loss_fn(
                    pred_real_validity, real - args.label_smoothing_ratio
                )
                d_real_auxiliary_loss = loss_fn(pred_real_labels, real_labels)
                d_real_loss = d_real_adversarial_loss + d_real_auxiliary_loss

                # train with fake images
                pred_fake_validity, pred_sampled_labels = discriminator(
                    fake_images.detach()
                )
                d_fake_adversarial_loss = loss_fn(pred_fake_validity, fake)
                d_fake_auxiliary_loss = loss_fn(
                    pred_sampled_labels, sampled_labels
                )
                d_fake_loss = d_fake_adversarial_loss + d_fake_auxiliary_loss

                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                discriminator_optimizer.step()

                # -----------------
                #  Train Generator
                # -----------------

                generator_optimizer.zero_grad()

                pred_validity, pred_labels = discriminator(fake_images)
                g_adversarial_loss = loss_fn(pred_validity, real)
                g_auxiliary_loss = loss_fn(pred_labels, sampled_labels)
                g_loss = g_adversarial_loss + g_auxiliary_loss

                g_loss.backward()
                generator_optimizer.step()

                # -----------------
                #  Evaluate batches
                # -----------------

                # calculate label accuracy
                pred = torch.concat(
                    [
                        pred_real_labels.detach().cpu(),
                        pred_sampled_labels.detach().cpu(),
                    ],
                    dim=0,
                )
                gt = torch.concat(
                    [
                        real_labels.detach().cpu(),
                        sampled_labels.detach().cpu(),
                    ],
                    dim=0,
                )
                train_d_label_accuracy = calculate_accuracy_of_classifier(
                    pred, gt
                )
                train_d_label_accuracy_list.append(train_d_label_accuracy)

                # evaluator
                test_accuracy, generated_images = evaluate(
                    args, generator, test_dataloader, n_classes
                )
                train_d_loss_list.append(d_loss.item())
                train_g_loss_list.append(g_loss.item())
                test_accuracy_list.append(test_accuracy)
                save_images(
                    f"{test_images_dir}/epoch={epoch}-batch={batch}.jpg",
                    generated_images,
                )

                # test_accuracy, generated_images = evaluate(
                #     args, generator, test_dataloader
                # )
                # save_images(
                #     f"{test_images_dir}/epoch={epoch}-batch={batch}.jpg",
                #     generated_images,
                # )

                with open(train_record_path, "a") as train_record:
                    train_record.write(
                        (
                            "[Epoch %d/%d] [Batch %d/%d] [D loss: %.4f, real loss: %.4f, fake loss: %.4f]\n"
                            % (
                                epoch,
                                args.epoch_size,
                                batch,
                                number_of_batches,
                                d_loss.item(),
                                d_real_loss.item(),
                                d_fake_loss.item(),
                            )
                        )
                    )
                    train_record.write(
                        (
                            "[Epoch %d/%d] [Batch %d/%d] [G loss: %.4f, real/fake loss: %.4f, classifier loss: %.4f]\n"
                            % (
                                epoch,
                                args.epoch_size,
                                batch,
                                number_of_batches,
                                g_loss.item(),
                                g_adversarial_loss.item(),
                                g_auxiliary_loss.item(),
                            )
                        )
                    )
                    train_record.write(
                        (
                            "[Epoch %d/%d] [Batch %d/%d] [D classifier accuracy: %.4f%%] [Eval accuracy: %.4f%%]\n"
                            % (
                                epoch,
                                args.epoch_size,
                                batch,
                                number_of_batches,
                                train_d_label_accuracy * 100,
                                test_accuracy * 100,
                            )
                        )
                    )

            train_d_loss = np.array(train_d_loss_list).mean()
            train_g_loss = np.array(train_g_loss_list).mean()
            train_d_label_accuracy = np.array(
                train_d_label_accuracy_list
            ).mean()
            test_accuracy = np.array(test_accuracy_list).mean()

            metrics["train_d_loss"].append(train_d_loss)
            metrics["train_g_loss"].append(train_g_loss)
            metrics["train_d_label_accuracy"].append(train_d_label_accuracy)
            metrics["test_accuracy"].append(test_accuracy)

            with open(train_record_path, "a") as train_record:
                train_record.write(
                    (
                        "[Epoch %d/%d] [D loss: %.4f] [G loss: %.4f] [D classifier accuracy: %.4f%%] [Eval accuracy: %.4f%%]\n"
                        % (
                            epoch,
                            args.epoch_size,
                            train_d_loss,
                            train_g_loss,
                            train_d_label_accuracy * 100,
                            test_accuracy * 100,
                        )
                    )
                )

            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
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
    finally:
        show_curves(args.log_dir, metrics)


if __name__ == "__main__":
    main()
