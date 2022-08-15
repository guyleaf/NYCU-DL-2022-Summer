import argparse
import os
import random

import numpy as np
import torch
import torch.optim as optim
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import CVAE
from dataset import BairRobotPushingDataset, data_collate_fn
from schedulers import KLAnnealingScheduler, TeacherForcingScheduler
from utils import init_weights, kl_criterion, finn_eval_seq, show_curves

torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lr", default=0.002, type=float, help="learning rate"
    )
    parser.add_argument(
        "--batch_size", default=12, type=int, help="batch size"
    )
    parser.add_argument(
        "--log_dir", default="./logs/fp", help="base directory to save logs"
    )
    parser.add_argument(
        "--model_dir", default="", help="base directory to save models"
    )
    parser.add_argument(
        "--data_root",
        default="./data",
        help="root directory for data",
    )
    parser.add_argument(
        "--optimizer", default="adam", help="optimizer to train with"
    )
    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        help="momentum term for optimizer",
    )
    parser.add_argument(
        "--epoch_size", type=int, default=600, help="epoch size"
    )
    parser.add_argument(
        "--tfr", type=float, default=1.0, help="teacher forcing ratio (0 ~ 1)"
    )
    parser.add_argument(
        "--tfr_start_decay_epoch",
        type=int,
        default=0,
        help="The epoch that teacher forcing ratio become decreasing",
    )
    parser.add_argument(
        "--tfr_decay_step",
        type=float,
        default=0,
        help="The decay step size of teacher forcing ratio (0 ~ 1)",
    )
    parser.add_argument(
        "--tfr_lower_bound",
        type=float,
        default=0,
        help="The lower bound of teacher forcing ratio for scheduling teacher forcing ratio (0 ~ 1)",
    )
    parser.add_argument(
        "--kl_anneal_beta",
        type=float,
        default=0.6,
        help="The trade-off between minimizing reconstruction error and KL divergence",
    )
    parser.add_argument(
        "--kl_anneal_scheduler",
        default=False,
        action="store_true",
        help="use KL scheduler",
    )
    parser.add_argument(
        "--kl_anneal_cyclical",
        default=False,
        action="store_true",
        help="use cyclical mode, otherwise monotonic mode (if kl_anneal_scheduler is enabled)",
    )
    parser.add_argument(
        "--kl_anneal_ratio",
        type=float,
        default=0.5,
        help="The decay ratio of kl annealing (if kl_anneal_scheduler is enabled)",
    )
    parser.add_argument(
        "--kl_anneal_cycle",
        type=int,
        default=3,
        help="The number of cycle for kl annealing during training (if kl_anneal_scheduler is enabled and use cyclical mode)",
    )
    parser.add_argument("--seed", default=1, type=int, help="manual seed")
    parser.add_argument(
        "--n_past",
        type=int,
        default=2,
        help="number of frames to condition on",
    )
    parser.add_argument(
        "--n_future", type=int, default=10, help="number of frames to predict"
    )
    parser.add_argument(
        "--n_eval",
        type=int,
        default=30,
        help="number of frames to predict at eval time",
    )
    parser.add_argument(
        "--rnn_size",
        type=int,
        default=256,
        help="dimensionality of hidden layer",
    )
    parser.add_argument(
        "--posterior_rnn_layers", type=int, default=1, help="number of layers"
    )
    parser.add_argument(
        "--predictor_rnn_layers", type=int, default=2, help="number of layers"
    )
    parser.add_argument(
        "--c_dim", type=int, default=32, help="dimensionality of c_t"
    )
    parser.add_argument(
        "--z_dim", type=int, default=64, help="dimensionality of z_t"
    )
    parser.add_argument(
        "--g_dim",
        type=int,
        default=128,
        help="dimensionality of encoder output vector and decoder input vector",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="number of data loading threads",
    )
    parser.add_argument(
        "--last_frame_skip",
        action="store_true",
        help="if true, skip connections go between frame t and frame t+t rather than last ground truth frame",
    )

    args = parser.parse_args()
    return args


def train(
    model: CVAE,
    sequences: torch.Tensor,
    conditions: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    kl_beta: float,
    teacher_forcing_ratio: float,
    n_past: int,
    last_frame_skip: bool = False,
    device: str = "cuda",
):
    mse = 0
    kld = 0
    use_teacher_forcing = (
        True if random.random() < teacher_forcing_ratio else False
    )

    model.init_hiddens(device)
    loss = MSELoss(reduction="sum")
    for t, x_t in enumerate(sequences, start=1):
        update_skips = last_frame_skip or t < n_past

        if use_teacher_forcing or t < n_past:
            x_t_1 = sequences[t - 1]

        c = conditions[t - 1]
        x_t_pred, mu, logvar = model(x_t_1, c, x_t, update_skips)
        mse = mse + loss(x_t_pred, x_t)
        kld = kld + kl_criterion(mu, logvar)

        x_t_1 = x_t_pred

    loss = mse + kld * kl_beta
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    length_of_sequence = sequences.size(0)
    return (
        loss.detach().cpu().item() / length_of_sequence,
        mse.detach().cpu().item() / length_of_sequence,
        kld.detach().cpu().item() / length_of_sequence,
    )


@torch.no_grad()
def evaluate(
    model: CVAE,
    sequences: torch.Tensor,
    conditions: torch.Tensor,
    n_past: int,
    last_frame_skip: bool = False,
    device: str = "cuda",
):
    pred_sequences = []
    model.init_hiddens(device)
    for t, x in enumerate(sequences):
        update_skips = last_frame_skip or t < n_past

        if t < n_past:
            x_t = x

        c = conditions[t]
        x_t_pred = model.predict(x_t, c, update_skips)

        x_t = x_t_pred
        pred_sequences.append(x_t_pred.cpu().numpy())

    return np.stack(pred_sequences, axis=0)


def main():
    args = parse_args()
    assert args.n_past + args.n_future <= 30 and args.n_eval <= 30
    assert 0 <= args.tfr and args.tfr <= 1
    assert 0 <= args.tfr_start_decay_epoch
    assert 0 <= args.tfr_decay_step and args.tfr_decay_step <= 1

    if torch.cuda.is_available():
        print("Use gpu to train the model.")
        device = "cuda"
    else:
        print("Use cpu to train the model.")
        device = "cpu"

    if args.model_dir != "":
        # load model and continue training from checkpoint
        saved_model = torch.load("%s/model.pth" % args.model_dir)
        model_dir = args.model_dir
        epoch_size = args.epoch_size
        args = saved_model["args"]
        args.model_dir = model_dir
        args.log_dir = "%s/continued" % args.log_dir
        last_epoch = saved_model["last_epoch"]
    else:
        name = (
            "rnn_size=%d-predictor-posterior-rnn_layers=%d-%d-n_past=%d-n_future=%d-optimizer=%s-lr=%.4f-momentum=%.4f-c_dim=%d-g_dim=%d-z_dim=%d-last_frame_skip=%s"
            % (
                args.rnn_size,
                args.predictor_rnn_layers,
                args.posterior_rnn_layers,
                args.n_past,
                args.n_future,
                args.optimizer,
                args.lr,
                args.momentum,
                args.c_dim,
                args.g_dim,
                args.z_dim,
                args.last_frame_skip,
            )
        )

        args.log_dir = "%s/%s" % (args.log_dir, name)
        epoch_size = args.epoch_size
        last_epoch = 0

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs("%s/gen/" % args.log_dir, exist_ok=True)

    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if os.path.exists("./{}/train_record.txt".format(args.log_dir)):
        os.remove("./{}/train_record.txt".format(args.log_dir))

    print(args)

    with open(
        "./{}/train_record.txt".format(args.log_dir), "a"
    ) as train_record:
        train_record.write("args: {}\n".format(args))

    # --------- load a dataset ------------------------------------
    training_data = BairRobotPushingDataset(args, "train")
    validation_data = BairRobotPushingDataset(args, "validate")
    training_loader = DataLoader(
        training_data,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        collate_fn=data_collate_fn,
    )
    validation_loader = DataLoader(
        validation_data,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        collate_fn=data_collate_fn,
    )

    # ------------ build the models  --------------

    if args.model_dir != "":
        model = saved_model["network"]
    else:
        model = CVAE(
            batch_size=args.batch_size,
            condition_embedding_input_features=7,
            condition_embedding_output_features=args.c_dim,
            encoder_input_channels=3,
            encoder_output_channels=args.g_dim,
            rnn_hidden_size=args.rnn_size,
            predictor_rnn_layers=args.predictor_rnn_layers,
            posterior_rnn_output_size=args.z_dim,
            posterior_rnn_layers=args.posterior_rnn_layers,
        )
        model.apply(init_weights)

    # --------- transfer to device ------------------------------------
    model.to(device)

    # ---------------- optimizers ----------------
    if args.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, betas=(args.momentum, 0.999)
        )
    elif args.optimizer == "rmsprop":
        optimizer = optim.RMSprop(
            model.parameters(), lr=args.lr, momentum=args.momentum
        )
    elif args.optimizer == "sgd":
        args.optimizer = optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum
        )
    else:
        raise ValueError("Unknown optimizer: %s" % args.optimizer)

    if args.kl_anneal_scheduler:
        if args.kl_anneal_cyclical:
            kl_scheduler = KLAnnealingScheduler(
                epoch_size=epoch_size,
                num_of_cycles=args.kl_anneal_cycle,
                ratio=args.kl_anneal_ratio,
                initial_epoch=last_epoch,
            )
        else:
            kl_scheduler = KLAnnealingScheduler(
                epoch_size=epoch_size,
                num_of_cycles=1,
                ratio=args.kl_anneal_ratio,
                initial_epoch=last_epoch,
            )
    else:
        kl_scheduler = None
        kl_beta = args.kl_anneal_beta

    teacher_forcing_scheduler = TeacherForcingScheduler(
        ratio=args.tfr,
        start_decay_epoch=args.tfr_start_decay_epoch,
        decay_step=args.tfr_decay_step,
        min_ratio=args.tfr_lower_bound,
        initial_epoch=last_epoch,
    )

    # --------- training loop ------------------------------------

    number_of_train_data = len(training_data)
    best_val_psnr = 0
    kl_weights = []
    tfrs = []
    train_losses = []
    average_psnrs = []
    for epoch in tqdm(range(last_epoch, last_epoch + epoch_size)):
        total_loss = 0
        total_mse = 0
        total_kld = 0

        teacher_forcing_ratio = teacher_forcing_scheduler.get_ratio()
        if kl_scheduler is not None:
            kl_beta = kl_scheduler.get_beta()

        model.train()
        for sequences, conditions in training_loader:
            sequences = sequences.to(device)
            conditions = conditions.to(device)

            loss, mse, kld = train(
                model,
                sequences,
                conditions,
                optimizer,
                kl_beta,
                teacher_forcing_ratio,
                args.n_past,
                args.last_frame_skip,
                device,
            )

            total_loss += loss
            total_mse += mse
            total_kld += kld

        teacher_forcing_scheduler.step()
        if kl_scheduler is not None:
            kl_scheduler.step()

        total_loss /= number_of_train_data
        total_mse /= number_of_train_data
        total_kld /= number_of_train_data

        kl_weights.append(kl_beta)
        tfrs.append(teacher_forcing_ratio)
        train_losses.append(total_loss)

        with open(
            "./{}/train_record.txt".format(args.log_dir), "a"
        ) as train_record:
            train_record.write(
                (
                    "[epoch: %02d] loss: %.5f | mse loss: %.5f | kld loss: %.5f\n"
                    % (
                        epoch,
                        total_loss,
                        total_mse,
                        total_kld,
                    )
                )
            )

        model.eval()

        if epoch % 5 == 0:
            n_past = args.n_past
            psnr_list = []
            for (
                val_sequences,
                val_conditions,
            ) in validation_loader:
                val_sequences = val_sequences.to(device)
                val_conditions = val_conditions.to(device)
                pred_sequences = evaluate(
                    model,
                    val_sequences,
                    val_conditions,
                    n_past,
                    args.last_frame_skip,
                    device,
                )
                _, _, psnr = finn_eval_seq(
                    val_sequences[n_past:].cpu().numpy(),
                    pred_sequences[n_past:],
                )
                psnr_list.append(psnr)
            ave_psnr = np.mean(np.concatenate(psnr_list))

            average_psnrs.append(ave_psnr)

            with open(
                "./{}/train_record.txt".format(args.log_dir), "a"
            ) as train_record:
                train_record.write(
                    (
                        "====================== validate psnr = {:.5f} ========================\n".format(
                            ave_psnr
                        )
                    )
                )

            if ave_psnr > best_val_psnr:
                best_val_psnr = ave_psnr
                # save the model
                torch.save(
                    {
                        "network": model,
                        "args": args,
                        "last_epoch": epoch,
                    },
                    "%s/model.pth" % args.log_dir,
                )

    show_curves(args.log_dir, kl_weights, tfrs, train_losses, average_psnrs)


if __name__ == "__main__":
    main()
