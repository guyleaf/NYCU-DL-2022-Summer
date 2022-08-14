import argparse
import os
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import CVAE
from dataset import BairRobotPushingDataset
from schedulers import KLAnnealingScheduler
from utils import (
    init_weights,
    kl_criterion,
    plot_pred,
    finn_eval_seq,
    evaluate,
)

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
        default="./data/processed_data",
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
        "--kl_anneal_cyclical",
        default=False,
        action="store_true",
        help="use cyclical mode",
    )
    parser.add_argument(
        "--kl_anneal_ratio",
        type=float,
        default=0.5,
        help="The decay ratio of kl annealing",
    )
    parser.add_argument(
        "--kl_anneal_cycle",
        type=int,
        default=3,
        help="The number of cycle for kl annealing during training (if use cyclical mode)",
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
    scheduler: KLAnnealingScheduler,
    teacher_forcing_ratio: float,
):
    mse = 0
    kld = 0
    use_teacher_forcing = (
        True if random.random() < teacher_forcing_ratio else False
    )

    model.init_hiddens()
    for i, x in enumerate(sequences):
        c = conditions[i]
        raise NotImplementedError

    beta = scheduler.get_beta()
    loss = mse + kld * beta
    loss.backward()

    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    length_of_sequence = x.size(0)
    return (
        loss.detach().cpu().numpy() / length_of_sequence,
        mse.detach().cpu().numpy() / length_of_sequence,
        kld.detach().cpu().numpy() / length_of_sequence,
    )


def main():
    args = parse_args()
    if torch.cuda.is_available():
        print("Use gpu to train the model.")
        device = "cuda"
    else:
        print("Use cpu to train the model.")
        device = "cpu"

    assert args.n_past + args.n_future <= 30 and args.n_eval <= 30
    assert 0 <= args.tfr and args.tfr <= 1
    assert 0 <= args.tfr_start_decay_epoch
    assert 0 <= args.tfr_decay_step and args.tfr_decay_step <= 1

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
    train_data = BairRobotPushingDataset(args, "train")
    validate_data = BairRobotPushingDataset(args, "validate")
    train_loader = DataLoader(
        train_data,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    validate_loader = DataLoader(
        validate_data,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
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

    if args.kl_anneal_cyclical:
        kl_scheduler = KLAnnealingScheduler(
            epoch_size=epoch_size,
            num_of_cycles=args.kl_anneal_cycle,
            ratio=args.kl_anneal_ratio,
        )
    else:
        kl_scheduler = KLAnnealingScheduler(
            epoch_size=epoch_size,
            num_of_cycles=1,
            ratio=args.kl_anneal_ratio,
        )

    # --------- training loop ------------------------------------

    number_of_train_data = len(train_data)
    best_val_psnr = 0
    for epoch in tqdm(range(last_epoch + 1, last_epoch + epoch_size + 1)):
        total_loss = 0
        total_mse = 0
        total_kld = 0

        model.train()
        for sequences, conditions in train_loader:
            sequences = sequences.to(device)
            conditions = conditions.to(conditions)
            loss, mse, kld = train(
                model, sequences, conditions, optimizer, kl_scheduler, args.tfr
            )
            total_loss += loss
            total_mse += mse
            total_kld += kld

        total_loss /= number_of_train_data
        total_mse /= number_of_train_data
        total_kld /= number_of_train_data

        if epoch >= args.tfr_start_decay_epoch:
            # Update teacher forcing ratio
            raise NotImplementedError

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

        # model.eval()

        # if epoch % 5 == 0:
        #     psnr_list = []
        #     for _ in range(len(validate_data) // args.batch_size):
        #         try:
        #             validate_seq, validate_cond = next(validate_iterator)
        #         except StopIteration:
        #             validate_iterator = iter(validate_loader)
        #             validate_seq, validate_cond = next(validate_iterator)

        #         pred_seq = pred(
        #             validate_seq, validate_cond, modules, args, device
        #         )
        #         _, _, psnr = finn_eval_seq(
        #             validate_seq[args.n_past :], pred_seq[args.n_past :]
        #         )
        #         psnr_list.append(psnr)

        #     ave_psnr = np.mean(np.concatenate(psnr_list))

        #     with open(
        #         "./{}/train_record.txt".format(args.log_dir), "a"
        #     ) as train_record:
        #         train_record.write(
        #             (
        #                 "====================== validate psnr = {:.5f} ========================\n".format(
        #                     ave_psnr
        #                 )
        #             )
        #         )

        #     if ave_psnr > best_val_psnr:
        #         best_val_psnr = ave_psnr
        #         # save the model
        #         torch.save(
        #             {
        #                 "encoder": encoder,
        #                 "decoder": decoder,
        #                 "frame_predictor": frame_predictor,
        #                 "posterior": posterior,
        #                 "args": args,
        #                 "last_epoch": epoch,
        #             },
        #             "%s/model.pth" % args.log_dir,
        #         )

        # if epoch % 20 == 0:
        #     try:
        #         validate_seq, validate_cond = next(validate_iterator)
        #     except StopIteration:
        #         validate_iterator = iter(validate_loader)
        #         validate_seq, validate_cond = next(validate_iterator)

        #     plot_pred(validate_seq, validate_cond, modules, epoch, args)


if __name__ == "__main__":
    main()
