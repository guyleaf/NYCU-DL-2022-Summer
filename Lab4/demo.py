import argparse
import torch
import numpy as np
import os

from dataset import BairRobotPushingDataset, data_collate_fn
from torch.utils.data import DataLoader

from models import CVAE
from utils import finn_eval_seq


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", default=12, type=int, help="batch size"
    )
    parser.add_argument(
        "--output_dir",
        default="./demo",
        help="base directory to save demo result",
    )
    parser.add_argument(
        "--model_file", default="", help="base directory to save models"
    )
    parser.add_argument(
        "--data_root",
        default="./data",
        help="root directory for data",
    )
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


@torch.no_grad()
def sample(
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
        x_t_pred = model.sample(x_t, c, update_skips)

        x_t = x_t_pred
        pred_sequences.append(x_t_pred.cpu().numpy())

    return np.stack(pred_sequences, axis=0)


@torch.no_grad()
def sample_from_posterior(
    model: CVAE,
    sequences: torch.Tensor,
    conditions: torch.Tensor,
    n_past: int,
    last_frame_skip: bool = False,
    device: str = "cuda",
):
    pred_sequences = [sequences[0].cpu().numpy()]
    model.init_hiddens(device)
    for t, x_t in enumerate(sequences, start=1):
        update_skips = last_frame_skip or t < n_past

        if t < n_past:
            x_t_1 = sequences[t - 1]

        c = conditions[t - 1]
        x_t_pred = model.sample_from_posterior(x_t_1, c, x_t, update_skips)

        x_t_1 = x_t_pred
        pred_sequences.append(x_t_pred.cpu().numpy())

    return np.stack(pred_sequences, axis=0)


def main():
    args = parse_args()
    assert args.n_past + args.n_future <= 30
    assert args.model_file != ""

    if torch.cuda.is_available():
        print("Use gpu to train the model.")
        device = "cuda"
    else:
        print("Use cpu to train the model.")
        device = "cpu"

    saved_model = torch.load(args.model_file)
    network = saved_model["network"]

    os.makedirs(args.output_dir, exist_ok=True)

    print(args)

    # --------- load dataset ------------------------------------

    testing_data = BairRobotPushingDataset(
        args.data_root,
        args.n_past + args.n_future,
        256,
        "test",
    )
    testing_loader = DataLoader(
        testing_data,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        collate_fn=data_collate_fn,
    )

    print("Testing data samples: ", len(testing_data))

    # ------------ load the model  --------------

    hyper_parameters = network["hyper_parameters"]
    hyper_parameters["batch_size"] = args.batch_size
    model = CVAE(**hyper_parameters)
    model.load_state_dict(network["parameters"])

    model.to(device)

    # ------------ get PSNR score  --------------

    model.eval()
    n_past = args.n_past
    psnr_list = []
    for (
        test_sequences,
        test_conditions,
    ) in testing_loader:
        test_sequences = test_sequences.to(device)
        test_conditions = test_conditions.to(device)
        pred_sequences = sample(
            model,
            test_sequences,
            test_conditions,
            n_past,
            args.last_frame_skip,
            device,
        )
        _, _, psnr = finn_eval_seq(
            test_sequences[n_past:].cpu().numpy(),
            pred_sequences[n_past:],
        )
        psnr_list.append(psnr)
    ave_psnr = np.mean(np.concatenate(psnr_list))

    print("Aveage PSNR:", ave_psnr)

    # # ------------ sample 100 sequences  --------------

    # ssim_list = []
    # sampled_sequences = []
    # sequence, condition = testing_loader[0]
    # sequence = sequence.to(device)
    # condition = condition.to(device)
    # for _ in range(100):
    #     pred_sequences = sample(
    #         model,
    #         sequence,
    #         condition,
    #         n_past,
    #         args.last_frame_skip,
    #         device,
    #     )
    #     _, ssim, _ = finn_eval_seq(
    #         test_sequences[n_past:].cpu().numpy(),
    #         pred_sequences[n_past:],
    #     )
    #     # sequence length, 1, x, x, x
    #     sampled_sequences.append(pred_sequences.squeeze())
    #     ssim_list.append(ssim)

    # # ------------ sample sequence from posterior  --------------

    # posterior_sequences = sample_from_posterior(
    #     model,
    #     sequence,
    #     condition,
    #     n_past,
    #     args.last_frame_skip,
    #     device,
    # ).squeeze()

    # # ------------ find the sequence with the best SSIM score and randomly pick 3 sequences  --------------

    # index = np.argmax(ssim_list)
    # best_sequence = sampled_sequences[index]
    # sampled_sequences = np.delete(sampled_sequences, index)
    # random_sampled_sequences = sampled_sequences[
    #     np.random.choice(len(sampled_sequences), 3)
    # ]

    # # ------------ make gif  --------------

    # make_gif(
    #     sequence.cpu().numpy().squeeze(),
    #     posterior_sequences,
    #     best_sequence,
    #     random_sampled_sequences,
    #     f"{args.output_dir}/demo.gif",
    # )

    # # ------------ export sequence  --------------

    # show_sequence(best_sequence, f"{args.output_dir}/demo.png")


if __name__ == "__main__":
    main()
