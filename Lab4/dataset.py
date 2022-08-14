import torch
import os
import csv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

default_transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


class BairRobotPushingDataset(Dataset):
    def __init__(self, args, mode="train", transform=default_transform):
        assert mode == "train" or mode == "test" or mode == "validate"
        self.root = "{}/{}".format(args.data_root, mode)
        self.seq_len = max(args.n_past + args.n_future, args.n_eval)

        self.transform = transform
        self.dirs = []
        for dir1 in os.listdir(self.root):
            for dir2 in os.listdir(os.path.join(self.root, dir1)):
                self.dirs.append(os.path.join(self.root, dir1, dir2))

    def __len__(self):
        return len(self.dirs)

    def _get_sequence(self, dir: str) -> torch.Tensor:
        image_seq = []
        for i in range(self.seq_len):
            fname = "{}/{}.png".format(dir, i)
            image_seq.append(self.transform(Image.open(fname)))
        image_seq = torch.stack(image_seq)

        return image_seq

    def _get_csv(self, dir: str) -> torch.Tensor:
        with open("{}/actions.csv".format(dir), newline="") as csvfile:
            rows = csv.reader(csvfile)
            actions = []
            for i, row in enumerate(rows):
                if i == self.seq_len:
                    break
                action = [float(value) for value in row]
                actions.append(torch.tensor(action))

            actions = torch.stack(actions)

        with open(
            "{}/endeffector_positions.csv".format(dir), newline=""
        ) as csvfile:
            rows = csv.reader(csvfile)
            positions = []
            for i, row in enumerate(rows):
                if i == self.seq_len:
                    break
                position = [float(value) for value in row]
                positions.append(torch.tensor(position))
            positions = torch.stack(positions)

        return torch.concat((actions, positions), dim=1)

    def __getitem__(self, index: int):
        dir = self.dirs[index]
        sequence = self._get_sequence(dir)
        condition = self._get_csv(dir)
        return sequence, condition


if __name__ == "__main__":
    train_data = BairRobotPushingDataset("train")
    train_loader = DataLoader(
        train_data,
        num_workers=3,
        batch_size=10,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
    )
    train_iterator = iter(train_loader)
    data = next(train_iterator)
    print("Sequence size:", data[0].size(), "Condition size:", data[1].size())
    print(data[1][0][0])
