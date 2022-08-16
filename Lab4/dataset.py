import torch
import numpy as np
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
    def __init__(
        self,
        root_path,
        sequence_length,
        data_samples=1,
        mode="train",
        transform=default_transform,
    ):
        assert root_path != ""
        assert sequence_length > 1 and data_samples > 0
        assert mode == "train" or mode == "test" or mode == "validate"
        self.root = f"{root_path}/{mode}"
        self.seq_len = sequence_length

        self.transform = transform
        self.dirs = []
        for dir1 in os.listdir(self.root):
            for dir2 in os.listdir(os.path.join(self.root, dir1)):
                self.dirs.append(os.path.join(self.root, dir1, dir2))

        if mode != "test":
            data_samples = min(data_samples, len(self.dirs))
            self.dirs = (np.random.choice(self.dirs, data_samples)).tolist()

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


def data_collate_fn(data):
    data = torch.utils.data.default_collate(data)
    return data[0].transpose(1, 0), data[1].transpose(1, 0)


if __name__ == "__main__":
    train_data = BairRobotPushingDataset("train")
    train_loader = DataLoader(
        train_data,
        num_workers=3,
        batch_size=10,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        collate_fn=data_collate_fn,
    )

    for data in train_loader:
        print(
            "Sequence size:", data[0].size(), "Condition size:", data[1].size()
        )
