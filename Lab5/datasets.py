import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils import read_json

DEFAULT_TRANSFORMS = transforms.Compose(
    [
        # transforms.RandomCrop(240),
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


class CLEVRDataset(Dataset):
    _images_path: Path
    _transforms: transforms.Compose
    _images: np.ndarray = None
    _labels: np.ndarray

    def __init__(
        self,
        root_path: str,
        mode: str = "train",
        transforms=DEFAULT_TRANSFORMS,
    ) -> None:
        super().__init__()
        assert root_path != ""
        assert mode == "train" or mode == "test" or mode == "new_test"
        root_path = Path(root_path)
        self._images_path = root_path / "images"
        self._transforms = transforms

        if mode == "train":
            images = []
            for image_name in os.listdir(self._images_path):
                if os.path.isfile(self._images_path / image_name):
                    images.append(image_name)
            self._images = np.array(images)
            labels = read_json(root_path / "train.json")
            labels = [labels[image] for image in self._images]
        else:
            if mode == "test":
                labels = read_json(root_path / "test.json")
            else:
                labels = read_json(root_path / "new_test.json")

        # convert objects to one-hot vector
        objects_map = read_json(root_path / "objects.json")
        number_of_objects = len(objects_map.keys())
        one_hot_vectors = []
        for label in labels:
            label = [objects_map[key] for key in label]
            one_hot_vectors.append(
                np.sum(
                    np.eye(number_of_objects, dtype=np.int64)[label],
                    axis=0,
                )
            )
        self._labels = np.array(one_hot_vectors)

    def get_n_classes(self):
        return self._labels.shape[1]

    def _get_image(self, index: int) -> torch.Tensor:
        image_name = self._images[index]
        image = Image.open(self._images_path / image_name).convert(mode="RGB")
        return self._transforms(image)

    def _get_label(self, index: int) -> torch.Tensor:
        label = self._labels[index]
        return torch.from_numpy(label).float()

    def __len__(self) -> int:
        return self._labels.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = []
        if self._images is not None:
            image = self._get_image(index)
        return image, self._get_label(index)


if __name__ == "__main__":
    data = CLEVRDataset("./data", mode="train")
    print("Data samples: ", len(data))
