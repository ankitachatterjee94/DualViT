import json
import os
from functools import lru_cache
from typing import Any

import PIL
import torch
from pytorch_lightning import LightningDataModule
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

from vish.lightning.data.common import PATH_DATASETS, NUM_WORKERS

# Need different value for this in case of ImageNet
BATCH_SIZE = 64 if torch.cuda.is_available() else 4
IMAGE_SIZE = 224
ROOT_DIR = "data/imagenet"

train_data_dir = os.path.join(ROOT_DIR, "train")
val_data_dir = os.path.join(ROOT_DIR, "val")
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_tf = transforms.Compose(
    [
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        # transforms.RandAugment(num_ops=2, magnitude=9), #28
        transforms.ToTensor(),
        normalize,
    ]
)

val_tf = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        normalize,
    ]
)


class ImageNet1kMultilabelDataset(ImageFolder):
    def __init__(self, mode, is_test: bool, depth: int = 2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.is_test = is_test
        self.depth = depth

        f = self.read_label_json()
        self._set_and_check_depth(depth, f)

        self.label_tree = f["label_tree"]
        self.nodes_at_depth = f["depthwise_nodes"][str(depth)]["non_leaves"]
        self.num_broad_classes = len(self.nodes_at_depth)

    def _set_and_check_depth(self, depth, f):
        self.max_allowable_depth = int(f["meta"]["min_depth"])
        if depth >= self.max_allowable_depth:
            pass
            # raise ValueError(f"{depth} is not < {self.max_allowable_depth}")

    @staticmethod
    def read_label_json():
        with open("vish/lightning/data/imagenet_v3.json") as fp:
            f = json.load(fp)
        return f

    @lru_cache(maxsize=1100)
    def get_broad_label(self, fine_label: int):
        # print("Getting Broad")
        label = self._compute_broad(fine_label)
        # print("Got Broad node", label)
        return self.nodes_at_depth.get(label, -1)

    def _compute_broad(self, fine_label):
        fl_str = str(fine_label)
        labels_set = set(self.label_tree[fl_str])
        depth_nodes = set(self.nodes_at_depth.keys())
        # print("Label Set", labels_set)
        # print("Depth Node", depth_nodes)
        broad_label = labels_set.intersection(depth_nodes)
        if len(broad_label) > 1:
            raise ValueError(f"{broad_label} has more than 1 labels")
        # print("BLabel", broad_label)
        if len(broad_label) == 0:
            return -1
        l, *_ = broad_label
        return l

    def __len__(self):
        if self.is_test:
            # more images for Imagenet 1k
            return 32768 if self.mode == "train" else 16384
        return super().__len__()

    def __getitem__(self, index: int) -> tuple[Any, Any, int | list[int]]:
        # print("OK")
        img_tensor, fine_label = super().__getitem__(index)
        return img_tensor, fine_label, self.get_broad_label(fine_label)


class ImageNet1kMultiLabelDataModule(LightningDataModule):
    """
    ImageNet 1k data module with broad and fine labels
    """

    def __init__(
        self,
        is_test,
        depth,
        train_transform=train_tf,
        test_transform=val_tf,
    ):
        super().__init__()
        self._test = None
        self._val = None
        self._train = None
        self.is_test = is_test
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.depth = depth
        self.data_dir = PATH_DATASETS
        self.batch_size = BATCH_SIZE
        self.num_workers = NUM_WORKERS
        self.num_broad_classes = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self._train = ImageNet1kMultilabelDataset(
                "train", self.is_test, self.depth, train_data_dir, train_tf
            )
            self._val = ImageNet1kMultilabelDataset(
                "val", self.is_test, self.depth, val_data_dir, val_tf
            )
            self.num_broad_classes = self._train.num_broad_classes
        if stage == "test" or stage is None:
            self._val = ImageNet1kMultilabelDataset(
                "val", self.is_test, self.depth, val_data_dir, val_tf
            )
            self.num_broad_classes = self._train.num_broad_classes

    def train_dataloader(self):
        return DataLoader(
            self._train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )
    def test_dataloader(self):
        return DataLoader(
            self._val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )
