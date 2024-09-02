from typing import Any

import torch
from lightning_fabric import seed_everything
from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10, CIFAR100

from vish.lightning.data.common import (
    PATH_DATASETS,
    NUM_WORKERS,
    CIFAR100_FINE_2_BROAD_MAP,
    CIFAR_10_FINE_2_BROAD_MAP,
)

BATCH_SIZE = 32 if torch.cuda.is_available() else 4


class CIFAR10MultiLabelDataset(CIFAR10):
    def __init__(self, is_test: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fine_to_broad = CIFAR_10_FINE_2_BROAD_MAP
        self.is_test = is_test

    def _broad(self, idx):
        return self.fine_to_broad[idx]

    def __len__(self):
        if self.is_test:
            return 8192 if self.train else 4096
        return super().__len__()

    def __getitem__(self, index: int) -> tuple[Any, Any, int | list[int]]:
        img_tensor, fine_label = super().__getitem__(index)
        return img_tensor, fine_label, self.get_broad_label(fine_label)

    def get_broad_label(self, fine_label):
        return self._broad(fine_label)


class CIFAR100MultiLabelDataset(CIFAR100):
    def __init__(self, is_test: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_test = is_test
        self.fine_to_broad = CIFAR100_FINE_2_BROAD_MAP

    def get_broad_label(self, fine_label: int):
        return self.fine_to_broad[fine_label]

    def _broad(self, idx):
        return self.fine_to_broad[idx]

    def __len__(self):
        if self.is_test:
            return 8192 if self.train else 4096
        return super().__len__()

    def __getitem__(self, index: int) -> tuple[Any, Any, int | list[int]]:
        img_tensor, fine_label = super().__getitem__(index)
        return img_tensor, fine_label, self.get_broad_label(fine_label)


class CIFAR10MultiLabelDataModule(LightningDataModule):
    def __init__(
        self,
        is_test,
        train_transform,
        val_transform,
    ):
        super().__init__()
        self.cifar_val = None
        self.cifar_train = None
        self.is_test = is_test
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.data_dir = PATH_DATASETS
        self.batch_size = BATCH_SIZE
        self.num_workers = NUM_WORKERS

    def prepare_data(self):
        # Download data
        CIFAR10MultiLabelDataset(
            self.is_test, root=self.data_dir, train=True, download=True
        )
        CIFAR10MultiLabelDataset(
            self.is_test, root=self.data_dir, train=False, download=True
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            cifar_full = CIFAR10MultiLabelDataset(
                self.is_test,
                root=self.data_dir,
                train=True,
                transform=self.train_transform,
            )

            # use 10% of training data for validation
            train_set_size = int(len(cifar_full) * 0.9)
            valid_set_size = len(cifar_full) - train_set_size

            seed_everything(42)

            self.cifar_train, self.cifar_val = random_split(
                cifar_full, [train_set_size, valid_set_size]
            )
            # self.cifar_train = cifar_full

        if stage == "test" or stage is None:
            self.cifar_test = CIFAR10MultiLabelDataset(
                self.is_test,
                root=self.data_dir,
                train=False,
                transform=self.val_transform,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.cifar_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.cifar_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.cifar_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class CIFAR100MultiLabelDataModule(CIFAR10MultiLabelDataModule):
    """
    DataModule for CIFAR 100 with 20 broad classes and 100 fine classes
    """

    def __init__(
        self,
        is_test,
        train_transform,
        val_transform,
    ):
        super().__init__(is_test, train_transform, val_transform)
        self.cifar_val = None
        self.cifar_train = None
        self.is_test = is_test
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.data_dir = PATH_DATASETS
        self.batch_size = BATCH_SIZE
        self.num_workers = NUM_WORKERS

    def prepare_data(self):
        # Download data
        CIFAR100MultiLabelDataset(
            self.is_test, root=self.data_dir, train=True, download=True
        )
        CIFAR100MultiLabelDataset(
            self.is_test, root=self.data_dir, train=False, download=True
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            cifar_full = CIFAR100MultiLabelDataset(
                self.is_test,
                root=self.data_dir,
                train=True,
                transform=self.train_transform,
            )

            # use 10% of training data for validation
            train_set_size = int(len(cifar_full) * 0.9)
            valid_set_size = len(cifar_full) - train_set_size

            seed_everything(42)

            self.cifar_train, self.cifar_val = random_split(
                cifar_full, [train_set_size, valid_set_size]
            )
            # self.cifar_train = cifar_full

        if stage == "test" or stage is None:
            self.cifar_test = CIFAR100MultiLabelDataset(
                self.is_test,
                root=self.data_dir,
                train=False,
                transform=self.val_transform,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.cifar_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.cifar_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.cifar_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
