"""
DataModule for the Human Protein Atlas Image Classification competition.
"""
import logging
import os
import shutil
import subprocess
from itertools import chain

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as TF
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from HPAImageClassifier.HPAImageClassifier.config.config import DatasetConfig
from HPAImageClassifier.HPAImageClassifier.data.dataset import HPAImageDataset


class HPADataModule(pl.LightningDataModule):
    """
    DataModule for the Human Protein Atlas Image Classification competition.
    """

    def __init__(
        self,
        num_classes: int = 28,
        valid_percentage: float = 0.2,
        resize_to: tuple = (512, 512),
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle_validation: bool = False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.valid_percentage = valid_percentage
        self.resize_to = resize_to
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_validation = shuffle_validation

        self.test_ds = None
        self.valid_ds = None
        self.train_ds = None
        self.class_weights = None

        self.train_transforms = TF.Compose(
            [
                TF.RandomHorizontalFlip(),
                TF.RandomVerticalFlip(),
                TF.RandomAffine(
                    degrees=30,
                    translate=(0.01, 0.12),
                    shear=0.05,
                ),
                TF.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.1,
                ),
                TF.ToTensor(),
                TF.Normalize(DatasetConfig.MEAN, DatasetConfig.STD, inplace=True),
                TF.RandomErasing(inplace=True),
            ]
        )

        self.valid_transforms = TF.Compose(
            [
                TF.ToTensor(),
                TF.Normalize(DatasetConfig.MEAN, DatasetConfig.STD, inplace=True),
            ]
        )
        self.test_transforms = self.valid_transforms

    def prepare_data(self):
        """Download data if needed from Kaggle"""
        if not os.path.exists(os.path.join(DatasetConfig.TRAIN_CSV)):
            KAGGLE_DIR = os.path.join(os.path.expanduser("~"), ".kaggle")
            KAGGLE_JSON_PATH = os.path.join(KAGGLE_DIR, "kaggle.json")

            if not os.path.exists(KAGGLE_JSON_PATH):
                os.makedirs(KAGGLE_DIR, exist_ok=True)
                shutil.copyfile("kaggle.json", KAGGLE_JSON_PATH)
                os.chmod(KAGGLE_JSON_PATH, 0o600)

            logging.info("Downloading data from Kaggle...")
            subprocess.run(
                "kaggle competitions download -c human-protein-atlas-image-classification -p datasets",
                shell=True,
                check=True,
            )  # noqa
            subprocess.run(
                "unzip data/human-protein-atlas-image-classification.zip -d datasets",
                shell=True,
                check=True,
            )  # noqa

    def setup(self, stage=None):
        """Split data into train, validation and test sets"""
        np.random.seed(42)
        data_df = pd.read_csv(DatasetConfig.TRAIN_CSV)
        msk = np.random.rand(len(data_df)) < (1.0 - self.valid_percentage)
        train_df = data_df[msk].reset_index()
        valid_df = data_df[~msk].reset_index()

        train_labels = list(chain.from_iterable([i.strip().split(" ") for i in train_df["Target"].values]))
        class_weights = compute_class_weight(
            "balanced",
            classes=list(range(self.num_classes)),
            y=[int(i) for i in train_labels],
        )
        self.class_weights = torch.tensor(class_weights)

        img_size = DatasetConfig.IMAGE_SIZE
        self.train_ds = HPAImageDataset(
            df_data=train_df,
            img_size=img_size,
            root_dir=DatasetConfig.TRAIN_IMG_DIR,
            transforms=self.train_transforms,
        )

        self.valid_ds = HPAImageDataset(
            df_data=valid_df,
            img_size=img_size,
            root_dir=DatasetConfig.TRAIN_IMG_DIR,
            transforms=self.valid_transforms,
        )

        test_df = pd.read_csv(DatasetConfig.TEST_CSV)
        self.test_ds = HPAImageDataset(
            df_data=test_df,
            img_size=img_size,
            root_dir=DatasetConfig.TEST_IMG_DIR,
            transforms=self.test_transforms,
            is_test=True,
        )

        logging.info(
            f"Number of images :: "
            f"Training: {len(self.train_ds)}, "
            f"Validation: {len(self.valid_ds)}, "
            f"Testing: {len(self.test_ds)}\n"
        )

    def train_dataloader(self):
        """Create training dataloader object"""
        train_loader = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return train_loader

    def val_dataloader(self):
        """Create validation dataloader object."""
        valid_loader = DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle_validation,
            num_workers=self.num_workers,
        )
        return valid_loader

    def test_dataloader(self):
        """Create test dataloader object."""
        test_loader = DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return test_loader
