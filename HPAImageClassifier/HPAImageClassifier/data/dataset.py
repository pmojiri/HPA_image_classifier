"""
Dataset for Human Protein Atlas Image Classification Challenge
"""
import os

import numpy as np
import pandas as pd
import torchvision
from PIL import Image
from torch.utils.data import Dataset

from HPAImageClassifier.HPAImageClassifier.utils.utils import encode_label


class HPAImageDataset(Dataset):
    """
    Dataset for Human Protein Atlas Image Classification Challenge
    """

    def __init__(
        self,
        df_data: pd.DataFrame,
        root_dir: str,
        img_size: tuple,
        transforms: torchvision.transforms = None,
        is_test: bool = False,
    ):
        self.df_data = df_data
        self.root_dir = root_dir
        self.img_size = img_size
        self.transforms = transforms
        self.is_test = is_test

    def load_image(self, file_name: str) -> Image:
        """
        Load four channels of an image and merge them into one image
        :param file_name: image file name
        :return: image in RGB format
        """
        R = np.array(Image.open(file_name + "_red.png"))
        G = np.array(Image.open(file_name + "_green.png"))
        B = np.array(Image.open(file_name + "_blue.png"))
        Y = np.array(Image.open(file_name + "_yellow.png"))
        image = np.stack((R + Y / 2, G + Y / 2, B), -1)

        return Image.fromarray(image.astype("uint8"), "RGB")

    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.df_data)

    def __getitem__(self, idx: int) -> tuple:
        row = self.df_data.loc[idx]
        img_id = row["Id"]

        img = self.load_image(self.root_dir + os.sep + str(img_id))
        img = img.resize(self.img_size, resample=3)
        img = self.transforms(img)

        if self.is_test:
            return img, img_id

        return img, encode_label(row["Target"])
