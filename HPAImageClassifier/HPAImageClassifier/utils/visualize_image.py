"""
Visualize images from the dataset.
"""
import glob
import os
import zipfile
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image

from HPAImageClassifier.HPAImageClassifier.config.config import LogsConfig
from HPAImageClassifier.HPAImageClassifier.config.constants import LABELS_REVERSE

# Download sample images with only one class per image.
URL = r"https://www.dropbox.com/scl/fi/caktugsvdbhnjoju5j78q/per_class_imgs.zip?rlkey=fu7q7fgpbrclxldq0xq9r2leb&dl=1"

file = requests.get(URL)
with open(LogsConfig.VIZ_DIR + "/test/per_class_imgs.zip", "wb") as f:
    f.write(file.content)

with zipfile.ZipFile(LogsConfig.VIZ_DIR + "/test/per_class_imgs.zip") as f:
    f.extractall(LogsConfig.VIZ_DIR + "/test/")

images = []
cls_names = []

for directory, _, files in os.walk(LogsConfig.VIZ_DIR + "/test/per_class_imgs"):
    if not len(files):
        continue

    class_name = os.path.split(directory)[-1]
    class_images = glob(directory + os.sep + "*.png")

    file = np.random.choice(class_images, size=1, replace=False)[0]

    image = Image.open(file).convert("RGB")
    image = np.asarray(image.resize((512, 512)))

    images.append(image)
    cls_names.append(class_name)

plt.figure(figsize=(20, 20))

for idx, (image, cls_name) in enumerate(zip(images, cls_names), 1):
    plt.subplot(5, 5, idx)
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"{LABELS_REVERSE[cls_name]} {cls_name}", fontsize=18)

plt.tight_layout()
plt.savefig(LogsConfig.VIZ_DIR + "/figs/per_class_images.png")
