"""
This script is used to visualize the model architecture and the dataset.
"""
import logging

from matplotlib import pyplot as plt
from torchinfo import summary
from torchvision.utils import make_grid

from HPAImageClassifier.HPAImageClassifier.config.config import DatasetConfig, LogsConfig, ModelTrainingConfig
from HPAImageClassifier.HPAImageClassifier.data.DataModule import HPADataModule
from HPAImageClassifier.HPAImageClassifier.model.model import get_model
from HPAImageClassifier.HPAImageClassifier.utils.utils import denormalize

data_module = HPADataModule(num_classes=DatasetConfig.NUM_CLASSES, batch_size=32, num_workers=0)

# Download dataset.
data_module.prepare_data()

# Split dataset into training, validation set.
data_module.setup()

model = get_model(
    model_name=ModelTrainingConfig.MODEL_NAME,
    num_classes=DatasetConfig.NUM_CLASSES,
    freeze_backbone=False,
)

model_info = summary(
    model,
    input_size=(1, DatasetConfig.CHANNELS, *DatasetConfig.IMAGE_SIZE[::-1]),
    depth=2,
    device="cpu",
    col_names=["output_size", "num_params", "trainable"],
)
logging.info(model_info)

# Get the validation data loader.
valid_loader = data_module.val_dataloader()
plt.figure(figsize=(15, 15))
for X, y in valid_loader:
    images = denormalize(X, mean=DatasetConfig.MEAN, std=DatasetConfig.STD).permute(0, 2, 3, 1).numpy()
    targets = y.numpy()

    for i in range(1, 26):
        plt.subplot(5, 5, i)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])

        target = " ".join([str(idx) for idx, cls_id in enumerate(targets[i]) if cls_id])
        plt.title(f"{target}", fontsize=12)

    plt.suptitle("Dataset Samples", fontsize=18)
    plt.tight_layout()
    plt.savefig(LogsConfig.VIZ_DIR + "/figs/medical_images.png", bbox_inches="tight")
    plt.close()
    break


# Visualize the grid of images.
data_module.batch_size = 132
data_module.resize_to = (128, 128)
valid_loader = data_module.val_dataloader()

batch = next(iter(valid_loader))
images = denormalize(batch[0], mean=DatasetConfig.MEAN, std=DatasetConfig.STD)
plt.figure(figsize=(32, 32))
grid_img = make_grid(images, nrow=22, padding=5, pad_value=1.0)

plt.imshow(grid_img.permute(1, 2, 0))
plt.axis("off")
plt.savefig(LogsConfig.VIZ_DIR + "/figs/medical_images_grid.png", bbox_inches="tight")
plt.tight_layout()
plt.close()
