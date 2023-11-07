import os
import platform
from dataclasses import dataclass

from HPAImageClassifier.HPAImageClassifier.config.constants import PROJECT_PATH


@dataclass
class LogsConfig:
    """This class contains all the logging configuration parameters"""

    VIZ_DIR: str = os.path.join(PROJECT_PATH, "logs")


@dataclass
class DatasetConfig:
    """This class contains all the dataset configuration parameters"""

    TRAIN_IMG_DIR: str = os.path.join(PROJECT_PATH, "datasets", "train")
    TEST_IMG_DIR: str = os.path.join(PROJECT_PATH, "datasets", "test")
    TRAIN_CSV: str = os.path.join(PROJECT_PATH, "datasets", "train.csv")
    TEST_CSV: str = os.path.join(PROJECT_PATH, "datasets", "sample_submission.csv")

    IMAGE_SIZE: tuple = (512, 512)
    CHANNELS: int = 3
    NUM_CLASSES: int = 28
    VALID_PCT: float = 0.2

    MEAN: tuple = (0.485, 0.456, 0.406)
    STD: tuple = (0.229, 0.224, 0.225)


@dataclass
class ModelTrainingConfig:
    """This class contains all the Model training configuration parameters"""

    MODEL_NAME: str = "resnet18"
    FREEZE_BACKBONE: bool = False
    BATCH_SIZE: int = 32
    NUM_EPOCHS: int = 30
    INIT_LR: float = 1e-4
    NUM_WORKERS: int = 0 if platform.system() == "Windows" else os.cpu_count()
    OPTIMIZER_NAME: str = "Adam"
    WEIGHT_DECAY: float = 1e-4
    USE_SCHEDULER: bool = True  # Use learning rate scheduler?
    SCHEDULER: str = "multi_step_lr"  # Name of the scheduler to use.
    METRIC_THRESH: float = 0.4
    LOSS_FN: str = "BCEWithLogitsLoss"
    MODEL_CKPT_PATH: str = os.path.join(PROJECT_PATH, "logs", "default", "version_0", "checkpoints", "epoch=29.ckpt")
