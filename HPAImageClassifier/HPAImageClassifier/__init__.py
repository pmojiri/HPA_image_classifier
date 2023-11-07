import os

import HPAImageClassifier.HPAImageClassifier.config
import HPAImageClassifier.HPAImageClassifier.data
import HPAImageClassifier.HPAImageClassifier.utils

VERSION = (0, 1, 0)
__version__ = ".".join(map(str, VERSION))

__all__ = ["config", "data", "model", "utils"]

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
