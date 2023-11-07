import torch

from HPAImageClassifier.HPAImageClassifier.config.config import DatasetConfig


def encode_label(labels: list, num_classes: int = 28) -> torch.Tensor:
    """This function converts the labels into one-hot encoded vectors"""
    target = torch.zeros(num_classes)
    for label in str(labels).split(" "):
        target[int(label)] = 1.0
    return target


def decode_target(
    target: list,
    text_labels: bool = False,
    threshold: float = 0.4,
    cls_labels: dict = None,
) -> str:
    """This function converts the labels from
    probabilities to outputs or string representations
    """

    result = []
    for i, x in enumerate(target):
        if x >= threshold:
            if text_labels:
                result.append(cls_labels[i] + "(" + str(i) + ")")
            else:
                result.append(str(i))
    return " ".join(result)


def de_normalize(tensors: torch.Tensor(), mean: tuple, std: tuple) -> torch.Tensor():
    """De normalizes image tensors using mean and std provided
    and clip values between 0 and 1
    This function is used for reversing the Normalization step perform during image preprocessing.
    Note the mean and std values must match the ones used."""

    for c in range(DatasetConfig.CHANNELS):
        tensors[:, c, :, :].mul_(std[c]).add_(mean[c])

    return torch.clamp(tensors, min=0.0, max=1.0)
