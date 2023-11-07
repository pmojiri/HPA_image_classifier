"""
This module contains a helper function to load and prepare any classification
"""
import logging

import torch
import torch.nn as nn
import torchvision


def get_model(model_name: str, num_classes: int, freeze_backbone: bool = True) -> torch.nn.Module:
    """
    A helper function to load and prepare any classification model
    available in Torchvision for transfer learning or fine-tuning.

    Args:
        model_name: Name of the model to be loaded.
        num_classes: Number of classes in the dataset.
        freeze_backbone: Whether to freeze the backbone of the model or not
            for fine-tuning.


    Returns:
        model: A PyTorch model with the output layer replaced with a new
            output layer with `num_classes` number of output nodes.

    """

    logging.info(f"Loading model: {model_name}")
    model = getattr(torchvision.models, model_name)(weights="DEFAULT")

    if freeze_backbone:
        # Set all layer to be non-trainable
        for param in model.parameters():
            param.requires_grad = False

    model_childs = [name for name, _ in model.named_children()]

    try:
        final_layer_in_features = getattr(model, f"{model_childs[-1]}")[-1].in_features
    except Exception:
        final_layer_in_features = getattr(model, f"{model_childs[-1]}").in_features

    new_output_layer = nn.Linear(in_features=final_layer_in_features, out_features=num_classes)

    try:
        getattr(model, f"{model_childs[-1]}")[-1] = new_output_layer
    except:
        setattr(model, model_childs[-1], new_output_layer)

    return model
