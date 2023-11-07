"""
General script for training and running inference on a trained model.
"""
import argparse
import os
import sys

import lightning.pytorch as pl
import torch

from HPAImageClassifier.HPAImageClassifier.config.config import DatasetConfig, ModelTrainingConfig
from HPAImageClassifier.HPAImageClassifier.data.DataModule import HPADataModule
from HPAImageClassifier.HPAImageClassifier.model.ModelModule import HPAModelModule

sys.path.append(os.getcwd())
# Seed everything for reproducibility.)
pl.seed_everything(42, workers=True)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="HPA Classification")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="HPAImageClassifier/HPAImageClassifier/logs/default/version_0/checkpoints/epoch=29.ckpt",
        help="Path to the checkpoint file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize data module.
    data_module = HPADataModule(
        num_classes=DatasetConfig.NUM_CLASSES,
        valid_percentage=DatasetConfig.VALID_PCT,
        resize_to=DatasetConfig.IMAGE_SIZE,
        batch_size=ModelTrainingConfig.BATCH_SIZE,
        num_workers=ModelTrainingConfig.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # Load trained model.
    model = HPAModelModule.load_from_checkpoint(args.CKPT_PATH)

    # Initialize trainer class for inference.
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        enable_checkpointing=False,
        inference_mode=True,
    )

    # Run evaluation.
    data_module.setup()
    valid_loader = data_module.val_dataloader()
    trainer.validate(model=model, dataloaders=valid_loader)


if __name__ == "__main__":
    main()
