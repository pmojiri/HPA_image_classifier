"""
Script for training and running evaluation on the best trained model.
"""
import argparse
import os
import sys

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from HPAImageClassifier.HPAImageClassifier.config.config import DatasetConfig, ModelTrainingConfig
from HPAImageClassifier.HPAImageClassifier.data.DataModule import HPADataModule
from HPAImageClassifier.HPAImageClassifier.model.ModelModule import HPAModelModule

sys.path.append(os.getcwd())

# Seed everything for reproducibility.
pl.seed_everything(42, workers=True)

torch.set_float32_matmul_precision('high')
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="HPA Image Classification")
    return parser.parse_args()


def main():
    parse_args()

    # Initialize data module.
    data_module = HPADataModule(
        num_classes=DatasetConfig.NUM_CLASSES,
        valid_percentage=DatasetConfig.VALID_PCT,
        resize_to=DatasetConfig.IMAGE_SIZE,
        batch_size=ModelTrainingConfig.BATCH_SIZE,
        num_workers=ModelTrainingConfig.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # Initialize model.
    model = HPAModelModule(
        model_name=ModelTrainingConfig.MODEL_NAME,
        num_classes=DatasetConfig.NUM_CLASSES,
        freeze_backbone=ModelTrainingConfig.FREEZE_BACKBONE,
        init_lr=ModelTrainingConfig.INIT_LR,
        optimizer_name=ModelTrainingConfig.OPTIMIZER_NAME,
        weight_decay=ModelTrainingConfig.WEIGHT_DECAY,
        use_scheduler=ModelTrainingConfig.USE_SCHEDULER,
        f1_metric_threshold=ModelTrainingConfig.METRIC_THRESH,
    )

    # Creating ModelCheckpoint callback.
    model_checkpoint = ModelCheckpoint(
        monitor="valid/f1score",
        mode="max",
        filename="ckpt_{epoch:03d}-vloss_{valid/loss:.4f}_vf1score_{valid/f1score:.4f}",
        auto_insert_metric_name=False,
    )

    # Creating a learning rate monitor callback.
    lr_rate_monitor = LearningRateMonitor(logging_interval="epoch")

    # Initializing the Trainer class object.
    trainer = pl.Trainer(
        accelerator="auto",  # Auto select accelerator (GPU, TPU, CPU).
        devices="auto",  # Auto select devices.
        strategy="auto",  # Auto select distributed backend.
        max_epochs=ModelTrainingConfig.NUM_EPOCHS,  # Number of epochs to train for.
        deterministic=True,  # Keep everything deterministic for reproducibility.
        enable_model_summary=False,  # Disable model summary.
        callbacks=[model_checkpoint, lr_rate_monitor],  # Add callbacks.
        precision="16",  # Use mixed precision training.
        logger=True,  # Use TensorBoard logger.
    )

    # Start training
    trainer.fit(model, data_module)

    # Run evaluation.
    model_path = model_checkpoint.best_model_path
    model = HPAModel.load_from_checkpoint(model_path)
    data_module.setup()
    valid_loader = data_module.val_dataloader()
    trainer.validate(model=model, dataloaders=valid_loader)


if __name__ == "__main__":
    main()
