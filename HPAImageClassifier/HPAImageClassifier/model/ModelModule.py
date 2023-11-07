"""
This module contains the model class that is used for training and inference.
"""

import lightning.pytorch as pl
import torch
import torch.nn as nn
from torchmetrics import MeanMetric
from torchmetrics.classification import MultilabelF1Score

from HPAImageClassifier.HPAImageClassifier.model.model import get_model


class HPAModelModule(pl.LightningModule):
    """
    A PyTorch Lightning Module for training and inference.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int = 28,
        freeze_backbone: bool = False,
        init_lr: float = 0.001,
        optimizer_name: str = "Adam",
        weight_decay: float = 1e-4,
        use_scheduler: bool = False,
        f1_metric_threshold: float = 0.4,
    ):
        super().__init__()

        # Save the arguments as hyperparameters.
        self.save_hyperparameters()

        # Loading model using the function defined above.
        self.model = get_model(
            model_name=self.hparams.model_name,
            num_classes=self.hparams.num_classes,
            freeze_backbone=self.hparams.freeze_backbone,
        )

        # Initialize loss class.
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Initializing the required metric objects.
        self.mean_train_loss = MeanMetric()
        self.mean_train_f1score = MultilabelF1Score(
            num_labels=self.hparams.num_classes,
            average="macro",
            threshold=self.hparams.f1_metric_threshold,
        )
        self.mean_valid_loss = MeanMetric()
        self.mean_valid_f1score = MultilabelF1Score(
            num_labels=self.hparams.num_classes,
            average="macro",
            threshold=self.hparams.f1_metric_threshold,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        return self.model(x)

    def training_step(self, batch, *args, **kwargs):
        """Training step."""
        data, target = batch
        logits = self(data)
        loss = self.loss_fn(logits, target)

        self.mean_train_loss(loss, weight=data.shape[0])
        self.mean_train_f1score(logits, target)

        self.log("train/batch_loss", self.mean_train_loss, prog_bar=True)
        self.log("train/batch_f1score", self.mean_train_f1score, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        """
        Training epoch end.
        Computing and logging the training mean loss & mean f1.
        """
        self.log("train/loss", self.mean_train_loss, prog_bar=True)
        self.log("train/f1score", self.mean_train_f1score, prog_bar=True)
        self.log("step", self.current_epoch)

    def validation_step(self, batch, *args, **kwargs):
        """Validation step."""
        data, target = batch
        logits = self(data)
        loss = self.loss_fn(logits, target)

        self.mean_valid_loss.update(loss, weight=data.shape[0])
        self.mean_valid_f1score.update(logits, target)

    def on_validation_epoch_end(self):
        """
        validation epoch end.
        Computing and logging the validation mean loss & mean f1.
        """
        self.log("valid/loss", self.mean_valid_loss, prog_bar=True)
        self.log("valid/f1score", self.mean_valid_f1score, prog_bar=True)
        self.log("step", self.current_epoch)

    def configure_optimizers(self):
        """Configuring the optimizer and the learning rate scheduler."""
        optimizer = getattr(torch.optim, self.hparams.optimizer_name)(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.hparams.init_lr,
            weight_decay=self.hparams.weight_decay,
        )

        if self.hparams.use_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[
                    self.trainer.max_epochs // 2,
                ],
                gamma=0.1,
            )

            # The lr_scheduler_config is a dictionary that contains the scheduler
            # and its associated configuration.
            lr_scheduler_config = {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "name": "multi_step_lr",
            }
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

        else:
            return optimizer
