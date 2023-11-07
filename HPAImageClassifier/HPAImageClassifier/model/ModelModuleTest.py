import lightning.pytorch as pl

from HPAImageClassifier.HPAImageClassifier.model.model import get_model


class HPAModelModuleTest(pl.LightningModule):
    def get_model_test(self):
        model = get_model(
            model_name="resnet18",
            num_classes=28,
            freeze_backbone=False,
        )
        assert model.type == "torchvision.models.resnet.ResNet"
