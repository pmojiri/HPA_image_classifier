"""
DataModuleTest for the Human Protein Atlas Image Classification.
"""
import lightning.pytorch as pl

from HPAImageClassifier.HPAImageClassifier.data.DataModule import HPADataModule


class HPADataModuleTest(pl.LightningDataModule):
    def test_data_module(self) -> None:
        data_module = HPADataModule(num_classes=28, batch_size=32, num_workers=0)

        # Download dataset.
        data_module.prepare_data()

        # Split dataset into training, validation set.
        data_module.setup()
        train_loader = data_module.train_dataloader()
        valid_loader = data_module.val_dataloader()

        assert train_loader.type == "torch.utils.data.dataloader.DataLoader"
        assert valid_loader.type == "torch.utils.data.dataloader.DataLoader"
