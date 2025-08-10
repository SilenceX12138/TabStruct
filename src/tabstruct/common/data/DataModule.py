import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class DataModule(L.LightningDataModule):
    """DataModule for both numpy and torch tensors.

    1. NumPy arrays: data_module.X_train
    2. PyTorch dataloader: data_module.train_dataloader()

    """

    def __init__(
        self,
        args,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ):
        super().__init__()
        self.args = args

        # === Prepare numpy arrays ===
        self.X_train = X_train.astype(np.float32)
        self.y_train = y_train
        self.X_valid = X_valid.astype(np.float32)
        self.y_valid = y_valid
        self.X_test = X_test.astype(np.float32)
        self.y_test = y_test

        # === Prepare PyTorch datasets ===
        self.train_dataset = CustomPytorchDataset(args, self.X_train, self.y_train)
        self.valid_dataset = CustomPytorchDataset(args, self.X_valid, self.y_valid)
        self.test_dataset = CustomPytorchDataset(args, self.X_test, self.y_test)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
        )

    def val_dataloader(self):
        # dataloader with original samples
        return DataLoader(
            self.valid_dataset,
            batch_size=min(self.args.valid_num_samples_processed, self.args.batch_size),
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=min(self.args.test_num_samples_processed, self.args.batch_size),
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
        )


class CustomPytorchDataset(Dataset):
    """Custom PyTorch dataset with numpy arrays as input."""

    def __init__(self, args, X: np.ndarray, y: np.ndarray) -> None:
        # X, y are numpy
        super().__init__()

        self.X = torch.tensor(
            X,
            dtype=torch.float32,
            requires_grad=False,
        )
        self.y = torch.tensor(
            y,
            dtype=torch.float32 if args.task == "regression" else torch.long,
            requires_grad=False,
        )

    def __getitem__(self, index):
        x = self.X[index]
        y = self.y[index]

        return x, y

    def __len__(self):
        return len(self.X)
