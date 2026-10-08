import lightning as L
import numpy as np
import torch
from tabcamel.data.dataset import TabularDataset
from torch.utils.data import DataLoader, Dataset


class DataModule(L.LightningDataModule):
    """DataModule for both numpy and torch tensors.

    1. NumPy arrays: data_module.X_train
    2. PyTorch dataloader: data_module.train_dataloader()

    """

    def __init__(
        self,
        args,
        train_set: TabularDataset,
        valid_set: TabularDataset,
        test_set: TabularDataset,
    ):
        super().__init__()
        # Save args
        self.args = args
        self.is_tensor = train_set.is_tensor

        # Store original dataframes
        self._X_train_df = train_set.X_df
        self._X_valid_df = valid_set.X_df
        self._X_test_df = test_set.X_df

        self._y_train_df = None
        self._y_valid_df = None
        self._y_test_df = None
        if args.task != "unsupervision":
            self._y_train_df = train_set.data_df[[train_set.target_col]]
            self._y_valid_df = valid_set.data_df[[valid_set.target_col]]
            self._y_test_df = test_set.data_df[[test_set.target_col]]

    def train_dataloader(self):
        return DataLoader(
            CustomPytorchDataset(self.args, self.X_train, self.y_train),
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            collate_fn=self.custom_collate_fn,
        )

    def val_dataloader(self):
        # dataloader with original samples
        return DataLoader(
            CustomPytorchDataset(self.args, self.X_valid, self.y_valid),
            batch_size=min(self.args.valid_num_samples_processed, self.args.batch_size),
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            collate_fn=self.custom_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            CustomPytorchDataset(self.args, self.X_test, self.y_test),
            batch_size=min(self.args.test_num_samples_processed, self.args.batch_size),
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            collate_fn=self.custom_collate_fn,
        )

    def custom_collate_fn(self, batch):
        """Custom collate function that handles None values in labels."""
        X_batch = torch.stack([item[0] for item in batch])
        y_batch = [item[1] for item in batch]
        indices_batch = torch.tensor([item[2] for item in batch], dtype=torch.long)

        # Check if all y values are None
        if all(y is None for y in y_batch):
            return X_batch, None, indices_batch
        else:
            # Stack non-None values
            return X_batch, torch.stack(y_batch), indices_batch

    # ================================================================
    # =                                                              =
    # =                      Properties                              =
    # =                                                              =
    # ================================================================
    @property
    def X_train_df(self):
        return self._X_train_df

    @X_train_df.setter
    def X_train_df(self, X_train_df):
        self._X_train_df = X_train_df

    @property
    def X_valid_df(self):
        return self._X_valid_df

    @X_valid_df.setter
    def X_valid_df(self, X_valid_df):
        self._X_valid_df = X_valid_df

    @property
    def X_test_df(self):
        return self._X_test_df

    @X_test_df.setter
    def X_test_df(self, X_test_df):
        self._X_test_df = X_test_df

    @property
    def y_train_df(self):
        return self._y_train_df

    @y_train_df.setter
    def y_train_df(self, y_train_df):
        self._y_train_df = y_train_df

    @property
    def y_valid_df(self):
        return self._y_valid_df

    @y_valid_df.setter
    def y_valid_df(self, y_valid_df):
        self._y_valid_df = y_valid_df

    @property
    def y_test_df(self):
        return self._y_test_df

    @y_test_df.setter
    def y_test_df(self, y_test_df):
        self._y_test_df = y_test_df

    @property
    def indices_train(self):
        return self._X_train_df.index.tolist()

    @property
    def indices_valid(self):
        return self._X_valid_df.index.tolist()

    @property
    def indices_test(self):
        return self._X_test_df.index.tolist()

    @property
    def X_train(self):
        return self._X_train_df.to_numpy().astype(np.float32) if self.is_tensor else self._X_train_df

    @property
    def X_valid(self):
        return self._X_valid_df.to_numpy().astype(np.float32) if self.is_tensor else self._X_valid_df

    @property
    def X_test(self):
        return self._X_test_df.to_numpy().astype(np.float32) if self.is_tensor else self._X_test_df

    @property
    def y_train(self):
        if self.args.task == "unsupervision":
            return None

        return self._get_tensor(self._y_train_df)

    @property
    def y_valid(self):
        if self.args.task == "unsupervision":
            return None
        return self._get_tensor(self._y_valid_df)

    @property
    def y_test(self):
        if self.args.task == "unsupervision":
            return None
        return self._get_tensor(self._y_test_df)

    def _get_tensor(self, data_df):
        """Convert DataFrame to tensor format if needed.

        Args:
            data_df: Input DataFrame
            dtype: Target numpy dtype (default: np.float32 for tensors)

        Returns:
            DataFrame or numpy array based on self.is_tensor
        """
        if not self.is_tensor:
            return data_df

        # Convert to numpy with specified dtype
        data = data_df.to_numpy()

        # Flatten if single column
        if data.shape[-1] == 1:
            data = data.flatten()

        return data


class CustomPytorchDataset(Dataset):
    """Custom PyTorch dataset with numpy arrays as input.

    - Uses torch.from_numpy() for memory efficiency (shares memory with numpy arrays)
    - Caches target dtype to avoid repeated conditional checks
    - Includes input validation
    """

    def __init__(self, args, X: np.ndarray, y: np.ndarray | None) -> None:
        super().__init__()

        # === Sanity checks ===
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be a numpy array, got {type(X)}")
        if y is not None and not isinstance(y, np.ndarray):
            raise TypeError(f"y must be a numpy array or None, got {type(y)}")
        if y is not None and len(X) != len(y):
            raise ValueError(f"X and y must have same length, got {len(X)} and {len(y)}")

        # === Prepare data tensors ===
        # Use from_numpy for memory efficiency (shares memory with numpy arrays)
        # Ensure arrays are contiguous for optimal performance
        X_contiguous = np.ascontiguousarray(X)
        self.X = torch.from_numpy(X_contiguous).to(torch.float32)

        # Ensure target array is contiguous
        self.has_targets = y is not None
        if self.has_targets:
            y_contiguous = np.ascontiguousarray(y)

            self.target_dtype = torch.float32 if args.task == "regression" else torch.long
            self.y = torch.from_numpy(y_contiguous).to(self.target_dtype)

    def __getitem__(self, index):
        X = self.X[index]

        y = None
        if self.has_targets:
            y = self.y[index]

        return X, y, index

    def __len__(self):
        return len(self.X)
