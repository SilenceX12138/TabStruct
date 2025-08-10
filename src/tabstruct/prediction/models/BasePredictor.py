from abc import abstractmethod

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor, RichProgressBar, Timer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

from src.tabstruct.common import LOG_DIR, WANDB_PROJECT
from src.tabstruct.common.model.BaseModel import BaseLightningModule, BaseModel
from src.tabstruct.common.runtime.log.TerminalIO import TerminalIO

from .utils.evaluation import compute_all_metrics


class BasePredictor(BaseModel):

    def __init__(self, args):
        super().__init__(args)

    def predict(self, X: np.ndarray):
        """Predicts the labels of the test dataset (X).

        Args:
            X (np.ndarray): The test dataset. Default to be on CPU.
        """
        return self._predict(X)

    def predict_proba(self, X: np.ndarray):
        """Predicts the probabilities of the test dataset (X).

        Args:
            X (np.ndarray): The test dataset. Default to be on CPU.
        """
        return self._predict_proba(X)

    def feature_selection(self, X=None):
        """Selects the features.

        Args:
            X (np.ndarray, optional): The test dataset. Defaults to None.
        """
        return self._feature_selection(X)

    @abstractmethod
    def _predict(self, X: np.ndarray):
        """Predicts the labels of the test dataset (X).

        Args:
            X (np.ndarray): The test dataset. Default to be on CPU.
        """
        raise NotImplementedError("This method has to be implemented by the sub class")

    @abstractmethod
    def _predict_proba(self, X: np.ndarray):
        """Predicts the probabilities of the test dataset (X).

        Args:
            X (np.ndarray): The test dataset. Default to be on CPU.
        """
        raise NotImplementedError("This method has to be implemented by the sub class")

    @abstractmethod
    def _feature_selection(self, X=None):
        """Selects the features.

        Args:
            X (np.ndarray, optional): The test dataset. Defaults to None.
        """
        raise NotImplementedError("This method has to be implemented by the sub class")


class BaseSklearnPredictor(BasePredictor):

    def __init__(self, args):
        super().__init__(args)

    # ================================================================
    # =                                                              =
    # =                        General                               =
    # =                                                              =
    # ================================================================
    def _fit(self, data_module):
        self.model.fit(data_module.X_train, data_module.y_train)

    def _predict(self, X: np.ndarray):
        """Predicts the labels of the test dataset (X).

        Args:
            X (np.ndarray): The test dataset. Default to be on CPU.
        """
        y_pred = self.model.predict(X)

        return y_pred

    def _predict_proba(self, X: np.ndarray):
        """Predicts the probabilities of the test dataset (X).

        Args:
            X (np.adarray): The test dataset. Default to be on CPU.
        """
        if self.args.task == "classification":
            y_hat = self.model.predict_proba(X)
        else:
            y_hat = None

        return y_hat


class BaseLitPredictor(BasePredictor):
    def __init__(self, args):
        super().__init__(args)

    # ================================================================
    # =                                                              =
    # =                        General                               =
    # =                                                              =
    # ================================================================
    def _fit(self, data_module):
        # Build the trainer
        trainer = self.create_lit_trainer()

        # Train the model
        self.train_lit_model(data_module, trainer)

    def _predict(self, X):
        X = torch.tensor(X, dtype=torch.float32, device=self.model.device)
        y_hat = self.model(X)
        y_pred = y_hat if self.args.task == "regression" else torch.argmax(y_hat, dim=1)

        return y_pred.detach().cpu().numpy()

    def _predict_proba(self, X):
        X = torch.tensor(X, dtype=torch.float32, device=self.model.device)
        y_hat = self.model(X)
        if self.args.task != "regression":
            y_hat = F.softmax(y_hat, dim=1)

        return y_hat.detach().cpu().numpy()

    # ================================================================
    # =                                                              =
    # =                     Lightning-specific                       =
    # =                                                              =
    # ================================================================
    def create_lit_trainer(self):
        # ===== Prepare callbacks =====
        callbacks = []
        # === Stop single run after 2 hours ===
        timer_callback = Timer(duration="00:02:00:00")
        callbacks.append(timer_callback)
        # === Set up training metric ===
        mode_metric = "max" if self.args.metric_model_selection == "balanced_accuracy" else "min"
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"{LOG_DIR}/{WANDB_PROJECT}/{wandb.run.id}/",
            monitor=f"valid_metrics/{self.args.metric_model_selection}",
            mode=mode_metric,
            save_last=True,
            verbose=True,
        )
        callbacks.append(checkpoint_callback)
        # === Terminal style ===
        callbacks.append(RichProgressBar())
        # === Add callback functions for training ===
        if self.args.patience_early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor=f"valid_metrics/{self.args.metric_model_selection}",
                    mode=mode_metric,
                    patience=self.args.patience_early_stopping,
                )
            )
        # === Only monitor when wandb is enabled ===
        if not self.args.disable_wandb:
            callbacks.append(LearningRateMonitor(logging_interval="step"))

        # ===== Set up trainer =====
        trainer = L.Trainer(
            # Training
            max_steps=self.args.max_steps,
            gradient_clip_val=self.args.gradient_clip_val,
            # logging
            logger=self.args.wandb_logger,  # lightning launches multiple wandb runs in DDP, while sub-processes do not have args ---> do not affect run retrieval
            log_every_n_steps=self.args.log_every_n_steps,
            check_val_every_n_epoch=self.args.check_val_every_n_epoch,
            callbacks=callbacks,
            # miscellaneous
            accelerator=self.args.accelerator,
            detect_anomaly=self.args.debugging,
            deterministic=self.args.deterministic,
            devices=(
                "auto" if self.args.train_num_samples_processed > 100000 else 1
            ),  # use DDP only when training on large dataset
            # used for debugging, but it may crash when validation is not performed before showing results
            # fast_dev_run=True,
        )

        return trainer

    def train_lit_model(self, data_module, trainer):
        # === Train ===
        trainer.fit(self.model, data_module)

        # === Load the best model for evaluation ===
        checkpoint_path = trainer.checkpoint_callback.best_model_path
        self.load_from_checkpoint(checkpoint_path)

    def load_from_checkpoint(self, checkpoint_path):
        model_checkpoint = torch.load(checkpoint_path)
        weights = model_checkpoint["state_dict"]

        TerminalIO.print("Loading weights into model from {}.".format(checkpoint_path), color=TerminalIO.OKGREEN)
        missing_keys, unexpected_keys = self.model.load_state_dict(weights, strict=False)
        self.model.to(self.args.device)
        self.model.eval()

        TerminalIO.print("Missing keys:", color=TerminalIO.WARNING)
        TerminalIO.print(missing_keys, color=TerminalIO.WARNING)

        TerminalIO.print("Unexpected keys:", color=TerminalIO.WARNING)
        TerminalIO.print(unexpected_keys, color=TerminalIO.WARNING)


class BaseLightingPredictionModule(BaseLightningModule):
    """Base class for all PyTorch Lightning models used in the prediction task.
    This class provides a common interface for training, validation, and testing of lightning models.

    Note: This class is different from BaseLitPredictor, which is used for wrapping lightning models in this codebase.
    """

    def __init__(self, args):
        super().__init__(args)

    # ================================================================
    # =                                                              =
    # =                         General                              =
    # =                                                              =
    # ================================================================
    def _step(self, X, y_true):
        # Compute probabilities and predictions
        y_hat = self.torch_model(X)
        y_pred = y_hat if self.args.task == "regression" else torch.argmax(y_hat, dim=1)

        # Compute losses
        loss_dict = self.compute_loss(y_hat, y_true)

        return {
            "total_loss": loss_dict["total_loss"],
            "loss_dict": {k: v.detach().cpu().numpy() for k, v in loss_dict.items()},
            "y_true": y_true.detach().cpu().numpy(),
            "y_pred": y_pred.detach().cpu().numpy(),
            "y_hat": y_hat.detach().cpu().numpy(),
        }

    def _compute_metric(self, step_dict_list):
        y_true = np.concatenate([step_dict["y_true"] for step_dict in step_dict_list])[:, np.newaxis]
        y_pred = np.concatenate([step_dict["y_pred"] for step_dict in step_dict_list], axis=0)
        y_hat = np.concatenate([step_dict["y_hat"] for step_dict in step_dict_list], axis=0)

        metric_dict = compute_all_metrics(self.args, y_true, y_pred, y_hat)

        return metric_dict

    # ================================================================
    # =                                                              =
    # =                     Model-specific                           =
    # =                                                              =
    # ================================================================
    @abstractmethod
    def _create_torch_model(self):
        """Creates the PyTorch model.

        Args:
            args (Namespace): The arguments for the model.
        """
        raise NotImplementedError("This method has to be implemented by the sub class")

    @abstractmethod
    def _compute_loss(self, y_hat: torch.Tensor, y_true: torch.Tensor):
        raise NotImplementedError("This method has to be implemented by the sub class")
