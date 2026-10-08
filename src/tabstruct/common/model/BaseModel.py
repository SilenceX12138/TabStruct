import inspect
from abc import abstractmethod

import lightning as L
import numpy as np
import optuna
import torch
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor, RichProgressBar, Timer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

from src.tabstruct.common import LOG_DIR, WANDB_PROJECT
from src.tabstruct.common.runtime.log.TerminalIO import TerminalIO


class BaseModel:
    """Basic interface for all models.

    All implemented models should inherit from this base class to provide a common interface.

    """

    # ================================================================
    # =                                                              =
    # =                       Initialisation                         =
    # =                                                              =
    # ================================================================
    def __init__(self, args):
        """Initializes the model. Within the sub class, the __init__() method needs to include:
        - Sanity check of the arguments (e.g., TabPFN does not support regression)
        - The definition of the model architecture (self.model)
        - The case-by-case definition of the model parameters (self.params, e.g., XGBoost's device setup)

        Args:
            args (argparse.Namespace): The arguments for the experiment.
        """
        self._args = args
        self.params = args.model_params

        # Model definition has to be implemented by the concrete model
        self.model = None

    # ================================================================
    # =                                                              =
    # =                       Top-level APIs                         =
    # =                                                              =
    # ================================================================
    def fit(self, data_module):
        """Fits the model to the training data.

        Args:
            data_module (DataModule): The data module containing the training data.
        """
        self._fit(data_module)

    def eval(self):
        """Set the model to evaluation/inference mode."""
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()

    def get_metadata(self):
        return {
            "name": self.__class__.__name__,
            "params": self.params,
        }

    @classmethod
    def define_params(cls, reg_test, trial=None, dev=False):
        if trial is not None:
            return cls._define_optuna_params(trial)
        elif reg_test:
            return cls._define_test_params()
        elif not dev:
            return cls._define_default_params()
        else:
            return cls._define_single_run_params()

    @classmethod
    def get_model_specific_scaler_config(cls):
        """Get the model-specific scaler configurations.
        Default configurations are provided for feature and target scalers.
        If needed, these configurations can be overridden in the model-specific implementation.

        Returns:
            dict: The model-specific scaler configurations.
        """
        try:
            scaler_config_dict = cls._get_model_specific_scaler_config()
        except NotImplementedError:
            scaler_config_dict = {
                "context": {},
                "feature_scaler": {},
                "target_scaler": {},
            }

        return scaler_config_dict

    # ================================================================
    # =                                                              =
    # =                     Common properties                        =
    # =                                                              =
    # ================================================================
    @property
    def name(self):
        return self.__class__.__name__

    # ================================================================
    # =                                                              =
    # =              Utils to implement in sub class                 =
    # =                                                              =
    # ================================================================
    @abstractmethod
    def _fit(self, data_module):
        raise NotImplementedError("This method has to be implemented by the sub class")

    @classmethod
    @abstractmethod
    def _define_default_params(cls):
        raise NotImplementedError("This method has to be implemented by the sub class")

    @classmethod
    @abstractmethod
    def _define_optuna_params(cls, trial: optuna.Trial):
        raise NotImplementedError("This method has to be implemented by the sub class")

    @classmethod
    @abstractmethod
    def _define_single_run_params(cls):
        raise NotImplementedError("This method has to be implemented by the sub class")

    @classmethod
    @abstractmethod
    def _define_test_params(cls):
        raise NotImplementedError("This method has to be implemented by the sub class")

    @classmethod
    @abstractmethod
    def _get_model_specific_scaler_config(cls):
        raise NotImplementedError("This method has to be implemented by the sub class")

    # ================================================================
    # =                       Properties                             =
    # ================================================================
    @property
    def args(self):
        return self._args

    @args.setter
    def args(self, value):
        self._args = value
        if self.model is not None and hasattr(self.model, "args"):
            self.model.args = value


class LitModelMixin:
    """Mixin class for all Lightning models."""

    # ================================================================
    # =                                                              =
    # =                         General                              =
    # =                                                              =
    # ================================================================
    def create_lit_trainer(self):
        # ===== Prepare callbacks =====
        callbacks = self._prepare_lit_trainer_callbacks()

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
        # === Create the torch model ===
        if not self.args.saved_checkpoint_path:
            self.model.create_torch_model()

        # === Train ===
        trainer.fit(self.model, data_module)

        # === Load the best model for evaluation ===
        checkpoint_path = trainer.checkpoint_callback.best_model_path
        self.load_from_checkpoint(checkpoint_path)

    @TerminalIO.trace_func
    def load_from_checkpoint(self, checkpoint_path):
        model_checkpoint = torch.load(checkpoint_path)
        weights = model_checkpoint["state_dict"]

        TerminalIO.print("Loading weights into model from {}.".format(checkpoint_path), color=TerminalIO.OKBLUE)
        missing_keys, unexpected_keys = self.model.load_state_dict(weights, strict=False)
        self.model.to(self.args.device)
        self.model.eval()

        TerminalIO.print("Missing keys:", color=TerminalIO.WARNING)
        TerminalIO.print(missing_keys, color=TerminalIO.WARNING)

        TerminalIO.print("Unexpected keys:", color=TerminalIO.WARNING)
        TerminalIO.print(unexpected_keys, color=TerminalIO.WARNING)

    # ================================================================
    # =                                                              =
    # =                         Utils                                =
    # =                                                              =
    # ================================================================
    def _prepare_lit_trainer_callbacks(self):
        callbacks = []
        # === Stop single run after 2 hours ===
        timer_callback = Timer(duration="00:02:00:00")
        callbacks.append(timer_callback)
        # === Set up training metric ===
        mode_metric = "max" if self.args.metric_early_stopping == "balanced_accuracy" else "min"
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"{LOG_DIR}/{WANDB_PROJECT}/{wandb.run.id}/",
            monitor=f"{self.args.split_early_stopping}_metrics/{self.args.metric_early_stopping}",
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
                    monitor=f"{self.args.split_early_stopping}_metrics/{self.args.metric_early_stopping}",
                    mode=mode_metric,
                    patience=self.args.patience_early_stopping,
                )
            )
        # === Only monitor when wandb is enabled ===
        if not self.args.disable_wandb:
            callbacks.append(LearningRateMonitor(logging_interval="step"))

        return callbacks


class BaseLightningModule(L.LightningModule):
    """Base class for all Lightning modules.

    This class provides a common interface for all Lightning modules used in the project.

    """

    def __init__(self, args):
        super().__init__()
        self._args = args
        self.torch_model = None
        self.optim_params = self._args.model_params["optimization"]

        self.training_step_output_list = []
        self.validation_step_output_list = []
        self.test_step_output_list = []

    # ================================================================
    # =                                                              =
    # =                         General                              =
    # =                                                              =
    # ================================================================
    def create_torch_model(self):
        """Creates the PyTorch model.

        Args:
            args (Namespace): The arguments for the model.
        """
        self.torch_model = self._create_torch_model()

    def forward(self, X):
        """Make LightningModule compatible with upper-level calls."""
        return self.torch_model(X)

    def step(self, X, y_true, index):
        func_signature = inspect.signature(self._step)
        if "index" in func_signature.parameters:
            step_dict = self._step(X, y_true, index=index)
        else:
            step_dict = self._step(X, y_true)

        return step_dict

    def compute_loss(self, y_hat: torch.Tensor, y_true: torch.Tensor):
        loss_dict = self._compute_loss(y_hat, y_true)

        if "total_loss" not in loss_dict:
            raise ValueError("The loss dictionary must contain a 'total_loss' key. ")

        return loss_dict

    # ================================================================
    # =                                                              =
    # =                        Training                              =
    # =                                                              =
    # ================================================================
    def training_step(self, batch):
        # Load data from batch
        X, y_true, index = batch

        # Forward pass
        step_dict = self.step(X, y_true, index)

        # Log the process
        self.training_step_output_list.append(step_dict)
        self.log_loss([step_dict], key="train_metrics")

        return step_dict["total_loss"]

    def on_train_epoch_end(self):
        # Log the training metrics every epoch
        self.log_metric(self.training_step_output_list, key="train_metrics")

        # Clear the list of training step outputs for memory efficiency
        self.training_step_output_list.clear()

    def configure_optimizers(self):
        params = self.parameters()

        optimizer = self._optimizer_handler(self.args.optimizer, params)

        lr_scheduler_name = self.args.lr_scheduler
        if lr_scheduler_name is None:
            return optimizer
        else:
            lr_scheduler = self._lr_scheduler_handler(lr_scheduler_name, optimizer)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "monitor": f"{self.args.split_early_stopping}_metrics/{self.args.metric_early_stopping}",
                    "interval": "step",
                    "name": "lr_scheduler",
                },
            }

    def _optimizer_handler(self, optimizer_name, params, **kwargs):
        lr = kwargs.get("lr", self.optim_params.get("lr", 1e-3))
        weight_decay = kwargs.get("weight_decay", self.optim_params.get("weight_decay", 0.0))

        match optimizer_name:
            case "adam":
                optimizer = torch.optim.Adam(
                    params,
                    lr=lr,
                    weight_decay=weight_decay,
                )
            case "adamw":
                optimizer = torch.optim.AdamW(
                    params,
                    lr=lr,
                    weight_decay=weight_decay,
                    betas=[0.9, 0.98],
                )
            case "sgd":
                optimizer = torch.optim.SGD(
                    params,
                    lr=lr,
                    weight_decay=weight_decay,
                )
            case _:
                raise ValueError(f"Optimizer {optimizer_name} is not supported.")

        return optimizer

    def _lr_scheduler_handler(self, lr_scheduler_name, optimizer):
        match lr_scheduler_name:
            case "plateau":
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=0.5,
                    patience=20,
                )
            case "cosine_warm_restart":
                # Usually the model trains in 1000 epochs. The paper "Snapshot ensembles: train 1, get M for free"
                # 	splits the scheduler for 6 periods. We split into 6 periods as well.
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=self.args.cosine_warm_restart_t_0,
                    eta_min=self.args.cosine_warm_restart_eta_min,
                    verbose=True,
                )
            case "linear":
                # Warm up to base lr for stable training
                lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.5,
                    end_factor=1.0,
                    # steps needed to schedule lr
                    total_iters=2 * self.args.log_every_n_steps,
                )
            case "lambda":

                def scheduler(epoch):
                    if epoch < 500:
                        return 0.995**epoch
                    else:
                        return 0.1

                lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler)
            case _:
                raise ValueError(f"LR scheduler {lr_scheduler_name} is not supported.")

        return lr_scheduler

    # ================================================================
    # =                                                              =
    # =                       Validation                             =
    # =                                                              =
    # ================================================================
    def validation_step(self, batch):
        # Load data from batch
        X, y_true, index = batch

        # Forward pass
        step_dict = self.step(X, y_true, index)

        # Log the process
        self.validation_step_output_list.append(step_dict)

    def on_validation_epoch_end(self):
        # Log the validation loss and metrics every epoch
        self.log_loss(self.validation_step_output_list, key="valid_metrics")
        self.log_metric(self.validation_step_output_list, key="valid_metrics")

        # Clear the list of validation step outputs for memory efficiency
        self.validation_step_output_list.clear()

    # ================================================================
    # =                                                              =
    # =                           Test                               =
    # =                                                              =
    # ================================================================
    def test_step(self, batch):
        # Load data from batch
        X, y_true, index = batch

        # Forward pass
        step_dict = self.step(X, y_true, index)

        # Log the process
        self.test_step_output_list.append(step_dict)

        return step_dict["total_loss"]

    def on_test_epoch_end(self):
        # Log the test loss and metrics every epoch
        self.log_loss(self.test_step_output_list, key="test_metrics")
        self.log_metric(self.test_step_output_list, key="test_metrics")

        # Clear the list of test step outputs for memory efficiency
        self.test_step_output_list.clear()

    # ================================================================
    # =                                                              =
    # =                        Logging                               =
    # =                                                              =
    # ================================================================
    def log_loss(self, step_dict_list, key, dataloader_name=""):
        loss_dict_agg = {}
        for loss_name in step_dict_list[0]["loss_dict"].keys():
            loss_dict_agg[loss_name] = np.mean(
                [step_dict["loss_dict"][loss_name].detach().cpu().numpy() for step_dict in step_dict_list]
            )

        for loss_name, loss_value in loss_dict_agg.items():
            self.log(f"{key}/{loss_name}{dataloader_name}", loss_value)

    def log_metric(self, step_dict_list, key, dataloader_name=""):
        metric_dict = self._compute_metric(step_dict_list)

        for metric_name, metric_value in metric_dict.items():
            self.log(f"{key}/{metric_name}{dataloader_name}", metric_value)

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
    def _step(self, X, y_true, **kwargs):
        """Performs a single step of the model.

        Args:
            X (torch.Tensor): The input data.
            y_true (torch.Tensor): The ground truth labels.

        Returns:
            dict: A dictionary containing the step outputs.
        """
        raise NotImplementedError("This method has to be implemented by the sub class")

    @abstractmethod
    def _compute_loss(self, y_hat: torch.Tensor, y_true: torch.Tensor):
        loss_dict = self._compute_loss(y_hat, y_true)

        if "total_loss" not in loss_dict:
            raise ValueError("The loss dictionary must contain a 'total_loss' key. ")

        return loss_dict

    @abstractmethod
    def _compute_metric(self, step_dict_list):
        """Computes the metrics from the step outputs.

        Args:
            step_dict_list (list): A list of dictionaries containing the step outputs.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        raise NotImplementedError("This method has to be implemented by the sub class")

    # ================================================================
    # =                       Properties                             =
    # ================================================================
    @property
    def args(self):
        return self._args

    @args.setter
    def args(self, value):
        self._args = value
        if self.torch_model is not None and hasattr(self.torch_model, "args"):
            self.torch_model.args = value
