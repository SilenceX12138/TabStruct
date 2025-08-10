from abc import abstractmethod

import lightning as L
import numpy as np
import optuna
import torch


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
        self.args = args
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


class BaseLightningModule(L.LightningModule):
    """Base class for all Lightning modules.

    This class provides a common interface for all Lightning modules used in the project.

    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.torch_model = self.create_torch_model()
        self.optim_params = self.args.model_params["optimization"]

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
        torch_model = self._create_torch_model()

        return torch_model

    def forward(self, X):
        """Make LightningModule compatible with upper-level calls."""
        return self.torch_model(X)

    def step(self, X, y_true):
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
        X, y_true = batch

        # Forward pass
        step_dict = self.step(X, y_true)

        # Log the process
        self.training_step_output_list.append(step_dict)
        self.log_loss([step_dict], key="train_metrics")

        return step_dict["total_loss"]

    def on_train_epoch_end(self):
        self.log_loss(self.training_step_output_list, key="train_metrics")
        self.log_metric(self.training_step_output_list, key="train_metrics")

        # Clear the list of training step outputs for memory efficiency
        self.training_step_output_list.clear()

    def configure_optimizers(self):
        params = self.parameters()

        if self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(
                params, lr=self.optim_params["lr"], weight_decay=self.optim_params["weight_decay"]
            )
        if self.args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                params, lr=self.optim_params["lr"], weight_decay=self.optim_params["weight_decay"], betas=[0.9, 0.98]
            )
        if self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                params, lr=self.optim_params["lr"], weight_decay=self.optim_params["weight_decay"]
            )
        else:
            raise ValueError(f"Optimizer {self.args.optimizer} is not supported.")

        if self.args.lr_scheduler == "none":
            return optimizer
        else:
            if self.args.lr_scheduler == "plateau":
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=30, verbose=True
                )
            elif self.args.lr_scheduler == "cosine_warm_restart":
                # Usually the model trains in 1000 epochs. The paper "Snapshot ensembles: train 1, get M for free"
                # 	splits the scheduler for 6 periods. We split into 6 periods as well.
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=self.args.cosine_warm_restart_t_0,
                    eta_min=self.args.cosine_warm_restart_eta_min,
                    verbose=True,
                )
            elif self.args.lr_scheduler == "linear":
                # Warm up to base lr for stable training
                lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.5,
                    end_factor=1.0,
                    # steps needed to schedule lr
                    total_iters=2 * self.args.log_every_n_steps,
                )
            elif self.args.lr_scheduler == "lambda":

                def scheduler(epoch):
                    if epoch < 500:
                        return 0.995**epoch
                    else:
                        return 0.1

                lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler)
            else:
                raise Exception()

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "monitor": f"valid_metrics/{self.args.metric_model_selection}",
                    "interval": "step",
                    "name": "lr_scheduler",
                },
            }

    # ================================================================
    # =                                                              =
    # =                       Validation                             =
    # =                                                              =
    # ================================================================
    def validation_step(self, batch):
        # Load data from batch
        X, y_true = batch

        # Forward pass
        step_dict = self.step(X, y_true)

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
        X, y_true = batch

        # Forward pass
        step_dict = self.step(X, y_true)

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
            loss_dict_agg[loss_name] = np.mean([step_dict["loss_dict"][loss_name] for step_dict in step_dict_list])

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
    def _step(self, X, y_true):
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
