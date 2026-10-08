from abc import abstractmethod

import numpy as np
import torch
import torch.nn.functional as F

from src.tabstruct.common.model.BaseModel import BaseLightningModule, BaseModel, LitModelMixin

from .utils.evaluation import compute_all_metrics


class BasePredictor(BaseModel):

    def __init__(self, args):
        super().__init__(args)

    # ================================================================
    # =                                                              =
    # =                   Model prediction                           =
    # =                                                              =
    # ================================================================
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

    # ================================================================
    # =                                                              =
    # =              Utils to implement in sub class                 =
    # =                                                              =
    # ================================================================
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


class BaseLitPredictor(LitModelMixin, BasePredictor):
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


class BaseLightningPredictionModule(BaseLightningModule):
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
            "loss_dict": loss_dict,
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
