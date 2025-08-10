import torch
import torch.nn as nn
import torch.nn.functional as F

from ..BasePredictor import BaseLightingPredictionModule, BaseLitPredictor
from ..utils.activation import get_activation


class LitMLP(BaseLitPredictor):

    def __init__(self, args):
        super().__init__(args)

        if args.task not in ["classification", "regression"]:
            raise ValueError(f"Task {args.task} is not supported for MLP model")

        self.model = _LitMLP(args)

    # ================================================================
    # =                                                              =
    # =                      Hyperparams                             =
    # =                                                              =
    # ================================================================
    @classmethod
    def _define_default_params(cls):
        params_arch = {
            "activation": "tanh",
            "hidden_layer_list": [100, 100, 10],
            "dropout_rate": 0,
            "batch_normalization": True,
        }

        params_optim = {
            "lr": 3e-3,
            "weight_decay": 1e-4,
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_optuna_params(cls, trial):
        hidden_dim = trial.suggest_int("hidden_dim", 10, 100)
        n_layers = trial.suggest_int("n_layers", 1, 5)
        params_arch = {
            "activation": trial.suggest_categorical("activation", ["tanh", "relu", "l_relu", "sigmoid", "none"]),
            "hidden_layer_list": n_layers * [hidden_dim],
            "dropout_rate": trial.suggest_float("dropout_rate", 0, 0.5),
            "batch_normalization": trial.suggest_categorical("batch_normalization", [True, False]),
        }

        params_optim = {
            "lr": trial.suggest_float("lr", 5e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_single_run_params(cls):
        params_arch = {
            "activation": "tanh",
            "hidden_layer_list": [100, 100, 10],
            "dropout_rate": 0,
            "batch_normalization": True,
        }

        params_optim = {
            "lr": 1e-3,
            "weight_decay": 1e-4,
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_test_params(cls):
        params_arch = {
            "activation": "tanh",
            "hidden_layer_list": [100, 100, 10],
            "dropout_rate": 0,
            "batch_normalization": True,
        }

        param_optim = {
            "lr": 1e-3,
            "weight_decay": 1e-4,
        }

        return {
            "architecture": params_arch,
            "optimization": param_optim,
        }


class _LitMLP(BaseLightingPredictionModule):

    def __init__(self, args):
        super().__init__(args)

    # ================================================================
    # =                                                              =
    # =                     Model-specific                           =
    # =                                                              =
    # ================================================================
    def _create_torch_model(self):
        model = MLP(
            input_dim=self.args.full_num_features_processed,
            output_dim=self.args.full_num_classes_processed if self.args.task == "classification" else 1,
            **self.args.model_params["architecture"],
        )

        return model

    def _compute_loss(self, y_hat: torch.Tensor, y_true: torch.Tensor):
        losses = {}
        losses["total_loss"] = torch.zeros(1, device=self.device)

        # compute loss for prediction
        if self.args.task == "regression":
            losses["mse_loss"] = F.mse_loss(input=y_hat.squeeze(-1), target=y_true)
            losses["total_loss"] = losses["mse_loss"]
        else:
            losses["cross_entropy_loss"] = F.cross_entropy(
                input=y_hat,
                target=y_true,
                weight=torch.tensor(self.args.train_class_weight_list, dtype=torch.float32, device=self.device),
            )
            losses["total_loss"] = losses["cross_entropy_loss"]

        return losses


class MLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str,
        hidden_layer_list: list,
        batch_normalization: bool = True,
        dropout_rate: float = 0,
    ) -> None:
        """MLP for classification or regression

        Args:
            output_dim (int): number of nodes for the output layer of the prediction net, 1 (regression) or 2 (classification)
            activation (str): activation function of the prediction net: 'relu', 'l_relu', 'sigmoid', 'tanh', or 'none'
            hidden_layer_list (list): number of nodes for each hidden layer for the prediction net, example: [200,200]
        """
        super().__init__()

        self.num_classes = output_dim
        self.act = get_activation(activation)
        full_layer_list = [input_dim, *hidden_layer_list]
        self.fn = nn.Sequential()
        for i in range(len(full_layer_list) - 1):
            self.fn.add_module("fn{}".format(i), nn.Linear(full_layer_list[i], full_layer_list[i + 1]))
            self.fn.add_module("act{}".format(i), self.act)
            # use BN after activation has better performance
            if batch_normalization:
                self.fn.add_module("bn{}".format(i), nn.BatchNorm1d(full_layer_list[i + 1]))
            if dropout_rate > 0:
                self.fn.add_module("dropout{}".format(i), nn.Dropout(dropout_rate))

        self.head = nn.Sequential()
        self.head.add_module("head", nn.Linear(full_layer_list[-1], output_dim))

        # when using cross-entropy loss in pytorch, we do not need to use softmax.
        # self.head.add_module('softmax', nn.Softmax(-1))

    def forward(self, x):
        x_emb = self.fn(x)
        x = self.head(x_emb)

        return x
