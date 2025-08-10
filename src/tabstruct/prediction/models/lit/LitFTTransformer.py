import torch
import torch.nn as nn
import torch.nn.functional as F
from rtdl_revisiting_models import FTTransformer

from ..BasePredictor import BaseLightingPredictionModule, BaseLitPredictor


class LitFTTransformer(BaseLitPredictor):

    def __init__(self, args):
        super().__init__(args)

        if args.task not in ["classification", "regression"]:
            raise ValueError(f"Task {args.task} is not supported for MLP model")

        self.model = _LitFTTransformer(args)

    # ================================================================
    # =                                                              =
    # =                      Hyperparams                             =
    # =                                                              =
    # ================================================================
    @classmethod
    def _define_default_params(cls):
        params_arch = {
            "n_blocks": 3,
            "d_block": 192,
            "attention_n_heads": 8,
            "attention_dropout": 0.2,
            "ffn_d_hidden": None,
            "ffn_d_hidden_multiplier": 4 / 3,
            "ffn_dropout": 0.1,
            "residual_dropout": 0.0,
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
        params_arch = {
            "n_blocks": trial.suggest_int("n_blocks", 1, 6),
            "d_block": trial.suggest_int("d_block", 64, 512, log=True),
            "attention_n_heads": trial.suggest_int("attention_n_heads", 1, 16),
            "attention_dropout": trial.suggest_float("attention_dropout", 0.0, 0.5),
            "ffn_d_hidden": trial.suggest_int("ffn_d_hidden", 32, 512, log=True),
            "ffn_d_hidden_multiplier": trial.suggest_float("ffn_d_hidden_multiplier", 1.0, 4.0, log=True),
            "ffn_dropout": trial.suggest_float("ffn_dropout", 0.0, 0.5),
            "residual_dropout": trial.suggest_float("residual_dropout", 0.0, 0.5),
        }

        params_optim = {
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_single_run_params(cls):
        params_arch = {
            "n_blocks": 3,
            "d_block": 192,
            "attention_n_heads": 8,
            "attention_dropout": 0.2,
            "ffn_d_hidden": None,
            "ffn_d_hidden_multiplier": 4 / 3,
            "ffn_dropout": 0.1,
            "residual_dropout": 0.0,
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
    def _define_test_params(cls):
        params_arch = {
            "n_blocks": 3,
            "d_block": 192,
            "attention_n_heads": 8,
            "attention_dropout": 0.2,
            "ffn_d_hidden": None,
            "ffn_d_hidden_multiplier": 4 / 3,
            "ffn_dropout": 0.1,
            "residual_dropout": 0.0,
        }

        params_optim = {
            "lr": 3e-3,
            "weight_decay": 1e-4,
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }


class _LitFTTransformer(BaseLightingPredictionModule):

    def __init__(self, args):
        super().__init__(args)

    # ================================================================
    # =                                                              =
    # =                     Model-specific                           =
    # =                                                              =
    # ================================================================
    def _create_torch_model(self):
        model = FTTransformerWrapper(
            n_cont_features=len(self.args.num_feature_col_list_processed),
            cat_cardinalities=self.args.cat_feature_cardinality_list_processed,
            d_out=self.args.full_num_classes_processed if self.args.task == "classification" else 1,
            model_params=self.args.model_params["architecture"],
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


class FTTransformerWrapper(nn.Module):
    """Wrapper to handle input splitting for FTTransformer."""

    def __init__(self, n_cont_features, cat_cardinalities, d_out, model_params):
        super().__init__()
        self.n_cont_features = n_cont_features
        self.model = FTTransformer(
            n_cont_features=n_cont_features,
            cat_cardinalities=cat_cardinalities,
            d_out=d_out,
            **model_params,
        )

    def forward(self, x):
        x_cont = x[:, : self.n_cont_features]
        # Ensure categorical features are in int64 format
        x_cat = x[:, self.n_cont_features :].to(torch.int64)
        if x_cat.numel() == 0:
            x_cat = None

        return self.model(x_cont, x_cat)
