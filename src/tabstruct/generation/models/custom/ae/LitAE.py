import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...BaseGenerator import BaseLightningGenerationModule, BaseLitJointGenerator


class LitAE(BaseLitJointGenerator):

    def __init__(self, args):
        super().__init__(args)

        if args.task not in ["classification", "regression", "unsupervision"]:
            raise ValueError(f"Task {args.task} is not supported for {self.name} model")

        self.model = _LitAE(args)

    # ================================================================
    # =                                                              =
    # =                     Model-specific                           =
    # =                                                              =
    # ================================================================
    def _fit_model(self, data_module):
        self.model.save_train_embedding(data_module)

    # ================================================================
    # =                                                              =
    # =                      Hyperparams                             =
    # =                                                              =
    # ================================================================
    @classmethod
    def _define_default_params(cls):
        params_arch = {
            "hidden_dim": 128,
        }

        params_optim = {
            "lr": 1e-1,
            "weight_decay": 1e-4,
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_optuna_params(cls, trial):
        params_arch = {
            "hidden_dim": trial.suggest_int("hidden_dim", 64, 256, step=64),
        }

        params_optim = {
            "lr": trial.suggest_loguniform("lr", 1e-5, 1e-1),
            "weight_decay": trial.suggest_loguniform("weight_decay", 1e-5, 1e-1),
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_single_run_params(cls):
        params_arch = {
            "hidden_dim": 128,
        }

        params_optim = {
            "lr": 1e-1,
            "weight_decay": 1e-4,
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_test_params(cls):
        params_arch = {
            "hidden_dim": 128,
        }

        params_optim = {
            "lr": 1e-1,
            "weight_decay": 1e-4,
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _get_model_specific_scaler_config(cls):
        """Get TabEval-specific scaler configuration."""
        scaler_config_dict = {
            "context": {
                "disable_preprocessing": False,
            },
            "feature_scaler": {
                "categorical_transform": "ordinal",
                "categorical_as_numerical": False,
            },
            "target_scaler": {},
        }

        return scaler_config_dict


class _LitAE(BaseLightningGenerationModule):

    def __init__(self, args):
        super().__init__(args)

    # ================================================================
    # =                                                              =
    # =                     Model-specific                           =
    # =                                                              =
    # ================================================================
    def _create_torch_model(self):
        # Initialize the AE model
        model = AE(
            input_dim=len(self.args.full_feature_list_model) + len(self.args.full_target_list_model),
            hid_dim=self.args.model_params["architecture"]["hidden_dim"],
        )

        return model

    def _compute_loss(self, data_real: torch.Tensor, forward_dict: dict):
        # Parse the forward output
        data_syn = forward_dict["data_syn"]

        # Initialise total loss
        losses = {
            "total_loss": torch.zeros(1, device=self.device),
        }

        # Compute all losses
        losses["mse_loss"] = F.mse_loss(input=data_syn, target=data_real)

        # Update total loss
        losses["total_loss"] = losses["mse_loss"]

        return losses

    def _generate(self, num_samples: int):
        # Randomly sample from train_embedding for interpolation
        indices = torch.randint(0, len(self.train_embedding), (num_samples, 2))
        embedding1 = torch.tensor(self.train_embedding[indices[:, 0]], dtype=torch.float32, device=self.device)
        embedding2 = torch.tensor(self.train_embedding[indices[:, 1]], dtype=torch.float32, device=self.device)

        # Random interpolation weights
        alpha = torch.rand(num_samples, 1, device=self.device)
        z = alpha * embedding1 + (1 - alpha) * embedding2
        z = z.to(self.device)

        # Generate samples
        generated_samples = self.torch_model.decoder(z)

        return generated_samples

    # ================================================================
    # =                                                              =
    # =                          Utils                               =
    # =                                                              =
    # ================================================================
    def save_train_embedding(self, data_module):
        train_embedding_list = []

        # === Iterate over the training data loader to get embeddings ===
        for X_batch, y_batch, _ in data_module.train_dataloader():
            # Prepare the data batch
            data_batch = X_batch.to(self.device)
            if self.args.task != "unsupervision":
                y_batch = y_batch.unsqueeze(1) if len(y_batch.shape) == 1 else y_batch
                data_batch = torch.cat([data_batch, y_batch.to(self.device)], dim=1)

            # Get the embedding
            embedding_temp = self.get_embedding(data_batch).detach().cpu().numpy()
            train_embedding_list.append(embedding_temp)

        self.train_embedding = np.concatenate(train_embedding_list, axis=0)

    def get_embedding(self, data: torch.Tensor):
        # Get the embedding for the input data
        self.torch_model.eval()
        with torch.no_grad():
            embedding = self.torch_model.encode(data.to(self.device))

        return embedding


class AE(nn.Module):
    def __init__(self, input_dim, hid_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hid_dim),
            nn.ReLU(),
            nn.Linear(in_features=hid_dim, out_features=hid_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=hid_dim, out_features=hid_dim),
            nn.ReLU(),
            nn.Linear(in_features=hid_dim, out_features=input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        h = self.decoder(z)

        return {
            "data_syn": h,
        }

    def encode(self, x):
        z = self.encoder(x)

        return z
