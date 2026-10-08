import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tabstruct.common.model.utils.component import MLP

from ...BaseGenerator import BaseLightningGenerationModule, BaseLitJointGenerator


class LitVAE(BaseLitJointGenerator):

    def __init__(self, args):
        super().__init__(args)

        if args.task not in ["classification", "regression", "unsupervision"]:
            raise ValueError(f"Task {args.task} is not supported for {self.name} model")

        self.model = _LitVAE(args)

    # ================================================================
    # =                                                              =
    # =                      Hyperparams                             =
    # =                                                              =
    # ================================================================
    @classmethod
    def _define_default_params(cls):
        params_arch = {
            "latent_dim": 512,
            # Encoder
            "activation_encoder": "tanh",
            "hidden_layer_list_encoder": [512, 512],
            "dropout_encoder": 0.1,
            # Decoder
            "activation_decoder": "tanh",
            "hidden_layer_list_decoder": [512, 512],
            "dropout_decoder": 0,
        }

        params_optim = {
            "lr": 1e-2,
            "weight_decay": 1e-3,
            "coef_reconstruction": 8.0,
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_optuna_params(cls, trial):
        params_arch = {
            "latent_dim": trial.suggest_categorical("latent_dim", [64, 128, 256, 512]),
            # Encoder
            "activation_encoder": trial.suggest_categorical("activation_encoder", ["relu", "tanh", "l_relu"]),
            "hidden_layer_list_encoder": trial.suggest_categorical(
                "hidden_layer_list_encoder",
                [[256, 256], [512, 512], [512, 256, 128], [512, 512, 512]],
            ),
            "dropout_encoder": trial.suggest_uniform("dropout_encoder", 0, 0.5),
            # Decoder
            "activation_decoder": trial.suggest_categorical("activation_decoder", ["relu", "tanh", "l_relu"]),
            "hidden_layer_list_decoder": trial.suggest_categorical(
                "hidden_layer_list_decoder",
                [[256, 256], [512, 512], [512, 256, 128], [512, 512, 512]],
            ),
            "dropout_decoder": trial.suggest_uniform("dropout_decoder", 0, 0.5),
        }

        params_optim = {
            "lr": trial.suggest_loguniform("lr", 1e-4, 5e-2),
            "weight_decay": trial.suggest_loguniform("weight_decay", 1e-5, 1e-1),
            "coef_reconstruction": trial.suggest_uniform("coef_reconstruction", 1.0, 10.0),
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_single_run_params(cls):
        params_arch = {
            "latent_dim": 512,
            # Encoder
            "activation_encoder": "tanh",
            "hidden_layer_list_encoder": [512, 512],
            "dropout_encoder": 0.1,
            # Decoder
            "activation_decoder": "tanh",
            "hidden_layer_list_decoder": [512, 512],
            "dropout_decoder": 0,
        }

        params_optim = {
            "lr": 1e-2,
            "weight_decay": 1e-3,
            "coef_reconstruction": 8.0,
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_test_params(cls):
        params_arch = {
            "latent_dim": 512,
            # Encoder
            "activation_encoder": "tanh",
            "hidden_layer_list_encoder": [512, 512],
            "dropout_encoder": 0.1,
            # Decoder
            "activation_decoder": "tanh",
            "hidden_layer_list_decoder": [512, 512],
            "dropout_decoder": 0,
        }

        params_optim = {
            "lr": 1e-2,
            "weight_decay": 1e-3,
            "coef_reconstruction": 8.0,
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


class _LitVAE(BaseLightningGenerationModule):

    def __init__(self, args):
        super().__init__(args)

    # ================================================================
    # =                                                              =
    # =                     Model-specific                           =
    # =                                                              =
    # ================================================================
    def _create_torch_model(self):
        # Initialize the VAE model with enhanced architecture
        model = VAE(
            input_dim=len(self.args.full_feature_list_model) + len(self.args.full_target_list_model),
            latent_dim=self.args.model_params["architecture"]["latent_dim"],
            # Encoder
            activation_encoder=self.args.model_params["architecture"]["activation_encoder"],
            hidden_layer_list_encoder=self.args.model_params["architecture"]["hidden_layer_list_encoder"],
            dropout_encoder=self.args.model_params["architecture"]["dropout_encoder"],
            # Decoder
            activation_decoder=self.args.model_params["architecture"]["activation_decoder"],
            hidden_layer_list_decoder=self.args.model_params["architecture"]["hidden_layer_list_decoder"],
            dropout_decoder=self.args.model_params["architecture"]["dropout_decoder"],
        )

        return model

    def _compute_loss(self, data_real: torch.Tensor, forward_dict: dict):
        # === Parse the forward output ===
        data_syn = forward_dict["data_syn"]
        z_mu = forward_dict["z_mu"]
        z_log_var = forward_dict["z_log_var"]

        # === Initialise total loss ===
        losses = {
            "total_loss": torch.zeros(1, device=self.device),
        }

        # === Compute all losses ===
        # Compute reconstruction loss
        losses["mse_loss"] = torch.zeros(1, device=self.device)
        losses["cross_entropy_loss"] = torch.zeros(1, device=self.device)
        col_start = 0
        scaler_list = [self.args.feature_scaler_model]
        if self.args.task != "unsupervision":
            scaler_list.append(self.args.target_scaler_model)
        for scaler in scaler_list:
            for col_info in scaler.layout():
                col_type = col_info.feature_type
                col_length = col_info.output_dimensions
                col_end = col_start + col_length

                if col_type == "continuous":
                    losses["mse_loss"] += F.mse_loss(
                        input=data_syn[:, col_start:col_end],
                        target=data_real[:, col_start:col_end],
                        reduction="sum",
                    )
                elif col_type == "discrete":
                    losses["cross_entropy_loss"] += F.cross_entropy(
                        input=data_syn[:, col_start:col_end],
                        target=data_real[:, col_start:col_end],
                        reduction="sum",
                    )

                col_start = col_end

        if col_start != data_real.shape[1]:
            raise RuntimeError(f"Invalid number of columns. Expected {col_start}, got {data_real.shape[1]}")

        # Reduce to sample-wise loss
        losses["mse_loss"] = losses["mse_loss"] / data_real.shape[0]
        losses["cross_entropy_loss"] = losses["cross_entropy_loss"] / data_real.shape[0]

        # Compute KL divergence loss
        kl_element = 1 + z_log_var - z_mu.pow(2) - z_log_var.exp()
        losses["kld_loss"] = -0.5 * torch.sum(kl_element) / data_real.shape[0]

        # === Update total loss ===
        losses["reconstruction_loss"] = losses["mse_loss"] + losses["cross_entropy_loss"]
        losses["reconstruction_loss"] *= self.args.model_params["optimization"]["coef_reconstruction"]
        losses["total_loss"] = losses["reconstruction_loss"] + losses["kld_loss"]

        return losses

    def _generate(self, num_samples: int):
        # Randomly sample from latent space
        z = torch.randn(num_samples, self.args.model_params["architecture"]["latent_dim"]).to(self.device)

        # Generate samples
        generated_samples = self.torch_model.decoder(z)

        return generated_samples


class VAE(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        activation_encoder,
        hidden_layer_list_encoder,
        dropout_encoder,
        activation_decoder,
        hidden_layer_list_decoder,
        dropout_decoder,
    ):
        super().__init__()

        # Encoder
        self.encoder = MLP(
            input_dim=input_dim,
            output_dim=latent_dim * 2,  # For mean and log variance
            activation=activation_encoder,
            hidden_layer_list=hidden_layer_list_encoder,
            dropout_rate=dropout_encoder,
        )

        # Decoder
        self.decoder = MLP(
            input_dim=latent_dim,
            output_dim=input_dim,
            activation=activation_decoder,
            hidden_layer_list=hidden_layer_list_decoder,
            dropout_rate=dropout_decoder,
        )

    def forward(self, x):
        # Compute latent variable
        x_emb = self.encoder(x)
        z_mu, z_log_var = torch.chunk(x_emb, 2, dim=-1)
        z = self._reparameterize(z_mu, z_log_var)

        # Decode the latent variable
        h = self.decoder(z)

        return {
            "data_syn": h,
            "z_mu": z_mu,
            "z_log_var": z_log_var,
        }

    def _reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std
