import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tabstruct.common.model.utils.component import MLP

from ...BaseGenerator import BaseLightningGenerationModule, BaseLitJointGenerator
from .utils.config import get_beta_schedule
from .utils.modules import TimestepEmbedding


class LitDDPM(BaseLitJointGenerator):

    def __init__(self, args):
        super().__init__(args)

        if args.task not in ["classification", "regression", "unsupervision"]:
            raise ValueError(f"Task {args.task} is not supported for {self.name} model")

        self.model = _LitDDPM(args)

    # ================================================================
    # =                                                              =
    # =                      Hyperparams                             =
    # =                                                              =
    # ================================================================
    @classmethod
    def _define_default_params(cls):
        params_arch = {
            "num_timesteps": 1000,
            "denoising_network_hidden_layer_list": [512, 512, 512],
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
    def _define_optuna_params(cls, trial):
        params_arch = {
            "num_timesteps": trial.suggest_categorical("num_timesteps", [500, 1000, 2000]),
            "denoising_network_hidden_layer_list": trial.suggest_categorical(
                "denoising_network_hidden_layer_list",
                [[256, 256], [512, 512], [512, 512, 512], [1024, 512, 1024]],
            ),
        }

        params_optim = {
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_single_run_params(cls):
        params_arch = {
            "num_timesteps": 1000,
            "denoising_network_hidden_layer_list": [512, 512, 512],
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
            "num_timesteps": 100,
            "denoising_network_hidden_layer_list": [128, 128],
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
    def _get_model_specific_scaler_config(cls):
        """Get DDPM-specific scaler configuration."""
        scaler_config_dict = {
            "context": {
                "disable_preprocessing": False,
            },
            "feature_scaler": {
                "categorical_transform": "ordinal",  # DDPM works better with ordinal encoding
                "categorical_as_numerical": False,
            },
            "target_scaler": {},
        }

        return scaler_config_dict


class _LitDDPM(BaseLightningGenerationModule):

    def __init__(self, args):
        super().__init__(args)

    # ================================================================
    # =                                                              =
    # =                     Model-specific                           =
    # =                                                              =
    # ================================================================
    def _create_torch_model(self):
        # Initialize the DDPM model
        input_dim = len(self.args.full_feature_list_model) + len(self.args.full_target_list_model)
        timestep_embedding = TimestepEmbedding(
            input_dim=input_dim,
            encode_mode="concat",
            encode_reduce="sum",
        )
        denoising_network = MLP(
            input_dim=timestep_embedding.output_dim,
            output_dim=input_dim,
            hidden_layer_list=self.args.model_params["architecture"]["denoising_network_hidden_layer_list"],
            activation="l_relu",
            dropout_rate=0,
        )
        model = DDPM(
            num_features=input_dim,
            num_timesteps=self.args.model_params["architecture"]["num_timesteps"],
            beta_schedule="linear",
            beta_start=1e-4,
            beta_end=0.02,
            timestep_embedding=timestep_embedding,
            denoising_network=denoising_network,
        )

        return model

    def _compute_loss(self, data_real: torch.Tensor, forward_dict: dict):
        # Parse the forward output
        noise_pred = forward_dict["noise_pred"]
        noise_true = forward_dict["noise_true"]

        # Initialise total loss
        losses = {
            "total_loss": torch.zeros(1, device=self.device),
        }

        # Compute diffusion loss (MSE between predicted and true noise)
        losses["diffusion_loss"] = F.mse_loss(input=noise_pred, target=noise_true, reduction="sum")
        losses["diffusion_loss"] = losses["diffusion_loss"] / data_real.shape[0]

        # Update total loss
        losses["total_loss"] = losses["diffusion_loss"]

        return losses

    def _generate(self, num_samples: int):
        # Generate samples using the reverse diffusion process
        generated_samples = self.torch_model.sample(num_samples, device=self.device)

        return generated_samples


class DDPM(nn.Module):
    """Tabular Denoising Diffusion Probabilistic Model (DDPM)"""

    def __init__(
        self,
        num_features,
        num_timesteps,
        beta_start,
        beta_end,
        beta_schedule,
        timestep_embedding,
        denoising_network,
    ):
        """Initialize the DDPM model.

        Timestep:     0 ────► 1 ────► 2 ────► ... ────► T
                    clean   noisy   noisier         pure noise

        β_t:           β_1    β_2     β_3     ...     β_T
        a_t:           a_1    a_2     a_3     ...     a_T
        ᾱ_t:           a_1   a_1a_2  a_1a_2a_3 ...   ∏a_i

        Signal:       100%    90%     70%     ...      0%
        Noise:         0%     10%     30%     ...    100%

        Args:
            input_dim (int): Dimension of the input data.
            hidden_dim (int): Dimension of the hidden layers.
            num_layers (int): Number of layers in the model.
            activation (str): Activation function to use.
            dropout (float): Dropout rate.
            num_timesteps (int): Number of timesteps for the diffusion process.
            beta_schedule (str): Beta schedule for the diffusion process.
            beta_start (float): Starting value of beta.
            beta_end (float): Ending value of beta.
        """
        super().__init__()

        # === Save parameters ===
        self.num_features = num_features
        self.num_timesteps = num_timesteps
        self.timestep_embedding = timestep_embedding

        # === Initialize beta schedule for noise ===
        # betas (β_t) define the variance of the noise added at each timestep (bigger beta = more noise)
        betas = get_beta_schedule(beta_start, beta_end, num_timesteps, beta_schedule)
        # alphas (a_t = 1 - β_t) are the amount of original signal preserved at each timestep (bigger alpha = more signal)
        alphas = 1.0 - betas
        # alpha_cumprod (ᾱ_t = ∏_{s=1}^t a_s) allows direct sampling at any timestep t without iterating
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        # alpha_cumprod_prev (ᾱ_{t-1}) is used in the reverse process to compute the mean of p(x_{t-1} | x_t)
        alpha_cumprod_prev = F.pad(alpha_cumprod[:-1], (1, 0), value=1.0)

        # === Precomputed Values for Efficiency ===
        # Forward pass scaling factors: x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
        sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)

        # Reverse pass scaling factors: x_{t-1} = (1/√a_t) * (x_t - (1-a_t)/√(1-ᾱ_t) * ε_θ(x_t, t)) + √β_t * z
        sqrt_recip_alpha = torch.sqrt(1.0 / alphas)

        # === Register buffers to ensure they are moved to the correct device with the model ===
        # Basic strengths for forward and reverse processes
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("alpha_cumprod_prev", alpha_cumprod_prev)
        # Forward pass buffers
        self.register_buffer("sqrt_alpha_cumprod", sqrt_alpha_cumprod)
        self.register_buffer("sqrt_one_minus_alpha_cumprod", sqrt_one_minus_alpha_cumprod)
        # Reverse pass buffers
        self.register_buffer("sqrt_recip_alpha", sqrt_recip_alpha)

        # === Initialize the denoising network ===
        # The network ε_θ(x_t, t) takes in the noisy data x_t and timestep t, and predicts the noise ε added
        self.denoising_network = denoising_network

    # ================================================================
    # =                                                              =
    # =                    Top-level APIs                            =
    # =                                                              =
    # ================================================================
    def forward(self, x):
        """Forward pass during training."""
        batch_size = x.shape[0]
        device = x.device

        # === Prepare noise for different timestamps ===
        # Randomly samples timesteps t from [0, num_timesteps) for each sample in the batch
        # This allows the model to learn denoising at different noise levels simultaneously
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()

        # Generates random Gaussian noise with the same shape as input data
        # This is the "true noise" that will be added to the data
        noise = torch.randn_like(x)

        # === Add noise to the data ===
        # Add noise to the data according to the forward process
        # sqrt_alpha_cumprod_t is the scaling factor for the original data
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[t][:, None]
        # sqrt_one_minus_alpha_cumprod_t is the scaling factor for the noise
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t][:, None]
        # x_noisy is the noisy version of the input data at timestep t
        x_noisy = sqrt_alpha_cumprod_t * x + sqrt_one_minus_alpha_cumprod_t * noise

        # === Predict the noise ===
        # Creates timestep embeddings to tell the network what noise level to expect
        x_input = self.timestep_embedding(x_noisy, t)
        # Neural network predicts what noise was added
        noise_pred = self.denoising_network(x_input)

        return {
            "noise_pred": noise_pred,
            "noise_true": noise,
        }

    def sample(self, num_samples, device):
        """Generate samples using the reverse diffusion process."""
        # === Reverse diffusion process ===
        # Start from pure noise
        x = torch.randn(num_samples, self.num_features, device=device)
        for t in reversed(range(self.num_timesteps)):
            # Goes backwards from T-1 down to 0
            # Broadcast to set the same timestep t for all samples in the batch
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)

            # Predict noise
            # Tells the network "what noise level am I looking at?"
            x_input = self.timestep_embedding(x, t_batch)
            noise_pred = self.denoising_network(x_input)

            beta_t = self.betas[t]
            if t > 0:
                noise_level = torch.sqrt(beta_t) * torch.randn_like(x)
            else:
                noise_level = torch.zeros_like(x)

            x = (
                self.sqrt_recip_alpha[t] * (x - self.betas[t] / self.sqrt_one_minus_alpha_cumprod[t] * noise_pred)
                + noise_level
            )

        return x
