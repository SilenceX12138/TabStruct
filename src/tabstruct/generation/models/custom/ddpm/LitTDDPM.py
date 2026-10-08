import torch
import torch.nn as nn
import torch.nn.functional as F

from ...BaseGenerator import BaseLightningGenerationModule, BaseLitJointGenerator
from .utils.config import get_beta_schedule
from .utils.modules import UniModMLP


class LitTDDPM(BaseLitJointGenerator):
    """LitTDDPM model for tabular data generation.
    Based on DDPM, but adapted for tabular data with mixed feature types.

    - Numerical features: Gaussian noise
    - Categorical features: Uniform noise in probability space
    """

    def __init__(self, args):
        super().__init__(args)

        if args.task not in ["classification", "regression", "unsupervision"]:
            raise ValueError(f"Task {args.task} is not supported for {self.name} model")

        self.model = _LitTDDPM(args)

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
            "num_timesteps": trial.suggest_categorical("num_timesteps", [100, 500, 1000]),
            "denoising_network_hidden_layer_list": trial.suggest_categorical(
                "denoising_network_hidden_layer_list",
                [[256, 256], [512, 512], [512, 512, 512]],
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
        """Get TDDPM-specific scaler configuration."""
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


class _LitTDDPM(BaseLightningGenerationModule):

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
        denoising_network = UniModMLP(
            d_numerical=len(self.args.full_indices_numerical_list_model),
            categories=self.args.train_cardinality_list_model,
            num_layers=2,
        )
        model = TDDPM(
            # Data parameters
            num_features=input_dim,
            indices_numerical_list=self.args.full_indices_numerical_list_model,
            indices_categorical_list=self.args.full_indices_categorical_list_model,
            categorical_cardinality_list=self.args.train_cardinality_list_model,
            # Diffusion parameters
            num_timesteps=self.args.model_params["architecture"]["num_timesteps"],
            beta_schedule="linear",
            beta_start=1e-4,
            beta_end=0.02,
            denoising_network=denoising_network,
        )

        return model

    def _compute_loss(self, data_real: torch.Tensor, forward_dict: dict):
        # Parse the forward output
        noise_dict = forward_dict["noise_dict"]
        posterior_dict = forward_dict["posterior_dict"]
        denoise_output = forward_dict["denoise_output"]

        # Initialize total loss
        losses = {
            "total_loss": torch.zeros(1, device=self.device),
        }

        # Compute individual losses
        losses["numerical_loss"] = self._compute_numerical_loss(
            noise_pred=denoise_output[:, self.args.full_indices_numerical_list_model],
            noise_true=noise_dict["numerical_noise"],
        )
        losses["categorical_loss"] = self._compute_categorical_loss(
            data_syn=denoise_output[:, self.args.full_indices_categorical_list_model],
            data_posterior=posterior_dict["categorical_posterior"],
        )

        # Update total loss
        losses["total_loss"] = losses["numerical_loss"] + losses["categorical_loss"]

        return losses

    def _compute_numerical_loss(self, noise_pred, noise_true):
        if noise_pred.shape[1] == 0:
            return torch.zeros(1, device=noise_pred.device)

        # Mean Squared Error loss for numerical features
        mse_loss = F.mse_loss(noise_pred, noise_true, reduction="sum")
        # Normalize by batch size
        mse_loss = mse_loss / noise_pred.shape[0]

        return mse_loss

    def _compute_categorical_loss(self, data_syn, data_posterior):
        """Compute the categorical loss for the DDPM model."""
        # Return zero if there are no categorical features
        if data_syn.shape[1] == 0:
            return torch.zeros(1, device=data_syn.device)

        # Compute cross-entropy loss
        ce_loss = torch.zeros(1, device=data_syn.device)
        start_col = 0
        for cardinality in self.args.train_cardinality_list_model:
            end_col = start_col + cardinality
            ce_loss += F.cross_entropy(
                data_syn[:, start_col:end_col],
                data_posterior[:, start_col:end_col],
                reduction="sum",
            )
            start_col = end_col
        # Normalize by batch size
        ce_loss = ce_loss / data_syn.shape[0]

        return ce_loss

    def _generate(self, num_samples: int):
        # Generate samples using the reverse diffusion process
        generated_samples = self.torch_model.sample(num_samples, device=self.device)

        return generated_samples


class TDDPM(nn.Module):
    def __init__(
        self,
        # Data parameters
        num_features,
        indices_numerical_list,
        indices_categorical_list,
        categorical_cardinality_list,
        # Diffusion parameters
        num_timesteps,
        beta_start,
        beta_end,
        beta_schedule,
        denoising_network,
    ):
        """Initialize the TDDPM model.

        Architecture Flow:
            Input: [numerical_features | categorical_onehot_features]
                ↓
            Add respective noise types
                ↓
            Noisy Input: [x_num_t | x_cat_t]
                ↓
            Project to embedding dimension: proj(x)
                ↓
            Create timestep embedding: time_embed(timestep_embedding(t))
                ↓
            ADD timestep embedding: proj(x) + emb
                ↓
            Pass through MLP layers
                ↓
            Output: [denoised_numerical | denoised_categorical_logits]

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
        # Data parameters
        self.num_features = num_features
        self.indices_numerical_list = indices_numerical_list
        self.indices_categorical_list = indices_categorical_list
        self.categorical_cardinality_list = categorical_cardinality_list
        # Diffusion parameters
        self.num_timesteps = num_timesteps
        self.denoising_network = denoising_network

        # === Initialize beta schedule for noise ===
        self._compute_diffusion_schedule(
            beta_start=beta_start,
            beta_end=beta_end,
            num_timesteps=num_timesteps,
            beta_schedule=beta_schedule,
        )

    def _compute_diffusion_schedule(self, beta_start, beta_end, num_timesteps, beta_schedule):
        # betas (β_t) define the variance of the noise added at each timestep (bigger beta = more noise)
        betas = get_beta_schedule(beta_start, beta_end, num_timesteps, beta_schedule)
        # alphas (a_t = 1 - β_t) are the amount of original signal preserved at each timestep (bigger alpha = more signal)
        alphas = 1.0 - betas
        one_minus_alpha = 1.0 - alphas
        # alpha_cumprod (ᾱ_t = ∏_{s=1}^t a_s) allows direct sampling at any timestep t without iterating
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        one_minus_alpha_cumprod = 1.0 - alpha_cumprod
        # alpha_cumprod_prev (ᾱ_{t-1}) is used in the reverse process to compute the mean of p(x_{t-1} | x_t)
        alpha_cumprod_prev = F.pad(alpha_cumprod[:-1], (1, 0), value=1.0)

        # === Precomputed Values for Efficiency ===
        # Forward pass scaling factors for numerical features: x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
        sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(one_minus_alpha_cumprod)
        # Forward pass log scaling factors for categorical features: q(x_t | x_0) = α̃_t · δ(x_t = x_0) + (1 - α̃_t) · Uniform(x_t)
        log_cumprod_alpha = torch.log(alpha_cumprod)

        # Reverse pass scaling factors: x_{t-1} = (1/√a_t) * (x_t - (1-a_t)/√(1-ᾱ_t) * ε_θ(x_t, t)) + √β_t * z
        sqrt_recip_alpha = torch.sqrt(1.0 / alphas)
        # standard deviation of the sampling noise to add in the reverse process
        sqrt_beta = torch.sqrt(betas)

        # === Register buffers to ensure they are moved to the correct device with the model ===
        # Basic strengths for forward and reverse processes
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("one_minus_alpha", one_minus_alpha)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("one_minus_alpha_cumprod", one_minus_alpha_cumprod)
        self.register_buffer("alpha_cumprod_prev", alpha_cumprod_prev)
        # Forward pass buffers
        self.register_buffer("sqrt_alpha_cumprod", sqrt_alpha_cumprod)
        self.register_buffer("sqrt_one_minus_alpha_cumprod", sqrt_one_minus_alpha_cumprod)
        self.register_buffer("log_cumprod_alpha", log_cumprod_alpha)
        # Reverse pass buffers
        self.register_buffer("sqrt_recip_alpha", sqrt_recip_alpha)
        self.register_buffer("sqrt_beta", sqrt_beta)

    # ================================================================
    # =                                                              =
    # =                   Forward diffusion                          =
    # =                                                              =
    # ================================================================
    def forward(self, x):
        """The forward pass of the diffusion model."""
        # Prepare data
        batch_size = x.shape[0]
        device = x.device

        # Prepare noise
        t_batch = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()
        noise_dict = self._prepare_mixed_noise(batch_size, device)

        # Forward diffusion: Add noise to the data
        x_noisy = self._apply_mixed_noise(x, t_batch, noise_dict)

        # Prepare posterior data for computing categorical loss
        posterior_dict = self._prepare_posterior(x, x_noisy, t_batch)

        # Denoising step
        x_num_noise_pred, x_cat_pred = self.denoising_network(
            x_num=x_noisy[:, self.indices_numerical_list],
            x_cat=x_noisy[:, self.indices_categorical_list],
            timesteps=t_batch,
        )
        denoise_output = torch.zeros_like(x, device=device)
        denoise_output[:, self.indices_numerical_list] = x_num_noise_pred
        denoise_output[:, self.indices_categorical_list] = x_cat_pred

        return {
            "noise_dict": noise_dict,
            "posterior_dict": posterior_dict,
            "denoise_output": denoise_output,
        }

    def _prepare_mixed_noise(self, batch_size, device):
        """Prepare mixed noise for both numerical and categorical features.

        Args:
            batch_size (int): The number of samples in the batch.
            device (torch.device): The device to create the noise tensor on.

        Returns:
            torch.Tensor: A tensor containing the mixed noise.
        """
        # Prepare numerical and categorical noise separately
        numerical_noise = self._prepare_numerical_noise(batch_size, device)
        categorical_noise = self._prepare_categorical_noise(batch_size, device)

        return {
            "numerical_noise": numerical_noise,
            "categorical_noise": categorical_noise,
        }

    def _prepare_numerical_noise(self, batch_size, device):
        """Prepare Gaussian noise for numerical features.

        Args:
            batch_size (int): The number of samples in the batch.
            device (torch.device): The device to create the noise tensor on.

        Returns:
            torch.Tensor: A tensor containing the Gaussian noise.
        """
        num_numerical_features = len(self.indices_numerical_list)
        if num_numerical_features > 0:
            return torch.randn(batch_size, num_numerical_features, device=device)
        else:
            return None

    def _prepare_categorical_noise(self, batch_size, device):
        """Prepare noise for categorical features.

        Args:
            batch_size (int): The number of samples in the batch.
            device (torch.device): The device to create the noise tensor on.

        Returns:
            torch.Tensor: A tensor containing the categorical noise to apply in the probability space.
        """
        categorical_noise_list = []
        for cardinality in self.categorical_cardinality_list:
            uniform_probs = torch.ones(batch_size, cardinality, device=device)
            uniform_probs = uniform_probs / cardinality
            categorical_noise_list.append(uniform_probs)

        if len(categorical_noise_list) > 0:
            return torch.cat(categorical_noise_list, dim=1)
        else:
            return None

    def _apply_mixed_noise(self, x, t, noise_dict):
        """Apply mixed noise to the data.

        Args:
            x (torch.Tensor): The original data tensor.
            t (torch.Tensor): The time steps for each sample.
            noise_dict (dict): A dictionary containing the mixed noise tensors.

        Returns:
            torch.Tensor: A tensor containing the noisy data.
        """
        # Parse data
        # The index list is empty, the corresponding tensor is empty
        x_numerical = x[:, self.indices_numerical_list]
        x_categorical = x[:, self.indices_categorical_list]

        # Parse noise
        numerical_noise = noise_dict["numerical_noise"]
        categorical_noise = noise_dict["categorical_noise"]

        # Apply noise separately
        x_numerical_noisy = x_numerical
        x_categorical_noisy = x_categorical
        if numerical_noise is not None:
            x_numerical_noisy = self._apply_numerical_noise(x_numerical, t, numerical_noise)
        if categorical_noise is not None:
            x_categorical_noisy = self._apply_categorical_noise(x_categorical, t, categorical_noise)

        # Combine noisy features
        x_noisy = x.clone()
        x_noisy[:, self.indices_numerical_list] = x_numerical_noisy
        x_noisy[:, self.indices_categorical_list] = x_categorical_noisy

        return x_noisy

    def _apply_numerical_noise(self, x_numerical, t, noise):
        """Apply Gaussian noise to numerical features.

        Args:
            x_numerical (torch.Tensor): The original numerical features.
            t (torch.Tensor): The time steps for each sample.
            noise (torch.Tensor): The noise to apply.

        Returns:
            torch.Tensor: The noisy version of the numerical features.
        """
        # sqrt_alpha_cumprod_t is the scaling factor for the original data
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[t][:, None]
        # sqrt_one_minus_alpha_cumprod_t is the scaling factor for the noise
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t][:, None]
        # x_noisy is the noisy version of the input data at timestep t
        x_numerical_noisy = sqrt_alpha_cumprod_t * x_numerical + sqrt_one_minus_alpha_cumprod_t * noise

        return x_numerical_noisy

    def _apply_categorical_noise(self, x_categorical, t, noise):
        """Apply noise to categorical features in the probability space.

        Args:
            x_categorical (torch.Tensor): The original categorical features in one-hot encoding.
            t (torch.Tensor): The time steps for each sample.
            noise (torch.Tensor): The noise to apply in the probability space.

        Returns:
            torch.Tensor: The noisy version of the categorical features in probability space.
        """
        x_categorical_noisy = x_categorical.clone()

        # Get the current timestep's alpha and one_minus_alpha_cumprod
        alpha_cumprod_t = self.alpha_cumprod[t][:, None]
        one_minus_alpha_cumprod_t = self.one_minus_alpha_cumprod[t][:, None]

        # Add noise to the probabilities
        col_start = 0
        for cardinality in self.categorical_cardinality_list:
            col_end = col_start + cardinality

            # x_t = a_t * x_0 + (1 - a_t) * noise
            # No need to renormalise as convex combination of two distributions is still a valid distribution
            x_categorical_noisy[:, col_start:col_end] = (
                alpha_cumprod_t * x_categorical[:, col_start:col_end]
                + one_minus_alpha_cumprod_t * noise[:, col_start:col_end]
            )

            col_start = col_end

        return x_categorical_noisy

    def _prepare_posterior(self, x, x_noisy, t):
        """Prepare posterior data for computing categorical loss.

        Args:
            x (torch.Tensor): The original data tensor.
            x_noisy (torch.Tensor): The noisy data tensor.
            t (torch.Tensor): The time steps for each sample.

        Returns:
            dict: A dictionary containing the posterior data for categorical features.
        """
        # Parse data
        x_numerical = x[:, self.indices_numerical_list]
        x_categorical = x[:, self.indices_categorical_list]
        x_numerical_noisy = x_noisy[:, self.indices_numerical_list]
        x_categorical_noisy = x_noisy[:, self.indices_categorical_list]

        # Prepare posterior data separately
        posterior_dict = {}
        posterior_dict["numerical_posterior"] = self._prepare_numerical_posterior(
            x_numerical,
            x_numerical_noisy,
            t,
        )
        posterior_dict["categorical_posterior"] = self._prepare_categorical_posterior(
            x_categorical,
            x_categorical_noisy,
            t,
        )

        return posterior_dict

    def _prepare_numerical_posterior(self, x_numerical, x_numerical_noisy, t):
        """Prepare posterior data for numerical features."""
        # We simplify the loss computation for numerical features by predicting the noise directly
        # Hence, we do not need to prepare a posterior distribution for numerical features
        numerical_posterior = None

        return numerical_posterior

    def _prepare_categorical_posterior(self, x_categorical, x_categorical_noisy, t):
        """Prepare posterior data for categorical features."""
        # Return None if there are no categorical features
        if x_categorical.shape[1] == 0:
            return None

        # Get the current timestep's scaling factors
        alpha_t = self.alphas[t][:, None]
        one_minus_alpha_t = self.one_minus_alpha[t][:, None]
        alpha_cumprod_t = self.alpha_cumprod[t][:, None]
        one_minus_alpha_cumprod_t = self.one_minus_alpha_cumprod[t][:, None]

        # Prepare categorical posterior
        categorical_posterior_list = []
        col_start = 0
        for cardinality in self.categorical_cardinality_list:
            col_end = col_start + cardinality

            # q(x_{t-1} | x_0, x_t) = Cat(pi / Z)
            # Compute unnormalized log probabilities for the posterior
            pi = alpha_t * x_categorical_noisy[:, col_start:col_end] + one_minus_alpha_t / cardinality
            pi *= alpha_cumprod_t * x_categorical[:, col_start:col_end] + one_minus_alpha_cumprod_t / cardinality

            # Normalize to get probabilities
            categorical_posterior_one_feature = pi / pi.sum(dim=1, keepdim=True)
            categorical_posterior_list.append(categorical_posterior_one_feature)

            col_start = col_end

        categorical_posterior = torch.cat(categorical_posterior_list, dim=1)

        return categorical_posterior

    # ================================================================
    # =                                                              =
    # =                    Reverse diffusion                         =
    # =                                                              =
    # ================================================================
    def sample(self, num_samples, device):
        """Generate samples from the model."""
        # Start from pure noise (default Gaussian for numerical, uniform categorical for categorical)
        x = self._initalise_sampling(num_samples, device)

        for t in reversed(range(self.num_timesteps)):
            # Goes backwards from T-1 down to 0
            # Broadcast to set the same timestep t for all samples in the batch
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)

            # Predict noise
            # Tells the network "what noise level am I looking at?"
            x_num_noise_pred, x_cat_pred = self.denoising_network(
                x_num=x[:, self.indices_numerical_list],
                x_cat=x[:, self.indices_categorical_list],
                timesteps=t_batch,
            )
            denoise_output = torch.zeros_like(x, device=device)
            denoise_output[:, self.indices_numerical_list] = x_num_noise_pred
            denoise_output[:, self.indices_categorical_list] = x_cat_pred

            x_numerical_prev = self._reverse_numerical(
                t=t_batch,
                x_numerical_t=x[:, self.indices_numerical_list],
                noise_pred_t=denoise_output[:, self.indices_numerical_list],
            )
            x_categorical_prev = self._reverse_categorical(
                t=t_batch,
                x_categorical_pred_t=denoise_output[:, self.indices_categorical_list],
            )
            x[:, self.indices_numerical_list] = x_numerical_prev
            x[:, self.indices_categorical_list] = x_categorical_prev

        return x

    def _initalise_sampling(self, num_samples, device):
        x = torch.zeros(num_samples, self.num_features, device=device)
        noise_dict = self._prepare_mixed_noise(num_samples, device)
        if noise_dict["numerical_noise"] is not None:
            x[:, self.indices_numerical_list] = noise_dict["numerical_noise"]
        if noise_dict["categorical_noise"] is not None:
            x[:, self.indices_categorical_list] = noise_dict["categorical_noise"]

        return x

    def _reverse_numerical(self, t, x_numerical_t, noise_pred_t):
        # Get the current timestep's scaling factors
        sqrt_recip_alpha_t = self.sqrt_recip_alpha[t][:, None]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t][:, None]
        beta_t = self.betas[t][:, None]
        sqrt_beta_t = self.sqrt_beta[t][:, None]

        # Add sampling noise except for the last step
        if t[0] > 0:
            noise_level = sqrt_beta_t * torch.randn_like(x_numerical_t)
        else:
            noise_level = torch.zeros_like(x_numerical_t)

        # Reverse step to get x_{t-1} from x_t
        x_numerical_prev = (
            sqrt_recip_alpha_t * (x_numerical_t - beta_t / sqrt_one_minus_alpha_cumprod_t * noise_pred_t) + noise_level
        )

        return x_numerical_prev

    def _reverse_categorical(self, t, x_categorical_pred_t):
        """Reverse step for categorical features using predicted logits."""
        x_categorical_prev = x_categorical_pred_t.clone()

        col_start = 0
        for cardinality in self.categorical_cardinality_list:
            col_end = col_start + cardinality

            # Sample from the predicted categorical distribution
            # Use softmax for t > 0 to maintain uncertainty, argmax for t = 0 to get discrete values
            if t[0] > 0:
                x_categorical_prev[:, col_start:col_end] = F.softmax(
                    x_categorical_pred_t[:, col_start:col_end],
                    dim=1,
                )
            else:
                x_categorical_prev[:, col_start:col_end] = F.one_hot(
                    torch.argmax(x_categorical_pred_t[:, col_start:col_end], dim=1),
                    num_classes=cardinality,
                )

            col_start = col_end

        return x_categorical_prev
