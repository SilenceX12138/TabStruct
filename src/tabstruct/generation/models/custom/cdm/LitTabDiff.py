"""TabDiff: A Multi-modal Diffusion Model for Tabular Data Generation.

This module implements TabDiff, extending the EDM framework with support for both
numerical and categorical features via a unified continuous-time diffusion process.

Key Features:
    - Power-mean noise schedule for numerical features (Karras et al.)
    - Log-linear absorbing diffusion for categorical features
    - EDM-style preconditioning for stable training
    - Mixed loss combining continuous MSE and discrete cross-entropy

Architecture Overview:
    The model consists of three layers:
    1. LitTabDiff: High-level TabStruct integration (data scaling, hyperparams)
    2. _LitTabDiff: PyTorch Lightning training loop (loss, optimization)
    3. TabDiff: Core diffusion model (forward/reverse processes)

    Supporting classes:
    - PowerMeanNoise: Polynomial noise schedule for continuous features
    - LogLinearNoise: Log-linear schedule for absorbing diffusion
    - DenoiseModel: EDM preconditioning wrapper (from .utils.modules)
    - UniModMLP: Multimodal denoising network (from ..ddpm.utils.modules)

Unified Continuous-Time Diffusion:
    TabDiff synchronizes numerical and categorical diffusion via shared time t:
    1. Sample σ_num ~ LogNormal(p_mean, p_std)
    2. Compute t = inverse_schedule(σ_num) ∈ [0,1]
    3. Compute σ_cat = log_linear_schedule(t)
    4. Apply Gaussian noise to numerical: x_t = x + σ_num * ε
    5. Apply masking to categorical: x_t ~ Bernoulli(1 - exp(-σ_cat))

    This ensures both modalities are corrupted in a coordinated way.

Data Preprocessing:
    - Numerical: Quantile transform → uniform/Gaussian distribution
    - Categorical: Ordinal encoding → {0, 1, ..., n-1} + mask token at n

References:
    - Official TabDiff: https://github.com/MinkaiXu/TabDiff
    - EDM: Karras et al. "Elucidating the Design Space of Diffusion Models"
    - Absorbing Diffusion: Austin et al. "Structured Denoising Diffusion Models"
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tabeval.plugins.core.models.tabular_encoder import TabularEncoder
from torch.distributions import Categorical

from ...BaseGenerator import BaseLightningGenerationModule, BaseLitJointGenerator
from ..ddpm.utils.modules import UniModMLP
from .utils.diffusion import LogLinearNoise, PowerMeanNoise
from .utils.modules import DenoiseModel


class LitTabDiff(BaseLitJointGenerator):
    """Lightning wrapper for TabDiff model integrating with TabStruct framework.

    This class handles high-level model configuration, data scaling, and hyperparameter
    management. It follows the TabStruct pattern of separating framework integration
    (this class) from core model logic (_LitTabDiff and TabDiff).

    The class is responsible for:
        - Task validation (classification, regression, unsupervised)
        - Data preprocessing configuration (quantile + ordinal encoding)
        - Hyperparameter management (noise schedules, sampling steps, etc.)
        - Model instantiation delegation to _LitTabDiff

    Args:
        args: Arguments object containing task, dataset info, and hyperparameters
    """

    def __init__(self, args):
        super().__init__(args)

        if args.task not in ["classification", "regression", "unsupervision"]:
            raise ValueError(f"Task {args.task} is not supported for {self.name} model")

        self.model = _LitTabDiff(args)

    # ================================================================
    # =                     Model-specific                           =
    # ================================================================
    def _prepare_data_scalers(self):
        """Configure data preprocessing scalers for TabDiff.

        TabDiff requires specific preprocessing:
        - Numerical: Quantile transform to uniform/Gaussian distribution
          This normalizes skewed distributions and provides stable ranges for diffusion
        - Categorical: Ordinal encoding (0, 1, 2, ..., n-1)
          Required for absorbing diffusion which adds a mask token at index n

        The same encoding is applied to targets in supervised tasks to enable
        joint diffusion of features and labels.
        """
        # Feature scaler: quantile transform maps data to uniform [0,1] or Gaussian distribution
        # This provides a stable range for diffusion and helps with numerical stability
        # Note: TabDiff expects categorical features as ordinal integers
        self.feature_scaler = TabularEncoder(
            categorical_encoder="ordinal",
            cat_encoder_params={
                "handle_unknown": "use_encoded_value",
                "unknown_value": -1,  # Unseen categories encoded as -1
            },
            continuous_encoder="quantile",
            cont_encoder_params={},
        )

        # Target scaler: same approach for supervised tasks
        # Ensures targets participate in diffusion without scale mismatch
        self.target_scaler = None
        if self.args.task != "unsupervision":
            self.target_scaler = TabularEncoder(
                categorical_encoder="ordinal",
                cat_encoder_params={
                    "handle_unknown": "use_encoded_value",
                    "unknown_value": -1,
                },
                continuous_encoder="quantile",
                cont_encoder_params={},
            )

    # ================================================================
    # =                      Hyperparams                             =
    # ================================================================
    @classmethod
    def _define_default_params(cls):
        """Define default hyperparameters for full-scale training.

        These values are optimized for high-quality generation on real datasets,
        balancing computational cost with sample fidelity.

        Returns:
            Dictionary with 'architecture' and 'optimization' parameter groups
        """
        params_arch = {
            # === Noise Schedule Parameters ===
            "sigma_min": 0.002,  # Minimum noise level (near-clean data)
            "sigma_max": 80.0,  # Maximum noise level (high corruption)
            "rho": 7.0,  # Time discretization exponent (7 = focus on low noise)
            "num_steps": 50,  # Number of sampling steps (higher = better quality)
            # === Preconditioning Parameters ===
            "sigma_data": 1.0,  # Expected data std (match your normalized data)
            # === Training Noise Distribution ===
            "p_mean": 0.0,  # Mean of log(σ) ~ N(0, 1) → σ ~ LogNormal(0, 1)
            "p_std": 1.0,  # Std of log(σ) - controls noise level diversity
            # === Sampling Options ===
            "enable_heun_correction": False,  # Second-order Heun correction (can improve accuracy but slower)
        }

        params_optim = {
            "lr": 1e-3,  # Learning rate (EDM is fairly robust to this)
            "weight_decay": 1e-5,  # Regularization (light, as EDM already well-conditioned)
        }

        return {"architecture": params_arch, "optimization": params_optim}

    @classmethod
    def _define_single_run_params(cls):
        """Lighter configuration for quick development runs.

        Reduces sampling steps and noise range for faster iteration during
        development and debugging. Quality may be slightly lower than default.

        Returns:
            Dictionary with 'architecture' and 'optimization' parameter groups
        """

        params_arch = {
            "sigma_min": 0.005,  # Slightly higher min noise (less aggressive denoising)
            "sigma_max": 50.0,  # Lower max noise (reduced corruption range)
            "rho": 5.0,  # Lower rho (more uniform step allocation)
            "num_steps": 24,  # Fewer steps for faster sampling (60% of default)
            "sigma_data": 1.0,
            "p_mean": 0.0,
            "p_std": 1.0,
            "enable_heun_correction": False,  # Second-order Heun correction disabled for faster sampling
        }

        params_optim = {
            "lr": 1e-3,
            "weight_decay": 1e-5,
        }

        return {"architecture": params_arch, "optimization": params_optim}

    @classmethod
    def _define_test_params(cls):
        """Minimal configuration used in CI / unit tests.

        Prioritizes speed over quality for rapid automated testing.
        Uses minimal steps and narrow noise range.

        Returns:
            Dictionary with 'architecture' and 'optimization' parameter groups
        """

        params_arch = {
            "sigma_min": 0.01,  # Higher min for faster convergence
            "sigma_max": 10.0,  # Much lower max noise (minimal corruption)
            "rho": 4.0,  # Lower rho (linear-like schedule)
            "num_steps": 12,  # Minimal steps (30% of default)
            "sigma_data": 1.0,
            "p_mean": 0.0,
            "p_std": 1.0,
            "enable_heun_correction": False,  # Second-order Heun correction disabled for faster sampling
        }

        params_optim = {
            "lr": 1e-3,
            "weight_decay": 1e-5,
        }

        return {"architecture": params_arch, "optimization": params_optim}

    @classmethod
    def _get_model_specific_scaler_config(cls):
        """Return scaler configuration specific to TabDiff requirements.

        TabDiff requires ordinal encoding for categorical features to support
        absorbing diffusion with mask tokens. Features remain discrete during
        the diffusion process (not treated as continuous).

        Returns:
            Dictionary with scaler configuration for context, features, and targets
        """
        return {
            "context": {
                "disable_preprocessing": False,  # Keep preprocessing pipeline active
            },
            "feature_scaler": {
                "categorical_transform": "ordinal",  # Map categories to ordinal values
                "categorical_as_numerical": False,  # Keep as discrete (not continuous)
            },
            "target_scaler": {},
        }


class _LitTabDiff(BaseLightningGenerationModule):
    """Core Lightning module for TabDiff training and generation.

    Handles model instantiation, loss computation, and sample generation within
    the PyTorch Lightning training loop.

    This internal class bridges the TabStruct framework with the core TabDiff
    implementation, managing:
        - Model architecture creation with proper feature indexing
        - Training loop with mixed numerical/categorical losses
        - Generation interface for sampling synthetic data

    Args:
        args: Arguments object containing model params and dataset metadata
    """

    def __init__(self, args):
        super().__init__(args)

    # ================================================================
    # =                     Model-specific                           =
    # ================================================================
    def _create_torch_model(self):
        """Instantiate the TabDiff model with mixed numerical/categorical architecture.

        Architecture Stack:
            1. UniModMLP: Transformer-MLP hybrid denoising network
            2. DenoiseModel: EDM preconditioning wrapper
            3. TabDiff: Core diffusion logic with mixed noise schedules

        Returns:
            TabDiff model instance configured for the current dataset
        """
        # Calculate total dimension: features + target (for supervised tasks)
        input_dim = len(self.args.full_feature_list_model) + len(self.args.full_target_list_model)

        # Add mask token to each categorical feature's vocabulary
        # Mask tokens enable absorbing diffusion (categorical analogue of Gaussian noise)
        categories_with_mask = [card + 1 for card in self.args.full_cardinality_list_model]

        # Build the multimodal denoiser matching TabDiff's mixed diffusion
        # UniModMLP tokenizes features and processes them through transformer layers
        score_net = UniModMLP(
            d_numerical=len(self.args.full_indices_numerical_list_model),
            categories=categories_with_mask,
            num_layers=2,
        )

        # Wrap with EDM-style preconditioning to stabilise training
        # Preconditioning applies noise-dependent input/output scaling for better gradients
        denoise_model = DenoiseModel(
            denoise_fn=score_net,
            sigma_data=self.args.model_params["architecture"]["sigma_data"],
            precond=True,
            net_conditioning="sigma",  # Use log(σ)/4 as timestep embedding
        )

        # Instantiate TabDiff with mixed continuous/discrete diffusion
        model = TabDiff(
            num_features=input_dim,
            numerical_idx=self.args.full_indices_numerical_list_model,
            categorical_idx=self.args.full_indices_categorical_list_model,
            full_cardinality_list=self.args.full_cardinality_list_model,
            train_cardinality_list=self.args.train_cardinality_list_model,
            sigma_min=self.args.model_params["architecture"]["sigma_min"],
            sigma_max=self.args.model_params["architecture"]["sigma_max"],
            rho=self.args.model_params["architecture"]["rho"],
            num_steps=self.args.model_params["architecture"]["num_steps"],
            sigma_data=self.args.model_params["architecture"]["sigma_data"],
            p_mean=self.args.model_params["architecture"]["p_mean"],
            p_std=self.args.model_params["architecture"]["p_std"],
            denoise_model=denoise_model,
            enable_heun_correction=self.args.model_params["architecture"]["enable_heun_correction"],
        )

        return model

    def _compute_loss(self, data_real: torch.Tensor, forward_dict: dict) -> dict:
        """Combine numerical and categorical losses from the forward pass.

        The total loss is a simple sum of weighted MSE (numerical) and
        weighted cross-entropy (categorical) losses. Each component already
        includes appropriate noise-level weighting from EDM and ELBO respectively.

        Args:
            data_real: Ground truth data [batch_size, num_features] (unused here)
            forward_dict: Dictionary containing:
                - 'numerical_loss': Per-sample numerical losses [batch]
                - 'categorical_loss': Per-sample categorical losses [batch]

        Returns:
            Dictionary with total_loss and component losses for logging:
                - 'total_loss': Combined scalar loss for optimization
                - 'numerical_loss': Mean numerical loss (for monitoring)
                - 'categorical_loss': Mean categorical loss (for monitoring)
        """
        num_loss = forward_dict["numerical_loss"]
        cat_loss = forward_dict["categorical_loss"]
        total_loss = num_loss + cat_loss

        return {
            "total_loss": total_loss,
            "numerical_loss": num_loss,
            "categorical_loss": cat_loss,
        }

    def _generate(self, num_samples: int) -> torch.Tensor:
        """Generate synthetic samples via reverse diffusion sampling.

        Args:
            num_samples: Number of synthetic samples to generate

        Returns:
            Generated samples [num_samples, num_features] in scaled space
            (quantile-transformed numerical + ordinal categorical)
        """
        return self.torch_model.sample(num_samples, device=self.device)


class TabDiff(nn.Module):
    """Core TabDiff diffusion model with mixed continuous/discrete noise.

    Implements unified continuous-time diffusion for tabular data by combining:
    - Power-mean noise schedule for numerical features (Karras et al.)
    - Log-linear absorbing diffusion for categorical features
    - EDM-style preconditioning and loss weighting

    Training:
        1. Sample noise level σ ~ LogNormal(p_mean, p_std)
        2. Add Gaussian noise to numerical features: x_num_t = x_num + σ * ε
        3. Mask categorical features with probability 1 - exp(-σ)
        4. Predict clean data from noisy observations
        5. Compute weighted MSE (numerical) + cross-entropy (categorical) losses

    Sampling:
        1. Initialize from maximum noise (Gaussian + all masked)
        2. Build Karras schedule: [σ_max, ..., σ_1, 0]
        3. For each step: denoise and update numerical/categorical separately
        4. Return generated samples at σ = 0
    """

    def __init__(
        self,
        num_features: int,
        numerical_idx: list[int],
        categorical_idx: list[int],
        full_cardinality_list: list[int],
        train_cardinality_list: list[int],
        sigma_min: float,
        sigma_max: float,
        rho: float,
        num_steps: int,
        sigma_data: float,
        p_mean: float,
        p_std: float,
        denoise_model: DenoiseModel,
        enable_heun_correction: bool,
    ):
        """Initialize TabDiff diffusion model.

        Args:
            num_features: Total number of features (numerical + categorical)
            numerical_idx: Indices of numerical features in the full feature vector
            categorical_idx: Indices of categorical features in the full feature vector
            full_cardinality_list: Number of categories for each categorical feature (full dataset)
            train_cardinality_list: Number of categories observed in training set for each categorical feature
            sigma_min: Minimum noise level for numerical features (near-clean)
            sigma_max: Maximum noise level for numerical features (high corruption)
            rho: Polynomial exponent for time discretization (higher = focus on low noise)
            num_steps: Number of discretization steps for sampling
            sigma_data: Expected standard deviation of normalized data (for preconditioning)
            p_mean: Mean of log-normal noise distribution for training
            p_std: Std of log-normal noise distribution for training
            denoise_model: Preconditioned denoising network (EDM-style)
            enable_heun_correction: Whether to use second-order Heun correction in sampling
        """
        super().__init__()

        # === Data layout metadata ===
        self.num_features = int(num_features)
        self.numerical_idx = list(numerical_idx)
        self.categorical_idx = list(categorical_idx)
        self.num_numerical = len(self.numerical_idx)
        self.num_categorical = len(self.categorical_idx)
        self.full_cardinality_list = [int(c) for c in full_cardinality_list]
        self.train_cardinality_list = [int(c) for c in train_cardinality_list]
        self.enable_heun_correction = enable_heun_correction

        # === Categorical mask tokens ===
        # mask_index[i] = cardinality of feature i (index of the mask token in one-hot)
        # Used in absorbing diffusion to represent the "unknown" state
        if self.num_categorical > 0:
            mask_index = torch.tensor(self.full_cardinality_list, dtype=torch.long)
        else:
            mask_index = torch.zeros(0, dtype=torch.long)
        self.register_buffer("mask_index", mask_index, persistent=False)
        self.register_buffer("neg_infinity", torch.tensor(-1e6, dtype=torch.float32), persistent=False)

        # Number of classes including mask token for each categorical feature
        self.num_classes_w_mask = [c + 1 for c in self.full_cardinality_list]

        # === Core denoising network ===
        self.denoise_model = denoise_model

        # === Noise schedule hyperparameters ===
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.rho = float(rho)
        self.num_steps = int(num_steps)
        self.sigma_data = float(sigma_data)
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)

        # === Noise schedules ===
        # PowerMeanNoise: polynomial schedule for numerical features (Karras et al.)
        # LogLinearNoise: log-linear schedule for categorical absorbing diffusion
        self.num_schedule = PowerMeanNoise(self.sigma_min, self.sigma_max, self.rho)
        self.cat_schedule = LogLinearNoise(self.num_categorical)

        # === Loss weighting and sampling parameters ===
        # Noise distribution for training: log(σ) ~ N(P_mean, P_std²)
        self.noise_dist_params = {"P_mean": self.p_mean, "P_std": self.p_std}
        # EDM preconditioning: matches expected data scale
        self.edm_params = {"sigma_data": self.sigma_data}
        # Stochastic churn for sampling: controls exploration vs exploitation
        self.sampler_params = {
            "sigma_churn": 1.0,  # Churn strength (0=deterministic, >0=stochastic)
            "sigma_churn_min": self.sigma_min,  # Only apply churn within this range
            "sigma_churn_max": self.sigma_max,
        }

    # =============================
    # Forward diffusion / training
    # =============================
    def forward(self, x0: torch.Tensor) -> dict:
        """Training forward pass: compute mixed numerical + categorical losses.

        Process:
            1. Split data into numerical and categorical parts
            2. Sample unified noise level σ ~ LogNormal
            3. Add Gaussian noise to numerical features
            4. Apply absorbing diffusion (masking) to categorical features
            5. Predict clean data from noisy observations
            6. Compute weighted MSE (num) + cross-entropy (cat) losses

        Args:
            x0: Clean data [batch_size, num_features]

        Returns:
            Dictionary with separate loss components for logging
        """
        # === Setup ===
        batch_size = x0.shape[0]
        device = x0.device

        # === Extract numerical and categorical parts ===
        x_num = self._extract_numerical(x0)
        x_cat_ordinal = self._extract_categorical(x0)

        # === Sample unified noise level and compute schedules ===
        sigma_dict = self._build_forward_sigma(batch_size, device)
        sigma_num, sigma_cat = sigma_dict["sigma_num"], sigma_dict["sigma_cat"]

        # === Forward diffusion: add noise to features ===
        x_num_t = self._apply_numerical_noise(x_num, sigma_num)
        x_cat_ordinal_t, x_cat_onehot_t = self._apply_categorical_noise(x_cat_ordinal, sigma_cat)

        # === Predict clean data from noisy observations ===
        x_num_denoised, x_cat_onehot_denoised = self._denoise(x_num_t, x_cat_onehot_t, sigma_num)

        # === Compute losses ===
        num_loss = self._numerical_loss(x_num_denoised, x_num, sigma_num)
        cat_loss = self._categorical_loss(x_cat_onehot_denoised, x_cat_ordinal, x_cat_ordinal_t, sigma_cat)

        return {
            "categorical_loss": cat_loss,
            "numerical_loss": num_loss,
        }

    # =============================
    # Sampling
    # =============================
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Generate samples via EDM-inspired reverse diffusion sampling.

        Implements the EDM sampling algorithm with stochastic churn and second-order
        correction, adapted for mixed numerical/categorical data via unified time schedules.

        Algorithm Overview:
            1. Build Karras polynomial noise schedule for numerical features
            2. Map numerical schedule to unified time t ∈ [0,1]
            3. Derive synchronized categorical noise schedule via t
            4. Initialize from maximum noise prior (Gaussian + fully masked)
            5. Iteratively denoise using EDM update with optional churn injection
            6. Return clean samples at σ = 0

        Stochastic Churn (S_churn > 0):
            - Temporarily increases noise level by factor γ at each step
            - γ = min(S_churn/steps, √2-1) within [S_tmin, S_tmax] range
            - Improves sample diversity and exploration of probability flow ODE
            - Set S_churn=0 for deterministic sampling (probability flow)

        Args:
            num_samples: Number of synthetic samples to generate
            device: Device for tensor allocation (CPU/CUDA)

        Returns:
            Generated samples [num_samples, num_features] with:
                - Numerical features: continuous values in quantile-normalized space
                - Categorical features: ordinal indices {0, 1, ..., n-1}
        """
        # === Step 1: Build noise schedules for numerical and categorical features ===
        sigma_num, sigma_cat = self._build_sampling_sigma(device)

        # === Step 2: Initialize from maximum noise prior ===
        x_num, x_cat = self._init_samples(num_samples, sigma_num, device)

        # === Step 3: Iterative denoising loop from σ_max → 0 ===
        # Each iteration reduces noise level, gradually revealing clean data
        total_reverse_steps = len(sigma_num) - 1
        for i in range(total_reverse_steps):
            # Current and target noise levels for this step
            # We move from σ[i] (higher noise) to σ[i+1] (lower noise)
            sigma_num_cur, sigma_num_next = sigma_num[i], sigma_num[i + 1]
            sigma_cat_cur, sigma_cat_next = sigma_cat[i], sigma_cat[i + 1]

            # Apply stochastic churn to increase noise temporarily (if S_churn > 0)
            # This creates σ̂ ≥ σ_cur, improving sample diversity via exploration
            sigma_num_hat, sigma_cat_hat = self._build_sampling_sigma_churn(
                sigma_num_cur,
                sigma_cat_cur,
                total_reverse_steps,
            )

            # Perform one EDM update step with numerical and categorical updates
            # This moves samples from (x_t, σ_cur) to (x_t-1, σ_next)
            x_num, x_cat = self._sample_step(
                x_num,
                x_cat,
                sigma_num_cur,
                sigma_num_next,
                sigma_num_hat,
                sigma_cat_cur,
                sigma_cat_next,
                sigma_cat_hat,
            )

        # === Step 4: Assemble final samples by placing features at correct indices ===
        samples = torch.zeros(num_samples, self.num_features, device=device)
        if self.num_numerical > 0:
            samples[:, self.numerical_idx] = x_num
        if self.num_categorical > 0:
            samples[:, self.categorical_idx] = x_cat.float()

        return samples

    # =============================
    # Diffusion utilities
    # =============================
    def _extract_numerical(self, x: torch.Tensor) -> torch.Tensor | None:
        """Extract numerical features from mixed feature vector.

        Args:
            x: Mixed feature tensor [batch, num_features]

        Returns:
            Numerical features [batch, num_numerical] or None if no numerical features
        """
        x_num = None
        if self.num_numerical > 0:
            x_num = x[:, self.numerical_idx]

        return x_num

    def _extract_categorical(self, x: torch.Tensor) -> torch.Tensor | None:
        """Extract categorical features from mixed feature vector.

        Args:
            x: Mixed feature tensor [batch, num_features]

        Returns:
            Categorical features [batch, num_categorical] as long tensor,
            or None if no categorical features
        """
        x_cat_ordinal = None
        if self.num_categorical > 0:
            x_cat_ordinal = x[:, self.categorical_idx].long()
            # Handle unseen categories: ordinal encoder uses -1 for unknown values
            # Clamp to 0 to avoid indexing errors
            # They will be ignored in loss computation as they are from valid/test samples
            x_cat_ordinal = torch.clamp(x_cat_ordinal, min=0)

        return x_cat_ordinal

    def _build_forward_sigma(self, batch_size: int, device: torch.device) -> dict:
        """Sample unified noise level for both numerical and categorical features.

        The key insight of TabDiff is using a unified continuous-time diffusion:
        1. Sample σ_num ~ LogNormal for numerical features
        2. Convert to normalized time t ∈ [0,1] via inverse schedule
        3. Use same t to compute σ_cat for categorical features
        This ensures synchronized corruption across modalities.

        Args:
            batch_size: Number of samples in the batch
            device: Device for tensor allocation

        Returns:
            Dictionary containing:
                - sigma_num: Numerical noise level [batch, 1]
                - sigma_cat: Categorical noise level [batch, num_categorical]
        """
        # === Sample sigma for numerical features ===
        # Sample σ ~ LogNormal(p_mean, p_std) for numerical features
        sigma_num = self._build_forward_sigma_num(batch_size, device)

        # === Sample sigma for categorical features ===
        sigma_cat = self._build_synced_sigma_cat(sigma_num)

        return {
            "sigma_num": sigma_num,
            "sigma_cat": sigma_cat,
        }

    def _build_forward_sigma_num(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample noise levels from log-normal distribution.

        σ ~ LogNormal(p_mean, p_std²) provides better coverage of noise scales
        compared to uniform sampling, as recommended in EDM (Karras et al.).

        With p_mean=0 and p_std=1 (default), log(σ) ~ N(0,1), so σ spans
        roughly [0.1, 10] with most mass around 1. The clamping to [σ_min, σ_max]
        ensures we stay within the schedule's valid range.

        Args:
            batch_size: Number of noise levels to sample
            device: Device for tensor allocation

        Returns:
            Sampled noise levels [batch_size, 1]
        """
        z = torch.randn(batch_size, 1, device=device)

        # log(σ) = p_mean + p_std * z → σ = exp(p_mean + p_std * z)
        # Coverage: LogNormal(0,1) gives 99.99%-ile at σ≈41, P(σ>80)≈0.0006%
        sigma = (self.noise_dist_params["P_mean"] + z * self.noise_dist_params["P_std"]).exp()
        # Clamp to valid schedule range to avoid extrapolation
        sigma = sigma.clamp(min=self.sigma_min, max=self.sigma_max)

        return sigma

    def _build_synced_sigma_cat(self, sigma_num: torch.Tensor) -> torch.Tensor | None:
        """Derive synchronized categorical noise schedule from numerical schedule.

        Uses the unified time mapping to ensure both modalities are corrupted
        in a synchronized manner, improving coherence in generated samples.

        Args:
            sigma_num: Numerical noise schedule [num_steps]

        Args:
            sigma_num: Numerical noise schedule [num_steps] or [num_steps+1] (with σ=0 appended)

        Returns:
            Categorical noise schedule [num_steps, num_categorical] or [num_steps+1, num_categorical],
            or None if no categorical features
        """
        # Step 1: Map numerical noise levels to unified normalized time t ∈ [0,1]
        # This enables synchronized corruption across numerical and categorical modalities
        # Note: When σ=0 is appended to sigma_num (during sampling), we compute t for all values
        t_schedule = self.num_schedule.inverse_to_t(sigma_num)

        # Step 2: Derive categorical noise schedule from unified time
        # Uses log-linear absorbing schedule: σ_cat(t) = -log(1-t)
        sigma_cat = self.cat_schedule.total_noise(t_schedule)

        return sigma_cat

    def _apply_numerical_noise(self, x_num: torch.Tensor | None, sigma_num: torch.Tensor) -> torch.Tensor | None:
        """Apply Gaussian noise to numerical features.

        Forward diffusion: x_t = x_0 + σ * ε where ε ~ N(0, I)

        Args:
            x_num: Clean numerical features [batch, num_numerical] or None
            sigma_num: Noise level [batch, 1]

        Returns:
            Noisy numerical features [batch, num_numerical] or None
        """
        x_num_t = x_num
        if self.num_numerical > 0:
            noise = torch.randn_like(x_num)
            x_num_t = x_num_t + noise * sigma_num

        return x_num_t

    def _apply_categorical_noise(
        self,
        x_cat_ordinal: torch.Tensor | None,
        sigma_cat: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Apply absorbing diffusion (masking) to categorical features.

        Absorbing diffusion replaces category values with a special mask token
        with probability p = 1 - exp(-σ). As σ increases, more features are masked.

        Args:
            x_cat_ordinal: Clean categorical features [batch, num_categorical] or None
            sigma_cat: Noise level [batch, num_categorical]

        Returns:
            Tuple of:
                - x_cat_ordinal_t: Noisy categorical indices [batch, num_categorical]
                - x_cat_onehot_t: One-hot encoded version [batch, sum(num_classes_w_mask)]
            Both are None if no categorical features exist.
        """
        x_cat_ordinal_t = x_cat_ordinal
        x_cat_onehot_t = x_cat_ordinal
        if self.num_categorical > 0:
            # move_chance = 1 - α(t) where α(t) = exp(-σ(t))
            # Higher σ → higher masking probability
            # expm1 for numerical stability: 1 - exp(-σ) = -expm1(-σ)
            move_chance = -torch.expm1(-sigma_cat)

            # Sample Bernoulli mask: whether each feature transitions to mask token
            bern_sample = torch.rand_like(move_chance)
            move_mask = bern_sample < move_chance

            # Apply mask: replace with mask_index where move_mask is True
            # mask_index broadcasts to [batch_size, num_categorical]
            mask = self.mask_index.view(1, -1)
            x_cat_ordinal_t = torch.where(move_mask, mask, x_cat_ordinal)

            # Convert to one-hot for network processing
            x_cat_onehot_t = self._convert_to_one_hot_with_mask(x_cat_ordinal_t)

        return x_cat_ordinal_t, x_cat_onehot_t

    def _convert_to_one_hot_with_mask(self, x_cat_ordinal: torch.Tensor) -> torch.Tensor:
        """Convert categorical indices to one-hot encoding.

        Each categorical feature is independently one-hot encoded with
        (cardinality + 1) classes to include the mask token.

        Args:
            x_cat: Categorical indices [batch, num_categorical]

        Returns:
            Concatenated one-hot vectors [batch, sum(num_classes_w_mask)]
        """
        one_hot_per_feature = []
        for i, card in enumerate(self.full_cardinality_list):
            # Include mask token
            num_classes = card + 1
            values = x_cat_ordinal[:, i].long()
            # Clamp to valid range to avoid indexing errors
            values = torch.clamp(values, min=0)
            one_hot_per_feature.append(F.one_hot(values, num_classes=num_classes))

        return torch.cat(one_hot_per_feature, dim=-1).float()

    def _denoise(
        self,
        x_num: torch.Tensor | None,
        x_cat: torch.Tensor | None,
        sigma: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Call the denoising network with appropriate conditioning.

        The denoising network predicts clean data D_θ(x_t, σ) from noisy
        observations x_t. Uses EDM-style preconditioning for stable training.

        Args:
            x_num: Noisy numerical features [batch, num_numerical] or None
            x_cat: Noisy categorical one-hot [batch, sum(num_classes_w_mask)] or None
            sigma: Noise level [batch, 1] (numerical schedule used for conditioning)

        Returns:
            Tuple of:
                - x_num_denoised: Predicted clean numerical [batch, num_numerical]
                - x_cat_denoised: Predicted clean categorical logits [batch, sum(num_classes_w_mask)]

        Note:
            Currently uses sigma_num for all features. Future work could explore
            using sigma_cat for categorical conditioning.
        """
        # Ensure sigma has shape [batch_size, 1] for proper broadcasting
        sigma_expanded = sigma.view(-1, 1)

        # Call the preconditioned denoising network
        # Note: TabDiff uses only sigma_num for conditioning, sigma_cat could be explored
        x_num_denoised, x_cat_denoised = self.denoise_model(x_num=x_num, x_cat=x_cat, t=None, sigma=sigma_expanded)

        return x_num_denoised, x_cat_denoised

    def _numerical_loss(
        self,
        x_num_syn: torch.Tensor | None,
        x_num_real: torch.Tensor | None,
        sigma_num: torch.Tensor,
    ) -> torch.Tensor:
        """Compute EDM weighted MSE loss for numerical features.

        Loss: λ(σ) * ||D_θ(x_noisy, σ) - x_0||²
        where λ(σ) = (σ² + σ_data²) / (σ * σ_data)²

        This adaptive weighting balances contributions across noise levels and
        improves training stability compared to unweighted MSE. The weighting
        downweights high noise levels where the task is easier, and upweights
        low noise levels where fine details matter.

        Args:
            x_num_syn: Predicted clean data [batch, num_numerical] or None
            x_num_real: Ground truth [batch, num_numerical] or None
            sigma_num: Noise levels [batch, 1]

        Returns:
            Scalar loss (mean over batch)
        """
        loss = torch.zeros(1, device=x_num_real.device)

        if self.num_numerical > 0:
            # Compute squared terms for EDM weighting formula
            sigma2 = sigma_num**2  # σ²
            sigma_data2 = self.edm_params["sigma_data"] ** 2  # σ_data²

            # Apply EDM weighting formula: λ(σ) = (σ² + σ_data²) / (σ · σ_data)²
            # This balances gradient magnitudes across noise levels:
            # - At high σ: weight ≈ 1/σ_data² (downweight easy high-noise predictions)
            # - At low σ: weight ≈ σ_data²/σ² (upweight difficult low-noise predictions)
            weight = (sigma2 + sigma_data2) / torch.clamp((sigma_num * self.edm_params["sigma_data"]) ** 2, min=1e-6)
            mse = F.mse_loss(x_num_syn, x_num_real, reduction="none")
            loss = (weight * mse).sum() / x_num_real.shape[0]

        return loss

    def _categorical_loss(
        self,
        x_cat_onehot_syn: torch.Tensor | None,
        x_cat_ordinal_real: torch.Tensor | None,
        x_cat_ordinal_t: torch.Tensor | None,
        sigma_cat: torch.Tensor,
    ) -> torch.Tensor:
        """Compute absorbing diffusion loss for categorical features.

        Combines substitution parameterization (ensuring proper probability
        distribution) with weighted cross-entropy loss (ELBO for absorbing diffusion).

        Args:
            x_cat_onehot_syn: Predicted logits [batch, sum(num_classes_w_mask)] or None
            x_cat_ordinal_real: Ground truth indices [batch, num_categorical] or None
            x_cat_ordinal_t: Noisy indices [batch, num_categorical] or None
            sigma_cat: Noise level [batch, num_categorical]

        Returns:
            Scalar loss (mean over batch), zero if no categorical features
        """
        loss = torch.zeros(1, device=x_cat_ordinal_real.device)

        if self.num_categorical > 0:
            logits = self._adjust_log_probabilities(x_cat_onehot_syn, x_cat_ordinal_t)
            loss = self._absorbing_loss(logits, x_cat_ordinal_real, sigma_cat)

        return loss

    def _adjust_log_probabilities(self, unnormalized_prob: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
        """Apply substitution parameterization for absorbing diffusion.

        This is a key technique to ensure the model learns a proper probability distribution
        while respecting the structure of partially observed data.

        For masked tokens in xt, the network predicts a distribution over all classes.
        For unmasked tokens, we force the probability to be 1 on the known value.

        This parameterization ensures:
        1. Mask token index has zero probability (set to -inf in log space)
        2. Known (unmasked) values have probability 1 (all others set to -inf)
        3. Unknown (masked) values use the network's predicted distribution

        Args:
            unnormalized_prob: Raw network outputs [batch, sum(num_classes_w_mask)]
            xt: Noisy categorical indices [batch, num_categorical]

        Returns:
            Log probabilities [batch, sum(num_classes_w_mask)]
        """
        logits_split = torch.split(unnormalized_prob, self.num_classes_w_mask, dim=-1)
        xt_split = torch.split(xt, 1, dim=-1)

        processed = []
        for i, (logit, xt_col) in enumerate(zip(logits_split, xt_split)):
            logit = logit.clone()
            mask_idx = self.mask_index[i].item() if self.num_categorical > 0 else 0

            # Set mask token probability to zero
            logit[:, mask_idx] = self.neg_infinity

            # Normalize to log probabilities
            log_prob = logit - torch.logsumexp(logit, dim=-1, keepdim=True)

            # For unmasked tokens, force probability 1 on known value
            xt_values = xt_col.squeeze(-1)
            if self.num_categorical > 0:
                unmasked = xt_values != mask_idx
                if unmasked.any():
                    idx = unmasked.nonzero(as_tuple=True)[0]
                    log_prob[idx] = self.neg_infinity
                    log_prob[idx, xt_values[idx]] = 0.0  # log(1) = 0
            processed.append(log_prob)

        return torch.cat(processed, dim=-1)

    def _absorbing_loss(
        self,
        model_output: torch.Tensor,
        x0: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted cross-entropy loss for absorbing diffusion.

        The ELBO for absorbing diffusion involves:
        L = -weight(t) * log p_θ(x_0 | x_t)

        where weight(t) = 1 / (1 - α) where α = exp(-σ) balances contributions across noise levels.

        Args:
            model_output: Log probabilities [batch, sum(num_classes_w_mask)]
            x0: Ground truth indices [batch, num_categorical]
            sigma: Noise levels [batch, num_categorical]

        Returns:
            Scalar loss averaged over batch and features
        """
        logits_split = torch.split(model_output, self.num_classes_w_mask, dim=-1)
        losses = []
        for i, logits in enumerate(logits_split):
            target = x0[:, i].long()

            # Extract log probability of ground truth class for each sample
            # This is the reconstruction term: log p_θ(x_0 | x_t)
            log_p_theta = logits.gather(-1, target.unsqueeze(-1)).squeeze(-1)

            # Compute ELBO weight: 1 / (1 - α) where α = exp(-σ)
            # This upweights loss at low noise levels (α ≈ 1) where prediction is harder
            # and downweights at high noise (α ≈ 0) where most features are masked
            alpha = torch.exp(-sigma[:, i])
            weight = 1.0 / torch.clamp(1 - alpha, min=1e-5)
            losses.append(-weight * log_p_theta)

        loss = torch.stack(losses, dim=1).sum() / x0.shape[0]

        return loss

    # =============================
    # Sampling utilities
    # =============================
    def _build_sampling_sigma(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Build synchronized noise schedules for sampling.

        Creates Karras polynomial schedule for numerical features and derives
        synchronized categorical schedule via unified time mapping.

        Args:
            device: Device for tensor allocation

        Returns:
            Tuple of:
                - sigma_num: Numerical noise schedule [num_steps+1] from σ_max to 0
                - sigma_cat: Categorical noise schedule [num_steps+1, num_categorical] or None
        """
        # Step 1: Build numerical noise schedule using Karras polynomial discretization
        # Returns [σ_max, σ_(T-1), ..., σ_1, 0] with T=num_steps+1 entries
        sigma_num = self._build_sampling_sigma_num(device)

        # Step 2: Derive synchronized categorical noise schedule via unified time mapping
        # Returns synchronized σ_cat or None if no categorical features
        sigma_cat = self._build_synced_sigma_cat(sigma_num)

        return sigma_num, sigma_cat

    def _build_sampling_sigma_num(self, device: torch.device) -> torch.Tensor:
        """Build Karras sigma schedule for sampling.

        Uses polynomial time discretization in transformed space:
        σ(t) = (σ_min^(1/ρ) + t * (σ_max^(1/ρ) - σ_min^(1/ρ)))^ρ

        This allocates more steps to low noise levels where fine details matter,
        improving sample quality compared to linear scheduling.

        Returns:
            Noise levels [num_steps+1] descending from σ_max to 0
        """
        if self.num_steps < 1:
            raise ValueError("num_steps must be a positive integer")

        # Linear ramp in time [1, 0] for reverse denoising
        ramp = torch.linspace(1, 0, self.num_steps, device=device)

        # Compute numerical noise levels via Karras schedule
        sigma_num = self.num_schedule.total_noise(ramp).reshape(*ramp.shape)

        # Append σ = 0 as final target (clean data)
        sigma_num = torch.cat([sigma_num, torch.zeros(1, device=device)])

        return sigma_num

    def _build_sampling_sigma_churn(
        self, sigma_num_cur: torch.Tensor, sigma_cat_cur: torch.Tensor, total_reverse_steps: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply stochastic churn to increase noise level temporarily.

        Computes perturbed noise levels σ̂ = σ * (1 + γ) where γ is controlled by
        S_churn parameter. This improves sample exploration and quality.

        Args:
            sigma_num_cur: Current numerical noise level (scalar)
            sigma_cat_cur: Current categorical noise level [num_categorical]
            total_reverse_steps: Total number of reverse diffusion steps

        Returns:
            Tuple of:
                - sigma_num_hat: Perturbed numerical noise level (scalar)
                - sigma_cat_hat: Perturbed categorical noise level [num_categorical]
        """
        sigma_num_hat = sigma_num_cur
        sigma_cat_hat = sigma_cat_cur

        # Increases noise temporarily to improve exploration and sample quality
        # γ = 0 for deterministic sampling, γ > 0 adds controlled randomness
        gamma = 0.0
        if (
            self.sampler_params["sigma_churn"] > 0
            and self.sampler_params["sigma_churn_min"] <= sigma_num_cur <= self.sampler_params["sigma_churn_max"]
        ):
            # Cap gamma at √2-1 ≈ 0.414 to prevent excessive noise amplification
            # Scale by 1/steps for consistent behavior across different step counts
            gamma = min(self.sampler_params["sigma_churn"] / total_reverse_steps, torch.sqrt(torch.tensor(2.0)) - 1.0)

            # If γ=0, σ̂ = σ_cur (no perturbation)
            sigma_num_hat = sigma_num_cur * (1.0 + gamma)

            # Map perturbed numerical noise to unified time and derive categorical noise
            sigma_cat_hat = self._build_synced_sigma_cat(sigma_num_hat)

        return sigma_num_hat, sigma_cat_hat

    def _init_samples(
        self,
        num_samples: int,
        sigma_num: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Initialize samples from maximum noise prior distribution.

        Starting point for reverse diffusion sampling:
        - Numerical: Pure Gaussian noise scaled by σ_max
        - Categorical: All features set to mask tokens (completely unknown)

        Args:
            num_samples: Number of samples to generate
            sigmas: Noise schedule [num_steps+1], sigmas[0] = σ_max
            device: Device for tensor allocation

        Returns:
            Tuple of:
                - x_num: Initial numerical samples [num_samples, num_numerical] or None
                - x_cat: Initial categorical samples [num_samples, num_categorical] or None
        """
        # Numerical: sample from N(0, σ_max²I)
        x_num = None
        if self.num_numerical > 0:
            x_num = torch.randn(num_samples, self.num_numerical, device=device) * sigma_num[0]

        # Categorical: all features start as masked (maximum corruption)
        x_cat = None
        if self.num_categorical > 0:
            x_cat = self.mask_index.unsqueeze(0).repeat(num_samples, 1)

        return x_num, x_cat

    def _sample_step(
        self,
        x_num_cur: torch.Tensor | None,
        x_cat_ordinal_cur: torch.Tensor | None,
        sigma_num_cur: torch.Tensor,
        sigma_num_next: torch.Tensor,
        sigma_num_hat: torch.Tensor,
        sigma_cat_cur: torch.Tensor | None,
        sigma_cat_next: torch.Tensor | None,
        sigma_cat_hat: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Perform one step of EDM-inspired mixed numerical/categorical update.

        Implements the core reverse diffusion update combining:
        1. Stochastic churn: optional noise injection (σ_cur → σ_hat)
        2. Numerical update: Euler ODE integration with optional Heun correction
        3. Categorical update: stochastic unmasking via predicted distributions

        Algorithm Flow:
            A. Churn injection (if γ > 0):
               - Add Gaussian noise to numerical: x̂_num = x_cur + √(σ̂²-σ_cur²) * ε
               - Stochastically mask categorical with probability 1-exp(σ_cur-σ̂)

            B. Denoise at perturbed state:
               - Query network: D_θ(x̂, σ̂) → (x_num_denoised, logits_cat)

            C. Numerical update (Euler step):
               - Compute derivative: d = (x̂ - D(x̂,σ̂)) / σ̂
               - Step forward: x_next = x̂ + (σ_next - σ̂) * d

            D. Categorical update (MDLM):
               - Apply substitution parameterization to logits
               - Sample from predicted categorical distributions

            E. Second-order correction (optional, if i>0):
               - Re-evaluate derivative at x_next: d' = (x_next - D(x_next,σ_next)) / σ_next
               - Refine: x_next = x̂ + (σ_next - σ̂) * (d + d')/2

        Args:
            x_num_cur: Current numerical state [batch, num_numerical] or None
            x_cat_cur: Current categorical state [batch, num_categorical] or None
            i: Step index (used for second-order correction check)
            sigma_num_cur: Current numerical noise level (scalar tensor)
            sigma_num_next: Target numerical noise level (scalar tensor)
            sigma_num_hat: Perturbed numerical noise level after churn (scalar tensor)
            sigma_cat_cur: Current categorical noise level [num_categorical] or None
            sigma_cat_next: Target categorical noise level [num_categorical] or None
            sigma_cat_hat: Perturbed categorical noise level [num_categorical] or None

        Returns:
            Tuple of:
                - x_num_next: Updated numerical state [batch, num_numerical] or None
                - x_cat_next: Updated categorical state [batch, num_categorical] or None
                - q_xs: Categorical probability distributions [batch, sum(num_classes_w_mask)]
                    Empty tensor if no categorical features
        """
        # ============================================================
        # Step A: Churn injection (stochastic noise perturbation)
        # ============================================================
        x_num_hat, x_cat_hat_ordinal, x_cat_hat_onehot = self._churn_injection(
            x_num_cur,
            x_cat_ordinal_cur,
            sigma_num_cur,
            sigma_num_hat,
            sigma_cat_cur,
            sigma_cat_hat,
        )

        # ============================================================
        # Step B: Denoise at perturbed state σ̂
        # ============================================================
        x_num_denoised, x_cat_denoised = self._denoise(x_num_hat, x_cat_hat_onehot, sigma_num_hat)

        # ============================================================
        # Step C: Numerical update via Euler ODE integration
        # ============================================================
        x_num_next, d_cur = self._euler_step(x_num_hat, x_num_denoised, sigma_num_next, sigma_num_hat)

        # ============================================================
        # Step D: Categorical update via MDLM posterior sampling
        # ============================================================
        x_cat_ordinal_next, x_cat_onehot_next = self._mdlm_step(
            x_cat_hat_ordinal, x_cat_denoised, sigma_cat_hat, sigma_cat_next
        )

        # ============================================================
        # Step E: Second-order correction (Heun's method)
        # ============================================================
        if self.enable_heun_correction:
            # Refine numerical update with second-order correction
            # Improves accuracy of ODE integration step
            x_num_next = self._correction_step(
                x_num_hat, x_num_next, x_cat_onehot_next, sigma_num_next, sigma_num_hat, d_cur
            )

        return x_num_next, x_cat_ordinal_next

    def _churn_injection(self, x_num_cur, x_cat_cur, sigma_num_cur, sigma_num_hat, sigma_cat_cur, sigma_cat_hat):
        """Inject stochastic noise (churn) into current samples.

        Applies temporary noise injection to both numerical and categorical features
        to improve sample quality and exploration.

        Args:
            x_num_cur: Current numerical state [batch, num_numerical] or None
            x_cat_cur: Current categorical state [batch, num_categorical] or None
            sigma_num_cur: Current numerical noise level (scalar)
            sigma_num_hat: Perturbed numerical noise level (scalar)
            sigma_cat_cur: Current categorical noise level [num_categorical] or None
            sigma_cat_hat: Perturbed categorical noise level [num_categorical] or None

        Returns:
            Tuple of:
                - x_num_hat: Perturbed numerical state [batch, num_numerical] or None
                - x_cat_hat_ordinal: Perturbed categorical ordinal state [batch, num_categorical] or None
                - x_cat_hat_onehot: Perturbed categorical one-hot state [batch, sum(num_classes_w_mask)] or None
        """
        # A1: Numerical churn - add Gaussian noise scaled by √(σ̂²-σ_cur²)
        # This implements x̂ = x + √(σ̂²-σ²) * S_noise * ε where ε ~ N(0,I)
        x_num_hat = x_num_cur
        if x_num_hat is not None:
            # Compute noise injection amount: √(σ̂²-σ_cur²)
            noise_scale = torch.sqrt(torch.clamp(sigma_num_hat**2 - sigma_num_cur**2, min=0.0))
            if torch.any(noise_scale > 0):
                # Add scaled Gaussian noise to current state
                x_num_hat = x_num_hat + noise_scale * torch.randn_like(x_num_hat)

        # A2: Categorical churn - stochastic masking with probability 1-exp(σ_cur-σ̂)
        # Probability of transitioning to mask = 1 - α(σ_hat)/α(σ_cur) = 1 - exp(-(σ̂ - σ_cur))
        x_cat_hat_ordinal = x_cat_cur
        x_cat_hat_onehot = x_cat_cur
        if x_cat_hat_ordinal is not None:
            x_cat_hat_ordinal, x_cat_hat_onehot = self._apply_categorical_noise(
                x_cat_cur, sigma_cat_hat - sigma_cat_cur
            )

        return x_num_hat, x_cat_hat_ordinal, x_cat_hat_onehot

    def _euler_step(
        self,
        x_num_hat: torch.Tensor | None,
        x_num_denoised: torch.Tensor | None,
        sigma_num_next: torch.Tensor,
        sigma_num_hat: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Perform Euler ODE integration step for numerical features.

        Implements first-order numerical integration of probability flow ODE:
        dx/dσ = (x - D(x,σ))/σ

        Args:
            x_num_hat: Perturbed numerical state [batch, num_numerical] or None
            x_num_denoised: Denoised prediction [batch, num_numerical] or None
            sigma_num_next: Target noise level (scalar)
            sigma_num_hat: Current noise level (scalar)

        Returns:
            Tuple of:
                - x_num_next: Updated numerical state [batch, num_numerical] or None
                - d_cur: Current derivative (for second-order correction) or None
        """
        # Numerical follows probability flow ODE: dx/dσ = (x - D(x,σ))/σ
        x_num_next = x_num_hat
        d_cur = None
        if x_num_hat is not None:
            # Compute derivative (gradient of log density)
            sigma_hat_safe = torch.clamp(sigma_num_hat, min=1e-8)
            d_cur = (x_num_hat - x_num_denoised) / sigma_hat_safe

            # Euler integration step: x_next = x̂ + Δσ * dx/dσ
            x_num_next = x_num_hat + (sigma_num_next - sigma_num_hat) * d_cur

        return x_num_next, d_cur

    def _mdlm_step(
        self,
        x_cat_hat_ordinal: torch.Tensor | None,
        x_cat_denoised: torch.Tensor | None,
        sigma_cat_hat: torch.Tensor | None,
        sigma_cat_next: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Perform MDLM (Masked Discrete Language Model) update for categorical features.

        Samples from posterior distribution p(x_t-1 | x_t, x_0) for discrete diffusion,
        implementing stochastic unmasking based on predicted class probabilities.

        Args:
            x_cat_hat_ordinal: Perturbed categorical state [batch, num_categorical] or None
            x_cat_denoised: Denoised logits [batch, sum(num_classes_w_mask)] or None
            sigma_cat_hat: Current categorical noise level [num_categorical] or None
            sigma_cat_next: Target categorical noise level [num_categorical] or None

        Returns:
            Tuple of:
                - x_cat_ordinal_next: Updated categorical ordinal state [batch, num_categorical] or None
                - x_cat_onehot_next: Updated categorical one-hot state [batch, sum(num_classes_w_mask)] or None
        """
        x_cat_ordinal_next = x_cat_hat_ordinal
        x_cat_onehot_next = x_cat_hat_ordinal

        if x_cat_ordinal_next is not None:
            # Compute masking probabilities at current and next noise levels
            # move_chance = 1 - exp(-σ) is the probability of being masked
            move_chance_hat = -torch.expm1(-sigma_cat_hat)
            move_chance_next = -torch.expm1(-sigma_cat_next)

            # unmask_amount represents how much probability mass should transition
            # from masked to unmasked state in this step
            unmask_amount = (move_chance_hat - move_chance_next).reshape(-1)

            # Apply substitution parameterization to get proper probability distribution
            logits = self._adjust_log_probabilities(x_cat_denoised, x_cat_hat_ordinal)

            # Process each categorical feature independently
            logits_split = torch.split(logits, self.num_classes_w_mask, dim=-1)
            sampled_class_indices = []
            for i, logits_i in enumerate(logits_split):
                # Convert log probabilities to probabilities
                p_x0_i = logits_i.exp()

                # Build posterior distribution q(x_s | x_t, x_0)
                # When unmasking (unmask_amount > 0), allocate probability to predicted classes
                q_xs_i = p_x0_i
                if unmask_amount[i] > 1e-8:
                    q_xs_i = p_x0_i * unmask_amount[i]

                # Allocate remaining probability mass to staying masked
                mask_idx = self.mask_index[i].item()
                q_xs_i[:, mask_idx] = move_chance_next[i]

                # Normalize to form valid probability distribution
                q_xs_i = q_xs_i / (q_xs_i.sum(dim=-1, keepdim=True) + 1e-10)

                # Sample from posterior distribution
                sampled_idx = Categorical(q_xs_i).sample()

                # Handle edge case: if no unmasking occurs (final step or no change)
                # validate sampled indices against training cardinality
                if unmask_amount[i] < 1e-8:
                    train_cardinality_per_feature = self.train_cardinality_list[i]
                    max_valid = int(train_cardinality_per_feature) - 1
                    invalid_mask = (sampled_idx < 0) | (sampled_idx > max_valid)
                    # Mark invalid samples as -1 (will be clamped later)
                    sampled_idx = torch.where(invalid_mask, torch.full_like(sampled_idx, -1), sampled_idx)

                sampled_class_indices.append(sampled_idx)

            x_cat_ordinal_next = torch.stack(sampled_class_indices, dim=1)
            x_cat_onehot_next = self._convert_to_one_hot_with_mask(x_cat_ordinal_next)

        return x_cat_ordinal_next, x_cat_onehot_next

    def _correction_step(self, x_num_hat, x_num_next, x_cat_onehot_next, sigma_num_next, sigma_num_hat, d_cur):
        """Apply second-order Heun correction for improved accuracy.

        Re-evaluates the derivative at the predicted state and uses the average
        of first and second derivatives for better ODE integration accuracy.

        Args:
            x_num_hat: Perturbed numerical state [batch, num_numerical] or None
            x_num_next: Predicted next state from Euler step [batch, num_numerical] or None
            x_cat_onehot_next: Updated categorical state [batch, sum(num_classes_w_mask)] or None
            sigma_num_next: Target noise level (scalar)
            sigma_num_hat: Current noise level (scalar)
            d_cur: Current derivative from Euler step or None

        Returns:
            Corrected numerical state [batch, num_numerical] or None
        """
        if x_num_next is not None and d_cur is not None:
            # Re-evaluate network at predicted state x_next
            denoised_prime, _ = self._denoise(x_num_next, x_cat_onehot_next, sigma_num_next)

            # Compute derivative at predicted point
            sigma_next_safe = torch.clamp(sigma_num_next, min=1e-8)
            d_prime = (x_num_next - denoised_prime) / sigma_next_safe

            # Corrector step: use average of both derivatives for higher accuracy
            # This gives O(Δσ³) truncation error vs O(Δσ²) for Euler
            x_num_next = x_num_hat + 0.5 * (d_cur + d_prime) * (sigma_num_next - sigma_num_hat)

        return x_num_next
