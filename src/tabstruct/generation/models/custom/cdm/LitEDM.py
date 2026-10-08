"""Elucidated Diffusion Model (EDM) for tabular generation.

This module implements a simplified variant of the Elucidated Diffusion Model
introduced in *Elucidating the Design Space of Diffusion-Based Generative Models*
by Karras et al. (2022). Compared with VPSDE/VE-SDE formulations, EDM focuses on
designing noise schedules, preconditioning, and sampling heuristics that lead to
stable and data-efficient training.

Key characteristics of the implementation below:
        - Training samples log-normal noise levels and applies EDM loss weighting.
        - A UniModMLP backbone is wrapped with the provided EDM preconditioning
          utilities (see ``modules.DenoiseModel``) for stable optimisation.
        - Sampling follows the recommended Karras schedule with a lightweight
          Heun (second-order) integrator for clarity.

The implementation mirrors the public Lightning generators in this repository
and keeps the API close to :class:`LitVESDE`, while emphasising the unique
preconditioning and noise scheduling aspects of EDM.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tabeval.plugins.core.models.tabular_encoder import TabularEncoder

from ...BaseGenerator import BaseLightningGenerationModule, BaseLitJointGenerator
from ..ddpm.utils.modules import UniModMLP
from .utils.modules import DenoiseModel


class LitEDM(BaseLitJointGenerator):
    """Elucidated Diffusion Model (EDM) generator for tabular data."""

    def __init__(self, args):
        super().__init__(args)

        if args.task not in ["classification", "regression", "unsupervision"]:
            raise ValueError(f"Task {args.task} is not supported for {self.name} model")

        self.model = _LitEDM(args)

    # ================================================================
    # =                     Model-specific                           =
    # ================================================================
    def _prepare_data_scalers(self):
        """Configure data scalers for EDM preprocessing.

        EDM works in continuous space, so we use quantile transformation to:
        1. Map categorical features to continuous representations
        2. Normalize continuous features to a stable range
        This aligns with EDM's assumption of data lying on a continuous manifold.
        """
        # Feature scaler: quantile transform maps data to uniform [0,1] or Gaussian distribution
        # This provides a stable range for diffusion and helps with numerical stability
        self.feature_scaler = TabularEncoder(
            categorical_encoder="quantile",
            cat_encoder_params={},
            continuous_encoder="quantile",
            cont_encoder_params={},
        )

        # Target scaler: same approach for supervised tasks
        # Ensures targets participate in diffusion without scale mismatch
        self.target_scaler = None
        if self.args.task != "unsupervision":
            self.target_scaler = TabularEncoder(
                continuous_encoder="quantile",
                cont_encoder_params={},
            )

    # ================================================================
    # =                      Hyperparams                             =
    # ================================================================
    @classmethod
    def _define_default_params(cls):
        """Default hyperparameters aligned with EDM best practices.

        These values are based on Karras et al. 2022 recommendations and adapted
        for tabular data. Key parameters:
        - sigma_min/max: Define the noise level range [0.002, 80.0]
        - rho: Controls the time-step distribution (higher = more steps at low noise)
        - sigma_data: Expected std of training data (used in preconditioning)
        - p_mean/p_std: Log-normal noise distribution parameters for training

        Tuning Guide:
        - If samples are too noisy: decrease sigma_max or increase num_steps
        - If training is unstable: adjust sigma_data to match actual data scale
        - If sampling is slow: decrease num_steps (but quality suffers)
        - If training focuses on wrong noise levels: adjust p_mean/p_std
        """

        params_arch = {
            # === Noise Schedule Parameters ===
            "sigma_min": 0.002,  # Minimum noise level (near-clean data)
            "sigma_max": 80.0,  # Maximum noise level (high corruption)
            "rho": 7.0,  # Time discretization exponent (7 = focus on low noise)
            "num_steps": 40,  # Number of sampling steps (higher = better quality)
            # === Preconditioning Parameters ===
            "sigma_data": 1.0,  # Expected data std (match your normalized data)
            # === Training Noise Distribution ===
            "p_mean": 0.0,  # Mean of log(σ) ~ N(0, 1) → σ ~ LogNormal(0, 1)
            "p_std": 1.0,  # Std of log(σ) - controls noise level diversity
        }

        params_optim = {
            "lr": 1e-3,  # Learning rate (EDM is fairly robust to this)
            "weight_decay": 1e-5,  # Regularization (light, as EDM already well-conditioned)
        }

        return {"architecture": params_arch, "optimization": params_optim}

    @classmethod
    def _define_single_run_params(cls):
        """Lighter configuration for quick development runs."""

        params_arch = {
            "sigma_min": 0.005,
            "sigma_max": 50.0,
            "rho": 5.0,
            "num_steps": 24,
            "sigma_data": 1.0,
            "p_mean": 0.0,
            "p_std": 1.0,
        }

        params_optim = {
            "lr": 1e-3,
            "weight_decay": 1e-5,
        }

        return {"architecture": params_arch, "optimization": params_optim}

    @classmethod
    def _define_test_params(cls):
        """Minimal configuration used in CI / unit tests."""

        params_arch = {
            "sigma_min": 0.01,
            "sigma_max": 10.0,
            "rho": 4.0,
            "num_steps": 12,
            "sigma_data": 1.0,
            "p_mean": 0.0,
            "p_std": 1.0,
        }

        params_optim = {
            "lr": 1e-3,
            "weight_decay": 1e-5,
        }

        return {"architecture": params_arch, "optimization": params_optim}

    @classmethod
    def _get_model_specific_scaler_config(cls):
        """Encode categorical features numerically to align with EDM's continuous space.

        EDM applies continuous diffusion over the entire feature vector, so we need
        to ensure categorical features are treated as numerical values in the model.
        This configuration:
        1. One-hot encodes categorical features (creates binary indicator vectors)
        2. Treats the resulting vectors as continuous values during diffusion
        3. Enables unified denoising across all feature types
        """

        return {
            "context": {
                "disable_preprocessing": False,  # Keep preprocessing pipeline active
            },
            "feature_scaler": {
                "categorical_transform": "onehot",  # Map categories to binary vectors
                "categorical_as_numerical": True,  # Treat as continuous in diffusion
            },
            "target_scaler": {},
        }


class _LitEDM(BaseLightningGenerationModule):
    """Lightning wrapper exposing the EDM training & sampling loops.

    This internal class bridges the TabStruct framework with the core EDM
    diffusion logic. It handles:
    - Model instantiation with hyperparameters from args
    - Loss computation during training (EDM weighted denoising)
    - Sample generation via the trained denoiser
    """

    def __init__(self, args):
        super().__init__(args)

    # ================================================================
    # =                     Model-specific                           =
    # ================================================================
    def _create_torch_model(self):
        """Create the EDM model with preconditioned denoising network.

        Architecture Stack (bottom to top):
        1. UniModMLP: Time-conditioned transformer-MLP hybrid for denoising
        2. DenoiseModel: Wrapper that applies EDM preconditioning
        3. EDM: Core diffusion model with noise scheduling and sampling

        The architecture processes data as:
        x_noisy → [EDMModel preconditioning] → [UniModMLP backbone] → x_denoised
        """
        # Calculate total dimension: features + target (for supervised tasks)
        # In joint generation, we diffuse features and targets together
        input_dim = len(self.args.full_feature_list_model) + len(self.args.full_target_list_model)

        # Layer 1: Build the core denoising network (UniModMLP)
        # This is a transformer-based architecture that predicts clean data from noisy input
        score_net = UniModMLP(
            d_numerical=input_dim,  # Input dimension (all features treated as numerical)
            categories=[],  # No categorical handling (already one-hot encoded)
            num_layers=2,  # Number of transformer encoder/decoder layers
        )

        # Layer 2: Wrap with EDM preconditioning for stable training
        # This applies noise-dependent input/output scaling (c_skip, c_out, c_in, c_noise)
        # Preconditioning is critical for EDM's performance - see Karras et al. 2022
        denoise_model = DenoiseModel(
            denoise_fn=score_net,  # The underlying denoising network
            sigma_data=self.args.model_params["architecture"]["sigma_data"],  # Expected data std
            precond=True,  # Enable EDM scaling
            net_conditioning="sigma",  # Use log(σ)/4 for noise conditioning (EDM standard)
        )

        # Layer 3: Instantiate the core EDM diffusion model
        # This handles noise scheduling, training, and sampling
        edm = EDM(
            num_features=input_dim,
            sigma_min=self.args.model_params["architecture"]["sigma_min"],  # Min noise level
            sigma_max=self.args.model_params["architecture"]["sigma_max"],  # Max noise level
            rho=self.args.model_params["architecture"]["rho"],  # Karras schedule exponent
            num_steps=self.args.model_params["architecture"]["num_steps"],  # Sampling steps
            sigma_data=self.args.model_params["architecture"]["sigma_data"],  # Data scale
            p_mean=self.args.model_params["architecture"]["p_mean"],  # Log-normal mean
            p_std=self.args.model_params["architecture"]["p_std"],  # Log-normal std
            denoise_model=denoise_model,  # Preconditioned denoising network
        )

        return edm

    def _compute_loss(self, data_real: torch.Tensor, forward_dict: dict):
        """Compute EDM denoising loss with adaptive weighting.

        EDM uses a noise-level dependent loss weighting scheme:
        L(θ) = E[λ(σ) ||D_θ(x + σε; σ) - x||²]

        where λ(σ) = (σ² + σ_data²) / (σ · σ_data)²

        This weighting balances contributions across different noise levels and
        improves training stability compared to unweighted MSE.

        Why weighted loss?
        - Prevents over-fitting to easy (high-noise) samples
        - Ensures all noise levels contribute meaningfully to gradients
        - Reduces variance in gradient estimates
        - Critical for EDM's superior performance

        Args:
            data_real: Ground truth clean data [batch_size, num_features]
            forward_dict: Dictionary from forward pass containing:
                - x_denoised: Network's prediction [batch_size, num_features]
                - loss_weight: Adaptive weight λ(σ) [batch_size, 1]

        Returns:
            Dictionary with:
                - total_loss: Weighted MSE loss (scalar)
                - reconstruction_loss: Same value for logging
        """
        # Extract predictions and weights from forward pass
        data_syn = forward_dict["x_denoised"]  # Network prediction D_θ(x_noisy; σ)
        weight = forward_dict["loss_weight"]  # EDM loss weight λ(σ)

        # Compute element-wise mean squared error
        # Shape: [batch_size, num_features]
        mse = F.mse_loss(data_syn, data_real, reduction="none")

        # Apply adaptive weighting and reduce to scalar
        # weight broadcasts to [batch_size, num_features] automatically
        # Sum over features, then average over batch
        loss = (weight * mse).sum() / data_real.shape[0]

        return {
            "total_loss": loss,  # Required by Lightning training loop
            "reconstruction_loss": loss,  # Same value, logged separately for monitoring
        }

    def _generate(self, num_samples: int):
        return self.torch_model.sample(num_samples, device=self.device)


class EDM(nn.Module):
    """Core Elucidated Diffusion Model (EDM) for continuous diffusion.

    This implements the EDM framework from Karras et al. (2022), which emphasizes:
    1. Log-normal noise level sampling during training (better coverage)
    2. Preconditioning of the denoising network (stable optimization)
    3. Adaptive loss weighting based on noise level (balanced learning)
    4. Karras sigma schedule for sampling (polynomial time discretization)
    5. Heun's method (2nd order ODE solver) for high-quality generation

    Key differences from VE-SDE/VP-SDE:
    - No score matching: directly predicts denoised data D(x_noisy, σ)
    - Preconditioning wrapper handles input/output scaling automatically
    - Log-normal noise distribution (vs uniform time sampling)
    - More sophisticated sampling with 2nd-order integrator

    EDM Philosophy:
    Instead of focusing on the theoretical SDE formulation, EDM asks:
    "What are the practical design choices that make diffusion models work well?"
    Answer: preconditioning, noise schedule, loss weighting, and sampling method.

    Training Loop:
    1. Sample clean data x_0 from dataset
    2. Sample noise level σ ~ LogNormal(p_mean, p_std²)
    3. Add noise: x_noisy = x_0 + σ * ε
    4. Predict: x_pred = D_θ(x_noisy, σ)
    5. Loss: λ(σ) * ||x_pred - x_0||²

    Sampling Loop:
    1. Start from pure noise: x_T ~ N(0, σ_max² I)
    2. Build schedule: [σ_max, ..., σ_1, 0]
    3. For each step, use Heun's method to solve ODE: dx/dσ = (x - D(x,σ))/σ
    4. Return x_0 (generated sample)

    Args:
        num_features: Total dimension of data vector (features + target)
        sigma_min: Minimum noise level (near-clean data, typically 0.002)
        sigma_max: Maximum noise level (high corruption, typically 80.0)
        rho: Time discretization exponent for Karras schedule (typically 7)
        num_steps: Number of sampling steps (quality vs speed tradeoff, typically 40)
        sigma_data: Expected std of training data (for preconditioning, typically 0.5-1.0)
        p_mean: Mean of log-normal noise distribution (typically 0.0)
        p_std: Std of log-normal noise distribution (typically 1.0)
        denoise_model: Preconditioned denoising network (EDMModel wrapping UniModMLP)
    """

    def __init__(
        self,
        num_features: int,
        sigma_min: float,
        sigma_max: float,
        rho: float,
        num_steps: int,
        sigma_data: float,
        p_mean: float,
        p_std: float,
        denoise_model: DenoiseModel,
    ):
        super().__init__()

        # Data and model dimensions
        self.num_features = int(num_features)  # Total features (X + y for joint generation)

        # Noise level range [sigma_min, sigma_max]
        # This defines the "diffusion path" from clean to noisy data
        self.sigma_min = float(sigma_min)  # Near-zero noise (clean data regime)
        self.sigma_max = float(sigma_max)  # High noise (structure formation regime)

        # Sampling parameters (Karras schedule)
        self.rho = float(rho)  # Time discretization exponent (higher = focus on low noise)
        self.num_steps = int(num_steps)  # Number of denoising steps (quality vs speed)

        # Preconditioning and weighting parameters
        # sigma_data is the most important: should match actual std of your data
        self.sigma_data = float(sigma_data)  # Expected data std (typically 0.5-1.0 for normalized data)

        # Training noise distribution (log-normal)
        # Controls which noise levels are sampled during training
        self.p_mean = float(p_mean)  # Mean of log σ (0 = geometric mean of sigma_min/max)
        self.p_std = float(p_std)  # Std of log σ (1 = wide coverage of noise scales)

        # The preconditioned denoising network (wrapped UniModMLP)
        # This is where the actual neural network computation happens
        self.denoise_model = denoise_model

    # =============================
    # Forward diffusion / training
    # =============================
    def forward(self, x0: torch.Tensor) -> dict:
        """Training forward pass: add noise and predict denoised output.

        EDM Training Procedure:
        1. Sample noise level σ from log-normal distribution
        2. Add Gaussian noise: x_noisy = x_0 + σ * ε, where ε ~ N(0, I)
        3. Predict denoised data: D_θ(x_noisy, σ) ≈ x_0
        4. Compute weighted loss: λ(σ) ||D_θ(x_noisy, σ) - x_0||²

        This differs from score-based models which predict the score (gradient).
        EDM directly predicts the clean data, which is simpler and more stable.

        Args:
            x0: Clean data samples [batch_size, num_features]

        Returns:
            Dictionary containing:
                - x_noisy: Noise-corrupted input
                - x_denoised: Network prediction of clean data
                - x_target: Original clean data (target)
                - sigma: Noise level used
                - loss_weight: EDM adaptive loss weight
        """
        batch_size = x0.shape[0]
        device = x0.device

        # Step 1: Sample noise levels from log-normal distribution
        # Log-normal provides better coverage of noise scales compared to uniform sampling
        # Shape: [batch_size] -> [batch_size, 1] for broadcasting with features
        sigma = self._sample_training_sigma(batch_size, device).view(batch_size, 1)

        # Step 2: Add Gaussian noise to clean data
        # Forward diffusion: x_noisy = x_0 + σ * ε, where ε ~ N(0, I)
        # This corrupts the data proportionally to the noise level σ
        noise = torch.randn_like(x0)  # Sample standard Gaussian noise
        x_noisy = x0 + sigma * noise  # Scale and add to clean data

        # Step 3: Predict denoised data using the preconditioned network
        # The network D_θ predicts the clean data x_0 from noisy observation x_noisy
        # The preconditioning wrapper (EDMModel) handles:
        #   - Input scaling: c_in * x_noisy
        #   - Noise conditioning: log(σ)/4 passed to the network
        #   - Output scaling: c_skip * x_noisy + c_out * network_output
        x_denoised = self._denoise(x_noisy, sigma)

        # Step 4: Compute EDM loss weight λ(σ) = (σ² + σ_data²) / (σ · σ_data)²
        # This adaptive weighting:
        #   - Balances contributions across different noise levels
        #   - Prevents over-emphasis on easy (high-noise) samples
        #   - Improves training stability and convergence
        loss_weight = self._loss_weight(sigma).view(batch_size, 1)

        return {
            "x_denoised": x_denoised,  # Network's prediction of clean data
            "loss_weight": loss_weight,  # Adaptive weight for loss balancing
        }

    # =============================
    # Sampling (Karras schedule)
    # =============================
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Generate samples using Heun's method with Karras sigma schedule.

        EDM Sampling Algorithm:
        1. Initialize from high-noise Gaussian: x_T ~ N(0, σ_max² I)
        2. Build Karras schedule: polynomial time discretization from σ_max to 0
        3. For each step, use Heun's method (2nd-order ODE solver):
           a. Predict derivative at current point: d_cur = (x - D(x, σ)) / σ
           b. Take Euler step to get predictor: x_pred = x + Δσ * d_cur
           c. Predict derivative at predictor: d_next = (x_pred - D(x_pred, σ_next)) / σ_next
           d. Corrector step using average: x_next = x + Δσ * (d_cur + d_next) / 2

        Heun's method provides better accuracy than simple Euler integration,
        allowing fewer steps while maintaining quality.

        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on

        Returns:
            Generated samples [num_samples, num_features]
        """
        # Step 1: Build Karras sigma schedule: polynomial time discretization
        # The schedule allocates more steps to low noise levels where fine details matter
        # Returns: [σ_max, σ_(T-1), ..., σ_1, 0] - decreasing noise levels
        sigmas = self._build_sigma_schedule(device)

        # Step 2: Initialize from high-noise Gaussian N(0, σ_max² I)
        # Start with pure noise at the maximum noise level
        # Each feature is independently sampled from N(0, σ_max²)
        x = torch.randn(num_samples, self.num_features, device=device) * sigmas[0]

        # Step 3: Iteratively denoise from σ_max down to σ = 0 (clean data)
        # Use Heun's method (2nd-order ODE solver) for each denoising step
        for i in range(len(sigmas) - 1):
            sigma_curr = sigmas[i]  # Current noise level
            sigma_next = sigmas[i + 1]  # Target noise level (lower)

            # Perform one Heun integration step
            # This moves x from noise level σ_curr to σ_next
            x = self._sample_heun_step(x, sigma_curr, sigma_next, num_samples, device)

        return x

    # =============================
    # Utilities
    # =============================
    def _denoise(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Apply the preconditioned denoising network.

        This method wraps the call to the EDM-preconditioned denoising network.
        The network predicts the clean data x_0 from noisy observation x and noise level σ.

        Args:
            x: Noisy input [batch_size, num_features]
            sigma: Noise level [batch_size, 1] or [batch_size]

        Returns:
            Denoised prediction [batch_size, num_features]
        """
        # Ensure sigma has shape [batch_size, 1] for proper broadcasting
        sigma_expanded = sigma.view(-1, 1)

        # Call the preconditioned denoising network
        # Arguments:
        #   - x_num: numerical features (all features in our case)
        #   - x_cat: categorical features (None - we use one-hot encoding)
        #   - t: timestep (None - unused when net_conditioning="sigma")
        #   - sigma: noise level (used for preconditioning and conditioning)
        # The EDMModel wrapper applies:
        #   1. Input scaling: c_in * x
        #   2. Noise conditioning: log(sigma)/4 passed to network
        #   3. Output scaling: c_skip * x + c_out * network_output
        x_denoised, _ = self.denoise_model(x_num=x, x_cat=None, t=None, sigma=sigma_expanded)

        return x_denoised

    def _sample_training_sigma(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample noise levels from log-normal distribution.

        EDM uses log-normal distribution for training noise levels:
        σ ~ exp(N(p_mean, p_std²))

        This provides better coverage of the noise scale space compared to
        uniform sampling, as important dynamics often occur at specific scales.

        Why log-normal?
        - Covers multiple orders of magnitude efficiently
        - Natural for multiplicative processes
        - Better gradient signal across all noise scales
        - Karras et al. found it superior to uniform sampling

        Args:
            batch_size: Number of noise levels to sample
            device: Device for tensor allocation

        Returns:
            Noise levels σ [batch_size]
        """
        # Sample from standard normal and transform to log-normal
        # Step 1: Sample z ~ N(0, 1)
        z = torch.randn(batch_size, device=device)

        # Step 2: Scale and shift: log(σ) = p_mean + p_std * z
        # This gives log(σ) ~ N(p_mean, p_std²)
        log_sigma = self.p_mean + z * self.p_std

        # Step 3: Exponentiate to get σ ~ LogNormal(p_mean, p_std²)
        sigma = log_sigma.exp()

        # Step 4: Clamp to valid range [sigma_min, sigma_max]
        # This prevents extreme noise levels that could destabilize training
        sigma = sigma.clamp(min=self.sigma_min, max=self.sigma_max)

        return sigma

    def _loss_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute EDM loss weighting function λ(σ).

        The EDM loss weight is:
        λ(σ) = (σ² + σ_data²) / (σ · σ_data)²

        This weighting scheme:
        1. Balances contributions across different noise levels
        2. Prevents over-emphasis on high-noise samples
        3. Improves training stability and convergence
        4. Reduces variance in gradient estimates

        Intuition:
        - At low noise (σ → 0): weight ≈ 1/σ² (emphasize clean data)
        - At high noise (σ → ∞): weight ≈ 1/σ_data² (bound the emphasis)
        - Transition at σ ≈ σ_data (expected data scale)

        Args:
            sigma: Noise level [batch_size, 1] or [batch_size]

        Returns:
            Loss weight λ(σ) with same shape as sigma
        """
        # Compute squared terms
        sigma2 = sigma**2  # σ²
        sigma_data2 = self.sigma_data**2  # σ_data²

        # Apply EDM weighting formula: (σ² + σ_data²) / (σ · σ_data)²
        # Numerator: total variance (data + noise)
        # Denominator: squared geometric mean of σ and σ_data
        return (sigma2 + sigma_data2) / ((sigma * self.sigma_data) ** 2)

    def _build_sigma_schedule(self, device: torch.device) -> torch.Tensor:
        """Build Karras time discretization schedule.

        The Karras schedule uses polynomial interpolation:
        σ(t) = (σ_max^(1/ρ) + t · (σ_min^(1/ρ) - σ_max^(1/ρ)))^ρ

        where t ∈ [0, 1] is linearly spaced.

        This schedule allocates more steps to low noise levels (where fine
        details are refined) and fewer steps to high noise levels (where
        coarse structure emerges quickly).

        Typical values: ρ = 7 provides good balance
        - Higher ρ: more concentration at low noise (more detail refinement)
        - Lower ρ: more uniform distribution (more structure formation)

        Why this schedule?
        - Empirically found to be more efficient than uniform spacing
        - Reflects that low-noise refinement needs more careful integration
        - High-noise structure formation is less sensitive to step size

        Args:
            device: Device for tensor allocation

        Returns:
            Sigma schedule [num_steps + 1], with final element = 0
            Example (ρ=7, 5 steps): [80.0, 15.2, 2.9, 0.5, 0.1, 0.0]
        """
        # Validation: ensure we have at least one step
        if self.num_steps < 1:
            raise ValueError("num_steps must be a positive integer")

        # Step 1: Create linear ramp in time: t ∈ [0, 1]
        # This gives uniform steps in "time space"
        ramp = torch.linspace(0, 1, self.num_steps, device=device)

        # Step 2: Apply polynomial interpolation in transformed space
        # Transform to ρ-th root space, interpolate linearly, then raise to ρ-th power
        # This creates non-uniform spacing in σ that favors low noise levels

        # Transform boundaries: σ_max^(1/ρ) and σ_min^(1/ρ)
        sigma_max_transformed = self.sigma_max ** (1.0 / self.rho)
        sigma_min_transformed = self.sigma_min ** (1.0 / self.rho)

        # Linear interpolation in transformed space
        sigmas_transformed = sigma_max_transformed + ramp * (sigma_min_transformed - sigma_max_transformed)

        # Transform back to original space: raise to ρ-th power
        sigmas = sigmas_transformed**self.rho

        # Step 3: Append σ = 0 as final step (clean data target)
        # This ensures the sampling ends at perfectly clean data
        sigmas = torch.cat([sigmas, torch.zeros(1, device=device)])

        return sigmas

    def _sample_heun_step(
        self,
        x: torch.Tensor,
        sigma_curr: torch.Tensor,
        sigma_next: torch.Tensor,
        num_samples: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Perform one step of Heun's method for ODE integration.

        Heun's method is a 2nd-order Runge-Kutta method (improved Euler):
        1. Predictor: Take Euler step using derivative at current point
        2. Corrector: Re-evaluate derivative at predicted point
        3. Final step: Average both derivatives for better accuracy

        This is more accurate than simple Euler integration (1st-order)
        and allows using fewer sampling steps while maintaining quality.

        Args:
            x: Current samples [num_samples, num_features]
            sigma_curr: Current noise level (scalar tensor)
            sigma_next: Target noise level (scalar tensor)
            num_samples: Batch size
            device: Device for tensor allocation

        Returns:
            Updated samples at noise level sigma_next
        """
        # Step 1: Denoise at current noise level
        # Predict clean data: D(x, σ_curr) ≈ x_0
        sigma_curr_batch = torch.full((num_samples,), sigma_curr.item(), device=device)
        denoised = self._denoise(x, sigma_curr_batch)

        # Step 2: Check if we've reached clean data (σ = 0)
        # At final step, just return the denoised prediction
        if sigma_next.item() < 1e-8:  # Use small epsilon instead of exact 0.0
            return denoised

        # Step 3: Compute derivative of diffusion ODE at current point
        # The ODE is: dx/dσ = (x - D(x, σ)) / σ
        # This represents the direction to move in to reduce noise
        delta = sigma_next - sigma_curr  # Step size (negative: moving toward lower noise)
        d_cur = (x - denoised) / sigma_curr  # Current derivative

        # Step 4: Heun's method - Predictor step (Euler)
        # Take full Euler step to get predicted position at σ_next
        x_euler = x + delta * d_cur

        # Step 5: Heun's method - Evaluate derivative at predicted point
        # Re-denoise at the predicted position to get better derivative estimate
        sigma_next_batch = torch.full((num_samples,), sigma_next.item(), device=device)
        denoised_next = self._denoise(x_euler, sigma_next_batch)
        d_prime = (x_euler - denoised_next) / sigma_next  # Derivative at predicted point

        # Step 6: Heun's method - Corrector step
        # Use average of both derivatives for more accurate integration
        # This gives 2nd-order accuracy: O(Δσ³) truncation error
        x_next = x + 0.5 * (d_cur + d_prime) * delta

        return x_next
