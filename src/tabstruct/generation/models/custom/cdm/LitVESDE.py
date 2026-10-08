"""Variance Exploding SDE diffusion model for tabular generation.

This module implements the Variance-Exploding SDE (VE-SDE) framework from
"Score-Based Generative Modeling through Stochastic Differential Equations"
(Song et al., 2021). Unlike the VP-SDE variant, VE-SDE exponentially grows the
noise level over time and uses a scaled Gaussian prior at t = 1.

Key Differences from VP-SDE:
    - Forward process: x_t = x_0 + σ(t) · z  (variance increases with t)
    - Noise schedule: Exponential σ(t) = σ_min · (σ_max / σ_min)^t
    - Reverse SDE: Drift depends only on the learned score (no -0.5 β x term)
    - Sample initialization: From N(0, σ_max² I)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tabeval.plugins.core.models.tabular_encoder import TabularEncoder

from ...BaseGenerator import BaseLightningGenerationModule, BaseLitJointGenerator
from ..ddpm.utils.modules import UniModMLP


class LitVESDE(BaseLitJointGenerator):
    """Variance-Exploding SDE (VE-SDE) diffusion model for tabular data.

    Design notes:
    - Treat all features as numerical: a single continuous diffusion over the full vector
    - Forward diffusion: VE-SDE with exponentially increasing noise level σ(t)
    - Training loss: weighted score matching against the analytic Gaussian score
    - Sampling: Predictor-Corrector (reverse-time SDE + Langevin corrector)
    - API mirrors :class:`LitVPSDE` for consistency across diffusion variants
    """

    def __init__(self, args):
        """Initialize the Variance-Exploding SDE diffusion model.

        Args:
            args: Argument namespace containing model configuration and task type.
                  Must specify ``task`` as one of ``{"classification", "regression", "unsupervision"}``.
        """
        super().__init__(args)

        if args.task not in ["classification", "regression", "unsupervision"]:
            raise ValueError(f"Task {args.task} is not supported for {self.name} model")

        # Initialize the Lightning module that handles training/generation
        self.model = _LitVESDE(args)

    # ================================================================
    # =                     Model-specific                           =
    # ================================================================
    def _prepare_data_scalers(self):
        """Configure data scalers for VE-SDE preprocessing.

        VE-SDE sampling begins from a zero-mean Gaussian prior N(0, σ_max² I).
        To align with this, we use passthrough scalers that preserve the data
        distribution without centering or standardization. The model will learn
        to denoise from the natural data manifold.

        Note: Categorical features are ordinally encoded upstream (see
        _get_model_specific_scaler_config) and treated as numerical values
        within the continuous diffusion space.
        """
        # Feature scaler: passthrough preserves original distributions
        # This is intentional—VE-SDE learns the score of the unmodified data manifold
        self.feature_scaler = TabularEncoder(
            categorical_encoder="quantile",
            cat_encoder_params={},
            continuous_encoder="quantile",
            cont_encoder_params={},
        )

        # Target scaler: also passthrough for supervised tasks
        # Ensures target values participate in diffusion without preprocessing bias
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
        """Define default hyperparameters for production training.

        Returns:
            dict: Nested dictionary with ``architecture`` and ``optimization`` parameters.
                  Architecture params control the VE-SDE noise schedule and sampler.
                  Optimization params control the learning rate and regularization.
        """
        params_arch = {
            # VE-SDE noise schedule parameters
            "sigma_min": 0.01,  # Minimum noise level at t=0 (controls small perturbations)
            "sigma_max": 50.0,  # Maximum noise level at t=1 (controls prior spread)
            # Predictor-Corrector sampler configuration
            "pc_num_steps": 1000,  # Number of reverse diffusion steps (higher = better quality)
            "pc_corrector_steps": 1,  # Langevin steps per time step (0-2 typical)
            "pc_snr": 0.16,  # Signal-to-noise ratio for adaptive corrector step size
        }

        params_optim = {
            "lr": 1e-3,  # Learning rate for Adam optimizer
            "weight_decay": 1e-5,  # L2 regularization coefficient
        }

        return {"architecture": params_arch, "optimization": params_optim}

    @classmethod
    def _define_single_run_params(cls):
        """Define lighter parameters for rapid experimentation."""
        params_arch = {
            "sigma_min": 0.01,
            "sigma_max": 20.0,
            "pc_num_steps": 200,
            "pc_corrector_steps": 1,
            "pc_snr": 0.16,
        }

        params_optim = {
            "lr": 1e-3,
            "weight_decay": 1e-5,
        }

        return {"architecture": params_arch, "optimization": params_optim}

    @classmethod
    def _define_test_params(cls):
        """Define minimal parameters for CI/testing runs."""
        params_arch = {
            "sigma_min": 0.01,
            "sigma_max": 10.0,
            "pc_num_steps": 50,
            "pc_corrector_steps": 1,
            "pc_snr": 0.16,
        }
        params_optim = {
            "lr": 1e-3,
            "weight_decay": 1e-5,
        }

        return {"architecture": params_arch, "optimization": params_optim}

    @classmethod
    def _get_model_specific_scaler_config(cls):
        """Configure preprocessing to treat all features as continuous values.

        VE-SDE applies a unified continuous diffusion process over the entire feature
        vector, so categorical features must be encoded as numerical values. This
        configuration ensures that:
        1. Categorical features are ordinally encoded (mapped to integers 0, 1, 2, ...)
        2. These encoded values are treated as numerical within the diffusion process
        3. The score network learns continuous gradients across the full data space

        Returns:
            dict: Scaler configuration with ``categorical_as_numerical=True`` to enable
                  unified treatment of all features in the continuous diffusion space.
        """
        return {
            "context": {
                "disable_preprocessing": False,  # Keep preprocessing pipeline active
            },
            "feature_scaler": {
                "categorical_transform": "onehot",  # Map categories to one-hot vectors
                "categorical_as_numerical": True,  # Treat ordinal codes as continuous values
            },
            "target_scaler": {},
        }


class _LitVESDE(BaseLightningGenerationModule):
    """Lightning module for VE-SDE training and inference.

    This internal class bridges the TabStruct framework with the core VE-SDE
    diffusion logic. It handles:
    - Model instantiation with hyperparameters from the args namespace
    - Loss computation during training (weighted score matching)
    - Sample generation via Predictor-Corrector sampling
    """

    def __init__(self, args):
        """Initialize the Lightning module with configuration from args."""
        super().__init__(args)

    # ================================================================
    # =                     Model-specific                           =
    # ================================================================
    def _create_torch_model(self):
        """Create the VE-SDE model with a time-conditioned score network.

        The score network is a continuous-time MLP that predicts the gradient
        of the log-density (the "score") for any noisy sample and timestep:
        s_θ(x_t, t) ≈ ∇_x log p_t(x_t)

        Architecture:
        - UniModMLP with sinusoidal timestep embeddings (dim_t=1024)
        - Treats all features as numerical (categories pre-encoded as ordinals)
        - Outputs a vector of same dimension as input (one score per feature)

        Returns:
            VESDE: Initialized VE-SDE model ready for training and sampling.
        """
        # Calculate total dimension: all features plus target (for supervised tasks)
        # In supervised mode, the target is concatenated and diffused jointly
        input_dim = len(self.args.full_feature_list_model) + len(self.args.full_target_list_model)

        # Build the score network: UniModMLP that predicts score vector s_θ(x_t, t)
        # This is a simple MLP with time conditioning via sinusoidal embeddings
        score_net = UniModMLP(
            d_numerical=input_dim,  # Input dimension matches feature + target size
            categories=[],  # No categorical features (all pre-encoded as numerical)
            num_layers=2,  # Shallow network (2 hidden layers)
            d_token=4,  # Token dimension for internal representations
            factor=32,  # Width multiplier for hidden layers
            dim_t=1024,  # Timestep embedding dimension (high-dim for expressiveness)
        )

        # Instantiate the VE-SDE diffusion model with score network and sampler config
        return VESDE(
            num_features=input_dim,
            sigma_min=self.args.model_params["architecture"]["sigma_min"],
            sigma_max=self.args.model_params["architecture"]["sigma_max"],
            score_net=score_net,
            pc_num_steps=self.args.model_params["architecture"]["pc_num_steps"],
            pc_corrector_steps=self.args.model_params["architecture"]["pc_corrector_steps"],
            pc_snr=self.args.model_params["architecture"]["pc_snr"],
        )

    def _compute_loss(self, data_real: torch.Tensor, forward_dict: dict):
        """Compute weighted denoising score matching loss.

        The loss trains the score network to predict the true score (gradient of
        log-density) of the perturbed data distribution. For Gaussian perturbations,
        the true score has a closed form: ∇_x log p(x_t|x_0) = -(x_t - x_0) / σ² = -z/σ.

        Following Song et al. (2020), we weight the loss by σ² to:
        1. Balance contributions across different noise levels during training
        2. Prevent small-noise samples from dominating the gradient
        3. Improve numerical stability and convergence

        Args:
            data_real: Original clean data samples [batch_size, num_features].
            forward_dict: Dictionary from model forward pass containing:
                - score_pred: Network prediction of the score [batch_size, num_features]
                - score_true: Analytical true score for Gaussian perturbation
                - sigma_t: Noise levels [batch_size, 1]

        Returns:
            dict: Loss dictionary with 'total_loss' and 'score_loss' (both identical).
        """
        score_pred = forward_dict["score_pred"]
        score_true = forward_dict["score_true"]
        sigma_t = forward_dict["sigma_t"]

        # Compute weighted score matching loss: E[σ(t)² || s_θ(x_t, t) - ∇log p(x_t|x_0) ||²]
        # The σ² weighting is crucial for balancing across noise levels
        weighted_mse = (sigma_t**2) * F.mse_loss(score_pred, score_true, reduction="none")
        loss = weighted_mse.sum() / data_real.shape[0]  # Average over batch

        return {
            "total_loss": loss,  # Required by Lightning training loop
            "score_loss": loss,  # Same value, logged separately for monitoring
        }

    def _generate(self, num_samples: int):
        """Generate synthetic samples using the trained score network.

        Delegates to the VESDE model's Predictor-Corrector sampler, which
        iteratively denoises random Gaussian noise by following the reverse-time SDE.

        Args:
            num_samples: Number of samples to generate.

        Returns:
            torch.Tensor: Generated samples [num_samples, num_features].
        """
        return self.torch_model.sample(num_samples, device=self.device)


class VESDE(nn.Module):
    """Variance-Exploding SDE (VE-SDE) continuous diffusion model for tabular data.

    This implements a continuous-time diffusion process where the noise level grows
    exponentially with time. Unlike discrete diffusion models (e.g., DDPM), this uses
    a continuous time variable t ∈ [0, 1] and trains a score network to predict the
    gradient of the log-density at any noise level.

    Mathematical Framework:
    ----------------------
    Forward Process (Training):
      1. Sample continuous time: t ~ Uniform(0, 1)
      2. Define noise schedule: σ(t) = σ_min · (σ_max / σ_min)^t
      3. Perturb data: x_t = x_0 + σ(t) · z,  where z ~ N(0, I)
      4. True score: ∇_x log p(x_t|x_0) = -(x_t - x_0) / σ(t)² = -z / σ(t)
      5. Train network s_θ(x_t, t) to predict the true score

    Reverse Process (Sampling):
      1. Initialize from prior: x_T ~ N(0, σ_max² · I)
      2. Discretize time: create grid from t=1 down to t=0
      3. For each time step:
         a. Corrector: Apply Langevin dynamics to refine estimate
         b. Predictor: Step backwards using reverse-time SDE

    The reverse SDE is: dx = -g(t)² · s_θ(x,t) dt + g(t) dW
    where g(t) = σ'(t) controls the diffusion coefficient.

    Args:
        num_features: Total dimension of data vector (features + target if applicable).
        sigma_min: Minimum noise level (small perturbations near clean data).
        sigma_max: Maximum noise level (defines the prior distribution spread).
        score_net: Neural network that predicts score vectors s_θ(x_t, t).
        pc_num_steps: Number of discretization steps for reverse sampling.
        pc_corrector_steps: Number of Langevin MCMC steps per time step (0-2 typical).
        pc_snr: Signal-to-noise ratio for adaptive Langevin step size.
    """

    def __init__(
        self,
        num_features: int,
        sigma_min: float,
        sigma_max: float,
        score_net: nn.Module,
        pc_num_steps: int,
        pc_corrector_steps: int,
        pc_snr: float,
    ):
        super().__init__()

        # Data dimension
        self.num_features = num_features

        # VE-SDE noise schedule parameters
        # σ(t) grows exponentially from σ_min at t=0 to σ_max at t=1
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

        # Score network: MLP that predicts ∇_x log p_t(x_t) for any (x_t, t)
        self.score_net = score_net

        # Predictor-Corrector sampler configuration
        self.pc_num_steps = int(pc_num_steps)  # Time discretization granularity
        self.pc_corrector_steps = int(pc_corrector_steps)  # MCMC refinement iterations
        self.pc_snr = float(pc_snr)  # Controls corrector step size adaptation

        # Precompute schedule constants for efficient computation
        # Exponential schedule: σ(t) = σ_min · k^t  where k = σ_max / σ_min
        self.k = self.sigma_max / self.sigma_min
        self.log_k = float(torch.log(torch.tensor(self.k)))

        # Small epsilon for numerical stability in division operations
        self._eps = 1e-8

    # =============================
    # Forward diffusion + training
    # =============================
    def forward(self, x0: torch.Tensor) -> dict:
        """Forward pass: perturb data and compute score prediction.

        This implements the training procedure for denoising score matching:
        1. Sample random continuous timesteps for each example in the batch
        2. Apply Gaussian perturbation with noise level σ(t)
        3. Compute the true score analytically (closed form for Gaussian)
        4. Predict the score using the neural network
        5. Return both for loss computation

        The true score has a simple closed form for Gaussian perturbations:
        If x_t = x_0 + σ(t) · z where z ~ N(0, I), then:
        ∇_x log p(x_t|x_0) = -(x_t - x_0) / σ² = -z / σ

        Args:
            x0: Clean data samples [batch_size, num_features].

        Returns:
            dict: Contains the following tensors:
                - x_t: Perturbed samples [batch_size, num_features]
                - t: Sampled timesteps [batch_size]
                - sigma_t: Noise levels [batch_size, 1]
                - score_true: Analytical true score [batch_size, num_features]
                - score_pred: Network predicted score [batch_size, num_features]
        """
        B, _ = x0.shape
        device = x0.device

        # Sample random continuous timesteps uniformly from [0, 1]
        # Each example in the batch gets a different noise level for better coverage
        t = torch.rand(B, device=device)

        # Compute noise level for each timestep: σ(t) = σ_min · k^t
        sigma_t = self._sigma(t).view(B, 1)

        # Sample Gaussian noise and perturb clean data: x_t = x_0 + σ(t) · z
        z = torch.randn_like(x0)
        x_t = x0 + sigma_t * z

        # Compute true score for Gaussian perturbation (closed form):
        # ∇_x log p(x_t|x_0) = -(x_t - x_0) / σ² = -z / σ
        score_true = -z / sigma_t

        # Predict score using the time-conditioned neural network
        # Network output: s_θ(x_t, t) ≈ ∇_x log p_t(x_t)
        score_pred = self.score_net(x_num=x_t, x_cat=None, timesteps=t)[0]

        return {
            "score_true": score_true,  # Target for training
            "score_pred": score_pred,  # Network output for training
            "sigma_t": sigma_t,  # Noise level for loss weighting
        }

    # =============================
    # Sampling via Predictor-Corrector
    # =============================
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Generate samples using Predictor-Corrector algorithm.

        Implements the PC sampler from Song et al. (2020), which alternates between:
        1. **Corrector**: Langevin MCMC steps to refine the current estimate
           - Moves towards regions of higher density at the current noise level
           - Uses adaptive step size based on score magnitude (SNR heuristic)
        2. **Predictor**: Reverse-time SDE step to move towards cleaner data
           - Follows the reverse SDE: dx = -g(t)² s_θ(x,t) dt + g(t) dW
           - Gradually reduces noise level as we step backwards in time

        The algorithm starts from pure noise x_T ~ N(0, σ_max² I) and gradually
        denoises by following the learned score field, producing clean samples at t=0.

        Args:
            num_samples: Number of samples to generate.
            device: Device to generate samples on (CPU or CUDA).

        Returns:
            torch.Tensor: Generated samples [num_samples, num_features].
        """
        B = num_samples
        D = self.num_features

        # Initialize from the prior distribution: x_T ~ N(0, σ_max² I)
        # This is the starting point of the reverse diffusion process
        x = self.sigma_max * torch.randn(B, D, device=device)

        # Create uniform time grid from t=0 (clean data) to t=1 (maximum noise)
        # We'll iterate backwards through this grid (t=1 → t=0)
        t_grid = torch.linspace(0.0, 1.0, self.pc_num_steps + 1, device=device)

        # Iterate backwards through time: from maximum noise to clean data
        for i in range(self.pc_num_steps, 0, -1):
            # Current and previous timesteps
            t_i = t_grid[i].expand(B)  # Current time (higher noise)
            t_prev = t_grid[i - 1].expand(B)  # Previous time (lower noise)
            dt = (t_i - t_prev).abs().view(B, 1)  # Time step size (positive)

            # --- Corrector: Langevin dynamics ---
            # Refines the sample using MCMC at the current noise level
            # Update rule: x ← x + ε s_θ(x,t) + √(2ε) z, where z ~ N(0,I)
            if self.pc_corrector_steps > 0:
                for _ in range(self.pc_corrector_steps):
                    # Compute score at current state
                    score = self._score(x, t_i)
                    noise = torch.randn_like(x)

                    # Adaptive step size based on signal-to-noise ratio (Song et al.)
                    # This balances gradient signal strength vs noise magnitude
                    grad_norm = score.view(B, -1).norm(dim=1).mean()
                    # Use theoretical expected norm for D-dimensional Gaussian: sqrt(D)
                    noise_norm = torch.sqrt(torch.tensor(float(D), device=device))
                    step_size = (self.pc_snr * noise_norm / (grad_norm + self._eps)) ** 2 * 2.0
                    step_size = step_size.clamp(min=1e-6)  # Prevent extremely small steps

                    # Langevin update: move in direction of score with added noise
                    # This is equivalent to one step of Langevin MCMC sampling from p_t(x)
                    x = x + step_size * score + torch.sqrt(2.0 * step_size) * noise

            # --- Predictor: reverse-time VE-SDE step ---
            # Solves the reverse SDE: dx = -g(t)² s_θ(x,t) dt + g(t) dW
            # This moves us backwards in time, reducing the noise level
            score = self._score(x, t_i)  # Score at current state
            g = self._g(t_i).view(B, 1)  # Diffusion coefficient g(t)
            noise = torch.randn_like(x)  # Brownian motion increment

            # Euler-Maruyama discretization of reverse SDE
            # Drift term: -g² s_θ dt  (moves towards higher density)
            # Diffusion term: g √dt dW  (adds stochasticity for exploration)
            # Since we iterate backwards (t_i → t_prev with t_prev < t_i),
            # we move in the direction that reduces noise
            x = x + (g**2) * score * dt - g * torch.sqrt(dt) * noise

        return x

    # =============================
    # Schedule utilities
    # =============================
    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Compute noise level at time t using exponential VE schedule.

        The VE-SDE uses an exponentially growing noise schedule:
        σ(t) = σ_min · (σ_max / σ_min)^t = σ_min · k^t

        This creates a smooth interpolation from small perturbations (t≈0)
        to the prior distribution (t≈1). The exponential growth ensures that
        the noise level spans many orders of magnitude, which is crucial for
        score-based models to learn gradients at all scales.

        Args:
            t: Timestep(s) in [0, 1]. Shape can be scalar or [batch_size].

        Returns:
            torch.Tensor: Noise level σ(t), same shape as input t.
        """
        return self.sigma_min * (self.k**t)

    def _g(self, t: torch.Tensor) -> torch.Tensor:
        """Compute diffusion coefficient g(t) for the reverse SDE.

        The reverse-time SDE has the form: dx = -g(t)² s_θ(x,t) dt + g(t) dW
        where g(t) is the diffusion coefficient.

        For the VE-SDE with σ(t) = σ_min · k^t, we have:
        - d/dt σ²(t) = 2 log(k) · σ²(t)  (derivative of variance)
        - Therefore: g(t) = σ(t) · √(2 log k)

        This ensures that the reverse SDE correctly inverts the forward process.

        Args:
            t: Timestep(s) in [0, 1]. Shape can be scalar or [batch_size].

        Returns:
            torch.Tensor: Diffusion coefficient g(t), same shape as input t.
        """
        sigma_t = self._sigma(t)
        return torch.sqrt(torch.tensor(2.0 * self.log_k, device=sigma_t.device)) * sigma_t

    def _score(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict score s_θ(x, t) using the trained neural network.

        The score is the gradient of the log-density: s(x,t) = ∇_x log p_t(x)
        The network is trained to approximate this gradient at any noise level.

        During training, we learn to match the analytical score of Gaussian
        perturbations. During sampling, we use the learned score to navigate
        the reverse-time SDE towards regions of high data density.

        Args:
            x: Input samples [batch_size, num_features].
            t: Timesteps [batch_size].

        Returns:
            torch.Tensor: Predicted score vectors [batch_size, num_features].
        """
        # UniModMLP returns (output, None) tuple; we take the first element
        return self.score_net(x_num=x, x_cat=None, timesteps=t)[0]
