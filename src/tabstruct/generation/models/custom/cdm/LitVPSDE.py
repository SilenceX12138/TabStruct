"""Variance Preserving SDE diffusion model for tabular generation.

This module implements the Variance-Preserving SDE (VP-SDE) framework from
"Score-Based Generative Modeling through Stochastic Differential Equations"
(Song et al., 2021). Unlike VE-SDE which grows noise exponentially, VP-SDE
preserves variance by gradually replacing signal with noise.

Key Differences from VE-SDE:
    - Forward process: x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) z  [variance-preserving]
    - Noise schedule: Linear β(t) schedule, common in discrete DDPM
    - Reverse SDE: Includes both drift from data structure (-0.5 β x) and score
    - Sample initialization: From standard Gaussian (not scaled by σ_max)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...BaseGenerator import BaseLightningGenerationModule, BaseLitJointGenerator
from ..ddpm.utils.modules import UniModMLP


class LitVPSDE(BaseLitJointGenerator):
    """Variance-Preserving SDE (VP-SDE) continuous diffusion model for tabular data.

    Design notes:
    - Treat all features as numerical: unified continuous diffusion over full vector
    - Forward diffusion: VP-SDE with linear β(t) schedule and variance preservation
    - Training loss: weighted score matching with σ²(t) = (1 - ᾱ(t)) weighting
    - Sampling: Predictor-Corrector (reverse-time SDE + Langevin corrector)
    - API mirrors LitCDM for consistency across diffusion variants
    """

    def __init__(self, args):
        """Initialize the Variance-Preserving SDE Diffusion Model.

        Args:
            args: Argument namespace containing model configuration and task type.
                  Must specify task as one of: classification, regression, unsupervision.
        """
        super().__init__(args)

        # Validate task type - VP-SDE supports all standard tabular tasks
        if args.task not in ["classification", "regression", "unsupervision"]:
            raise ValueError(f"Task {args.task} is not supported for {self.name} model")

        # Initialize the Lightning module that handles training/generation
        self.model = _LitVPSDE(args)

    # ================================================================
    # =                      Hyperparams                             =
    # ================================================================
    @classmethod
    def _define_default_params(cls):
        """Define default hyperparameters for production training.

        Returns:
            dict: Nested dictionary with 'architecture' and 'optimization' parameters.
                  Architecture params control the VP-SDE schedule and sampler.
                  Optimization params control the learning rate and regularization.
        """
        params_arch = {
            # VP-SDE linear β schedule parameters
            "beta_min": 0.1,  # Minimum diffusion rate at t=0 (small perturbations)
            "beta_max": 20.0,  # Maximum diffusion rate at t=1 (approaching prior)
            # Predictor-Corrector sampler configuration
            "pc_num_steps": 1000,  # Number of reverse diffusion steps (higher = better quality)
            "pc_corrector_steps": 1,  # Langevin MCMC steps per time step (0-2 typical)
            "pc_snr": 0.16,  # Signal-to-noise ratio for adaptive corrector step size
        }

        params_optim = {
            "lr": 1e-3,  # Learning rate for Adam optimizer
            "weight_decay": 1e-5,  # L2 regularization coefficient
        }

        return {"architecture": params_arch, "optimization": params_optim}

    @classmethod
    def _define_single_run_params(cls):
        """Define parameters for quick single development runs.

        Uses faster settings with reduced sampling steps to accelerate
        iteration during development and debugging.

        Returns:
            dict: Lightweight parameters suitable for rapid experimentation.
        """
        params_arch = {
            "beta_min": 0.1,
            "beta_max": 10.0,  # Lower max for faster diffusion
            "pc_num_steps": 200,  # Fewer sampling steps for speed
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
        """Define minimal parameters for unit testing and CI/CD.

        Uses the smallest viable configuration to ensure tests run quickly
        while still exercising all code paths.

        Returns:
            dict: Minimal parameters for fast testing.
        """
        params_arch = {
            "beta_min": 0.1,
            "beta_max": 5.0,  # Minimal diffusion range for testing
            "pc_num_steps": 50,  # Very few steps for speed
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
        """Configure data preprocessing for VP-SDE.

        VP-SDE applies a unified continuous diffusion process, but unlike VE-SDE,
        it can work with the default categorical handling since the variance-preserving
        property makes it more robust to discrete features.

        Note: Set categorical_as_numerical=True if you want to force all features
        into the continuous diffusion space (more theoretically pure but may lose
        categorical structure).

        Returns:
            dict: Scaler configuration with standard preprocessing settings.
        """
        return {
            "context": {
                "disable_preprocessing": False,  # Keep preprocessing active
            },
            "feature_scaler": {
                "categorical_transform": "ordinal",  # Encode categories as integers
                "categorical_as_numerical": False,  # Keep categorical/numerical distinction
            },
            "target_scaler": {},
        }


class _LitVPSDE(BaseLightningGenerationModule):
    """Lightning module for Variance-Preserving SDE training and inference.

    Implements the PyTorch Lightning interface for the VP-SDE model, handling:
    - Model initialization and architecture setup
    - Loss computation (weighted denoising score matching)
    - Sample generation via the trained score network
    """

    def __init__(self, args):
        """Initialize the Lightning module.

        Args:
            args: Argument namespace containing model configuration.
        """
        super().__init__(args)

    # ================================================================
    # =                     Model-specific                           =
    # ================================================================
    def _create_torch_model(self):
        """Create the VP-SDE model with time-conditioned score network.

        Constructs a UniModMLP-based score network that predicts the score function
        (gradient of log density) for any noisy sample and timestep. This network
        architecture is designed for tabular data and includes timestep conditioning.

        Returns:
            VPSDE: Initialized variance-preserving diffusion model ready for training.
        """
        # Calculate total dimension: all features plus target (for supervised tasks)
        input_dim = len(self.args.full_feature_list_model) + len(self.args.full_target_list_model)

        # Build the score network: UniModMLP that predicts score vector s_theta(x_t, t)
        # This is a specialized MLP for tabular data with timestep embedding
        score_net = UniModMLP(
            d_numerical=input_dim,  # Input dimension (all features treated as numerical)
            categories=[],  # No categorical features in the diffusion space
            num_layers=2,  # Number of hidden layers
            d_token=4,  # Token dimension for internal processing
            factor=32,  # Width multiplier for hidden dimensions
            dim_t=1024,  # Timestep embedding dimension
        )

        # Instantiate the VP-SDE diffusion model with all components
        return VPSDE(
            num_features=input_dim,
            beta_min=self.args.model_params["architecture"]["beta_min"],
            beta_max=self.args.model_params["architecture"]["beta_max"],
            score_net=score_net,
            pc_num_steps=self.args.model_params["architecture"]["pc_num_steps"],
            pc_corrector_steps=self.args.model_params["architecture"]["pc_corrector_steps"],
            pc_snr=self.args.model_params["architecture"]["pc_snr"],
        )

    def _compute_loss(self, data_real: torch.Tensor, forward_dict: dict):
        """Compute weighted denoising score matching loss.

        The loss trains the score network to predict the true score (gradient of log density)
        of the perturbed data distribution. For VP-SDE with Gaussian perturbations:
            x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) z
        The true score has closed form: score_true = -z / √(1-ᾱ_t)

        Following Song et al. (2021), we weight the loss by σ²(t) = (1-ᾱ_t) to balance
        contributions across different noise levels and improve training stability.

        Args:
            data_real: Original clean data samples [batch_size, num_features].
            forward_dict: Dictionary from model forward pass containing:
                - score_pred: Network prediction of the score [batch_size, num_features]
                - score_true: Analytical true score for Gaussian perturbation
                - std_t: Noise standard deviation √(1-ᾱ_t) [batch_size, 1]

        Returns:
            dict: Loss dictionary with 'total_loss' and 'score_loss' (both identical).
        """
        # Extract predicted and true scores from forward pass
        score_pred = forward_dict["score_pred"]
        score_true = forward_dict["score_true"]
        std_t = forward_dict["std_t"]

        # Compute weighted score matching loss: E[σ²(t) ||s_θ(x_t, t) - ∇log p(x_t|x_0)||²]
        # The σ² weighting is crucial for balancing high vs low noise contributions
        weight = std_t**2
        weighted_mse = weight * F.mse_loss(score_pred, score_true, reduction="none")
        loss = weighted_mse.sum() / data_real.shape[0]

        return {
            "total_loss": loss,  # Required by Lightning training loop
            "score_loss": loss,  # Same value, logged separately for monitoring
        }

    def _generate(self, num_samples: int):
        """Generate synthetic samples using the trained score network.

        Args:
            num_samples: Number of samples to generate.

        Returns:
            torch.Tensor: Generated samples [num_samples, num_features].
        """
        return self.torch_model.sample(num_samples, device=self.device)


class VPSDE(nn.Module):
    """Variance-Preserving SDE (VP-SDE) continuous diffusion model for tabular data.

    This implements a continuous-time diffusion process where the total variance
    is preserved at 1.0 throughout the diffusion process. As noise increases,
    signal decreases proportionally. This is the continuous-time analog of DDPM.

    Mathematical Framework:
    ----------------------
    Forward Process (Training):
      1. Sample continuous time: t ~ Uniform(0, 1)
      2. Define linear schedule: β(t) = β_min + t(β_max - β_min)
      3. Compute mean coefficient: ᾱ(t) = exp(-∫₀ᵗ β(s)ds)
      4. Perturb data: x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) z,  where z ~ N(0, I)
      5. True score: ∇_x log p(x_t|x_0) = -z / √(1-ᾱ_t)
      6. Train network s_θ(x_t, t) to predict the true score

    Reverse Process (Sampling):
      1. Initialize from prior: x_T ~ N(0, I)  [standard Gaussian, not scaled]
      2. Discretize time: create grid from t=1 down to t=0
      3. For each time step:
         a. Corrector: Apply Langevin dynamics to refine estimate
         b. Predictor: Step backwards using reverse-time VP-SDE

    The reverse VP-SDE is: dx = [-0.5 β(t) x - β(t) s_θ(x,t)] dt + √β(t) dW
    Note the extra drift term -0.5 β(t) x compared to VE-SDE.

    Args:
        num_features: Total dimension of data vector (features + target if applicable).
        beta_min: Minimum diffusion rate at t=0 (small perturbations).
        beta_max: Maximum diffusion rate at t=1 (approaching prior).
        score_net: Neural network that predicts score vectors.
        pc_num_steps: Number of discretization steps for reverse sampling.
        pc_corrector_steps: Number of Langevin steps per time step (0-2 typical).
        pc_snr: Signal-to-noise ratio for adaptive Langevin step size.
    """

    def __init__(
        self,
        num_features: int,
        beta_min: float,
        beta_max: float,
        score_net: nn.Module,
        pc_num_steps: int,
        pc_corrector_steps: int,
        pc_snr: float,
    ):
        super().__init__()

        # Data dimension
        self.num_features = num_features

        # VP-SDE linear β schedule parameters
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.beta_diff = self.beta_max - self.beta_min

        # Neural network component
        self.score_net = score_net  # UniModMLP that predicts score vectors

        # Predictor-Corrector sampler configuration
        self.pc_num_steps = int(pc_num_steps)  # Time discretization granularity
        self.pc_corrector_steps = int(pc_corrector_steps)  # MCMC refinement iterations
        self.pc_snr = float(pc_snr)  # Controls corrector step size adaptation

        # Small epsilon for numerical stability
        self._eps = 1e-8

    # =============================
    # Forward diffusion + training
    # =============================
    def forward(self, x0: torch.Tensor) -> dict:
        """Forward pass: perturb data and compute score prediction.

        This implements the training procedure for denoising score matching in VP-SDE:
        1. Sample random continuous timesteps for each example in the batch
        2. Compute ᾱ(t) from the integrated β schedule
        3. Apply variance-preserving Gaussian perturbation: x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) z
        4. Compute the true score analytically (closed form for Gaussian)
        5. Predict the score using the neural network
        6. Return both for loss computation

        Args:
            x0: Clean data samples [batch_size, num_features].

        Returns:
            dict: Contains the following tensors:
                - score_true: Analytical true score [batch_size, num_features]
                - score_pred: Network predicted score [batch_size, num_features]
                - std_t: Noise standard deviation √(1-ᾱ_t) [batch_size, 1]
        """
        B, _ = x0.shape
        device = x0.device

        # Sample random continuous timesteps uniformly from [0, 1]
        t = torch.rand(B, device=device)

        # Compute ᾱ(t) = exp(-∫₀ᵗ β(s)ds) - the cumulative signal retention
        alpha_bar = self._alpha_bar(t).view(B, 1)
        mean_coeff = torch.sqrt(alpha_bar)  # Signal coefficient √(ᾱ_t)
        std_t = torch.sqrt(torch.clamp(1.0 - alpha_bar, min=1e-12))  # Noise std √(1-ᾱ_t)

        # Sample Gaussian noise and apply variance-preserving perturbation
        z = torch.randn_like(x0)
        x_t = mean_coeff * x0 + std_t * z

        # Compute true score for Gaussian perturbation (closed form)
        # For p(x_t | x_0) = N(√(ᾱ_t) x_0, (1-ᾱ_t) I), the score is: ∇_x log p = -z/√(1-ᾱ_t)
        score_true = -z / std_t

        # Predict score using the time-conditioned neural network
        score_pred = self.score_net(x_num=x_t, x_cat=None, timesteps=t)[0]

        return {
            "score_true": score_true,  # Target for training
            "score_pred": score_pred,  # Network output for training
            "std_t": std_t,  # Noise level for loss weighting
        }

    # =============================
    # Sampling via Predictor-Corrector
    # =============================
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Generate samples using Predictor-Corrector algorithm for VP-SDE.

        Implements the PC sampler from Song et al. (2021), which alternates between:
        1. Corrector: Langevin MCMC steps to refine the current estimate
        2. Predictor: Reverse-time SDE step to move towards cleaner data

        The algorithm starts from standard Gaussian noise (not scaled like VE-SDE) and
        gradually denoises by following the reverse-time VP-SDE:
            dx = [-0.5 β(t) x - β(t) s_θ(x,t)] dt + √β(t) dW

        Key difference from VE-SDE: The drift includes both score and data structure terms.

        Args:
            num_samples: Number of samples to generate.
            device: Device to generate samples on (CPU or CUDA).

        Returns:
            torch.Tensor: Generated samples [num_samples, num_features].
        """
        B = num_samples
        D = self.num_features

        # Initialize from the prior distribution: x_T ~ N(0, I)
        # Note: For VP-SDE, we start from STANDARD Gaussian (not scaled by σ_max)
        # This is because the forward process preserves variance at 1.0
        x = torch.randn(B, D, device=device)

        # Create uniform time grid from t=0 (clean data) to t=1 (maximum noise)
        t_grid = torch.linspace(0.0, 1.0, self.pc_num_steps + 1, device=device)
        eps = self._eps

        # Iterate backwards through time (from noisy to clean)
        for i in range(self.pc_num_steps, 0, -1):
            # Current and previous timesteps
            t_i = t_grid[i].expand(B)
            t_prev = t_grid[i - 1].expand(B)
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
                    # Use theoretical expected norm for D-dimensional Gaussian: √D
                    noise_norm = torch.sqrt(torch.tensor(float(D), device=device))
                    step_size = (self.pc_snr * noise_norm / (grad_norm + eps)) ** 2 * 2.0
                    step_size = step_size.clamp(min=1e-6)  # Prevent extremely small steps

                    # Langevin update: move in direction of score with added noise
                    x = x + step_size * score + torch.sqrt(2.0 * step_size) * noise

            # --- Predictor: reverse-time VP-SDE step ---
            # Solves the reverse VP-SDE: dx = [-0.5 β x - β s_θ] dt + √β dW
            score = self._score(x, t_i)  # Score at current state
            beta_t = self._beta(t_i).view(B, 1)  # Diffusion rate β(t)
            noise = torch.randn_like(x)  # Brownian motion increment

            # VP-SDE drift has TWO components (unlike VE-SDE):
            # 1. Data structure term: -0.5 β(t) x  (pulls toward origin, preserves variance)
            # 2. Score term: -β(t) s_θ(x,t)  (pulls toward high-density regions)
            drift = -0.5 * beta_t * x - beta_t * score

            # Diffusion coefficient for VP-SDE
            diffusion = torch.sqrt(torch.clamp(beta_t * dt, min=1e-12))

            # Euler-Maruyama discretization of reverse VP-SDE
            # Since we iterate backwards (t_i → t_prev with t_prev < t_i),
            # we subtract the drift+diffusion to move towards cleaner data
            x = x - drift * dt + diffusion * noise

        return x

    # =============================
    # Schedule utilities
    # =============================
    def _beta(self, t: torch.Tensor) -> torch.Tensor:
        """Compute diffusion rate β(t) using linear schedule.

        The VP-SDE uses a linear β schedule: β(t) = β_min + t(β_max - β_min)
        This controls the rate at which signal is replaced by noise.

        Args:
            t: Timestep(s) in [0, 1]. Shape can be scalar or [batch_size].

        Returns:
            torch.Tensor: Diffusion rate β(t), same shape as input t.
        """
        return self.beta_min + t * self.beta_diff

    def _integral_beta(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the integral ∫₀ᵗ β(s)ds for the VP-SDE schedule.

        For linear β(t) = β_min + t(β_max - β_min), the integral is:
        ∫₀ᵗ β(s)ds = β_min·t + 0.5(β_max - β_min)·t²

        This integral is used to compute ᾱ(t) = exp(-∫₀ᵗ β(s)ds), which controls
        the signal retention in the forward process.

        Args:
            t: Timestep(s) in [0, 1]. Shape can be scalar or [batch_size].

        Returns:
            torch.Tensor: Integral value, same shape as input t.
        """
        return self.beta_min * t + 0.5 * self.beta_diff * (t**2)

    def _alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Compute mean coefficient ᾱ(t) for the forward process.

        The cumulative signal retention coefficient is:
        ᾱ(t) = exp(-∫₀ᵗ β(s)ds)

        This controls how much of the original signal x_0 remains in x_t:
        x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) z

        At t=0: ᾱ(0) = 1 → x_0 = x_0 (no noise)
        At t=1: ᾱ(1) ≈ 0 → x_1 ≈ z (pure noise)

        Args:
            t: Timestep(s) in [0, 1]. Shape can be scalar or [batch_size].

        Returns:
            torch.Tensor: Mean coefficient ᾱ(t), same shape as input t.
        """
        return torch.exp(-self._integral_beta(t))

    def _score(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict score s_θ(x, t) using the trained neural network.

        The score is the gradient of the log-density: s(x,t) = ∇_x log p_t(x)
        The network is trained to approximate this gradient at any noise level.

        Args:
            x: Input samples [batch_size, num_features].
            t: Timesteps [batch_size].

        Returns:
            torch.Tensor: Predicted score vectors [batch_size, num_features].
        """
        # UniModMLP returns a tuple, we take the first element (numerical output)
        return self.score_net(x_num=x, x_cat=None, timesteps=t)[0]
