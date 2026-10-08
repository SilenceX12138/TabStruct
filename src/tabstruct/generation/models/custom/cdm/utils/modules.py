import torch
import torch.nn as nn


class DenoiseModel(nn.Module):
    """
    Wrapper class for denoising functions with optional EDM-style preconditioning.

    This class provides a unified interface for denoising models, allowing them to be used
    with or without preconditioning. When preconditioning is enabled, it applies EDM
    (Elucidating Diffusion Models) style input/output scaling for improved training stability.

    Args:
        denoise_fn: The underlying denoising neural network
        sigma_data: Expected standard deviation of the training data (default: 0.5)
        precond: Whether to apply EDM-style preconditioning (default: False)
        net_conditioning: Type of noise conditioning - "sigma" or "t" (default: "sigma")
    """

    def __init__(
        self,
        denoise_fn,
        sigma_data=0.5,
        precond=False,
        net_conditioning="sigma",
    ):
        super().__init__()
        # Store preconditioning flag
        self.precond = precond

        if precond:
            # Wrap denoising function with EDM-style preconditioning
            # This applies input/output scaling for better training dynamics
            self.denoise_fn_D = EDMModel(denoise_fn, sigma_data=sigma_data, net_conditioning=net_conditioning)
        else:
            # Use raw denoising function without any preprocessing
            self.denoise_fn_D = denoise_fn

    def forward(self, x_num, x_cat, t, sigma=None):
        """
        Forward pass through the denoising model.

        Args:
            x_num: Numerical features tensor [batch_size, num_numerical_features]
            x_cat: Categorical features tensor [batch_size, num_categorical_features]
            t: Timestep tensor [batch_size] - current diffusion timestep
            sigma: Optional noise level tensor [batch_size, num_features] - only used if precond=True

        Returns:
            Tuple of (denoised_numerical, denoised_categorical) predictions
        """
        if self.precond:
            # Use preconditioning wrapper that applies EDM-style scaling and conditioning
            # Requires sigma parameter for proper noise level conditioning
            return self.denoise_fn_D(x_num, x_cat, t, sigma)
        else:
            # Use raw denoising function without preconditioning
            # Standard approach that only uses timestep conditioning
            return self.denoise_fn_D(x_num, x_cat, t)


class EDMModel(nn.Module):
    """
    EDM (Elucidating Diffusion Models) style preconditioning wrapper.

    This class implements the preconditioning scheme from "Elucidating the Design Space
    of Diffusion-Based Generative Models" by Karras et al. It applies input/output scaling
    to improve training stability and convergence for diffusion models.

    The preconditioning transforms the denoising network prediction according to:
    D(x) = c_skip * x + c_out * F(c_in * x, c_noise)

    where F is the underlying denoising network and c_skip, c_out, c_in, c_noise are
    noise-level dependent scaling factors.

    Args:
        denoise_fn: The underlying denoising neural network
        sigma_data: Expected standard deviation of the training data
        net_conditioning: Type of noise conditioning - "sigma" (log noise level) or "t" (timestep)
    """

    def __init__(
        self,
        denoise_fn,
        sigma_data,
        net_conditioning,
    ):
        super().__init__()
        # Store expected data standard deviation for scaling computations
        self.sigma_data = sigma_data
        # Store conditioning type (sigma vs timestep)
        self.net_conditioning = net_conditioning
        # Store the underlying denoising network
        self.denoise_fn_F = denoise_fn

        # === Sanity checks ===
        if net_conditioning not in ["sigma", "t"]:
            raise ValueError("net_conditioning must be either 'sigma' or 't'")

    def forward(self, x_num, x_cat, t, sigma):
        """
        Apply EDM-style preconditioning to the denoising process.

        Args:
            x_num: Numerical features tensor [batch_size, num_numerical_features]
            x_cat: Categorical features tensor [batch_size, num_categorical_features]
            t: Timestep tensor [batch_size] - current diffusion timestep
            sigma: Noise level tensor [batch_size, num_features] - current noise scale

        Returns:
            Tuple of (preconditioned_numerical, categorical_prediction)
        """
        # === Sanity checks ===
        if self.net_conditioning == "sigma" and t is not None:
            raise ValueError("t must be None when net_conditioning is 'sigma'")
        if self.net_conditioning == "t" and t is None:
            raise ValueError("t must be provided when net_conditioning is 't'")

        # Ensure numerical precision for preconditioning computations
        x_num = x_num.to(torch.float32)
        sigma = sigma.to(torch.float32)

        # Compute EDM preconditioning coefficients based on noise level
        # c_skip: Skip connection weight (how much original signal to preserve)
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)

        # c_out: Output scaling factor (scales the network prediction)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()

        # c_in: Input scaling factor (normalizes input by total noise level)
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()

        # c_noise: Noise conditioning (log of noise level, scaled for network input)
        c_noise = sigma.log() / 4

        # Apply input scaling to numerical features
        x_in = c_in * x_num

        # Prepare conditioning input based on chosen conditioning type
        timestep = c_noise.flatten() if self.net_conditioning == "sigma" else t

        # Forward pass through the underlying denoising network
        if x_cat is None:
            F_x = self.denoise_fn_F(x_in, timestep)
            x_cat_pred = None
        else:
            F_x, x_cat_pred = self.denoise_fn_F(x_in, x_cat, timestep)

        # Apply EDM preconditioning formula: D(x) = c_skip * x + c_out * F(...)
        # This combines the skip connection with the scaled network prediction
        D_x = c_skip * x_num + c_out * F_x.to(torch.float32)

        return D_x, x_cat_pred
