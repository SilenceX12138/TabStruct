import torch


class PowerMeanNoise:
    """Power-mean noise schedule for numerical features (Karras et al.).

    Implements polynomial interpolation between σ_min and σ_max:
        σ(t) = (σ_min^(1/ρ) + t * (σ_max^(1/ρ) - σ_min^(1/ρ)))^ρ

    where t ∈ [0, 1] is normalized time. Higher ρ allocates more steps
    to low noise levels, improving quality for fine details.

    This schedule is used for continuous-time diffusion of numerical features,
    providing better noise coverage than linear or exponential schedules.

    Key Properties:
        - σ(0) = σ_min (near-clean data)
        - σ(1) = σ_max (maximum corruption)
        - Higher ρ → more steps at low noise (default ρ=7 for TabDiff)
        - Bidirectional: supports both forward (t → σ) and inverse (σ → t)

    Example:
        >>> schedule = PowerMeanNoise(sigma_min=0.002, sigma_max=80.0, rho=7.0)
        >>> t = torch.tensor([0.0, 0.5, 1.0])
        >>> sigma = schedule.total_noise(t)  # [0.002, ~1.0, 80.0]
        >>> t_recovered = schedule.inverse_to_t(sigma)  # [0.0, 0.5, 1.0]
    """

    def __init__(self, sigma_min: float, sigma_max: float, rho: float):
        """Initialize noise schedule.

        Args:
            sigma_min: Minimum noise level (near-clean data, typically 0.002)
            sigma_max: Maximum noise level (high corruption, typically 80.0)
            rho: Polynomial exponent (typically 7.0 for TabDiff/EDM)
        """
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.rho = float(rho)

        # Precompute transformed bounds for efficiency
        self._sigma_min_root = self.sigma_min ** (1.0 / self.rho)
        self._sigma_max_root = self.sigma_max ** (1.0 / self.rho)

    @staticmethod
    def _as_tensor(value, device=None) -> torch.Tensor:
        """Convert value to float32 tensor.

        Helper method for ensuring inputs are tensors with correct dtype.

        Args:
            value: Input value (tensor, scalar, or array-like)
            device: Optional device for tensor allocation

        Returns:
            Float32 tensor
        """
        if torch.is_tensor(value):
            return value.to(torch.float32)
        return torch.tensor(value, dtype=torch.float32, device=device)

    def total_noise(self, t: torch.Tensor) -> torch.Tensor:
        """Compute noise level σ(t) from normalized time t ∈ [0,1].

        Args:
            t: Normalized time [batch] or [batch, 1]

        Returns:
            Noise level σ(t) with same shape as input
        """
        t = self._as_tensor(t, device=t.device if torch.is_tensor(t) else None)
        t = t.view(-1, 1)
        t = t.clamp(0.0, 1.0)
        # Polynomial interpolation in transformed space
        sigma_root = self._sigma_min_root + t * (self._sigma_max_root - self._sigma_min_root)
        return sigma_root.pow(self.rho)

    def inverse_to_t(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute normalized time t from noise level σ (inverse mapping).

        Args:
            sigma: Noise level [batch] or [batch, 1]

        Returns:
            Normalized time t ∈ [0,1] with same shape as input
        """
        sigma = self._as_tensor(sigma, device=sigma.device if torch.is_tensor(sigma) else None)
        if sigma.ndim == 1:
            sigma = sigma[:, None]
        sigma = sigma.clamp(min=self.sigma_min, max=self.sigma_max)
        # Invert polynomial transformation
        sigma_root = sigma.pow(1.0 / self.rho)
        denom = max(self._sigma_max_root - self._sigma_min_root, 1e-8)
        t = (sigma_root - self._sigma_min_root) / denom
        return t.clamp(0.0, 1.0)


class LogLinearNoise:
    """Log-linear noise schedule for categorical features (absorbing diffusion).

    Implements log-linear schedule for masking probability:
        σ(t) = -log(1 - t)

    where t ∈ [0, 1) is normalized time. As t → 1, σ → ∞, corresponding
    to fully masked categorical features. The schedule provides a smooth
    interpolation for the absorbing diffusion process where categories
    gradually transition to mask tokens.

    This differs from numerical diffusion noise schedules as it operates
    on discrete categorical spaces rather than continuous Gaussian noise.

    Key Properties:
        - σ(0) = 0 (no masking)
        - σ(t) → ∞ as t → 1 (fully masked)
        - Masking probability: p_mask = 1 - exp(-σ) = t
        - Rate: dσ/dt = 1/(1-t) (increases near t=1)

    The relationship to masking:
        α(t) = exp(-σ(t)) = 1 - t  (probability of staying unmasked)
        p_mask(t) = 1 - α(t) = t    (probability of being masked)

    Example:
        >>> schedule = LogLinearNoise(num_categories=5)
        >>> t = torch.tensor([[0.0], [0.5], [0.9]])
        >>> sigma = schedule.total_noise(t)  # [0, ~0.69, ~2.30] × 5 features
        >>> rate = schedule.rate_noise(t)    # [1.0, 2.0, 10.0] × 5 features
    """

    def __init__(self, num_categories: int):
        """Initialize categorical noise schedule.

        Args:
            num_categories: Number of categorical features (after one-hot encoding)
        """
        self.num_categories = int(num_categories)

    @staticmethod
    def _prepare_t(t: torch.Tensor) -> torch.Tensor:
        """Standardize time tensor to [batch, 1] shape and clamp to valid range.

        Args:
            t: Time values (scalar, 1D, or 2D tensor)

        Returns:
            Clamped time tensor of shape [batch, 1] in range [0, 1 - ε)
        """
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.float32)
        t = t.to(torch.float32)
        t = t.view(-1, 1)
        return t.clamp(0.0, 1.0 - 1e-5)  # Prevent t=1 to avoid log(0)

    def total_noise(self, t: torch.Tensor) -> torch.Tensor:
        """Compute noise level σ(t) = -log(1 - t) for categorical features.

        This represents the accumulated masking intensity. Higher σ means
        more categories are replaced with mask tokens.

        Args:
            t: Normalized time [batch] or [batch, 1]

        Returns:
            Noise level σ(t) of shape [batch, num_categories]
            Returns empty tensor if num_categories=0
        """
        t = self._prepare_t(t)
        sigma = -torch.log(torch.clamp(1.0 - t, min=1e-5))

        # Handle edge case of no categorical features
        if self.num_categories == 0:
            return torch.zeros(t.shape[0], 0, device=t.device, dtype=t.dtype)

        # Expand to [batch, num_categories] for broadcasting with one-hot features
        return sigma.expand(-1, self.num_categories)

    def rate_noise(self, t: torch.Tensor) -> torch.Tensor:
        """Compute noise rate dσ/dt = 1/(1-t) for categorical features.

        The rate increases as t → 1, reflecting accelerating masking
        near the end of the forward diffusion process. Used in loss
        weighting for continuous-time training.

        Args:
            t: Normalized time [batch] or [batch, 1]

        Returns:
            Noise rate dσ/dt of shape [batch, num_categories]
            Returns empty tensor if num_categories=0
        """
        t = self._prepare_t(t)
        rate = 1.0 / torch.clamp(1.0 - t, min=1e-5)
        # Handle edge case of no categorical features
        if self.num_categories == 0:
            return torch.zeros(t.shape[0], 0, device=t.device, dtype=t.dtype)
        # Expand to [batch, num_categories] for broadcasting
        return rate.expand(-1, self.num_categories)
