import torch


def get_beta_schedule(
    beta_start: float,
    beta_end: float,
    num_timesteps: int,
    schedule_mode: str,
):
    """Get beta schedule for the diffusion process.

    Beta values control how much noise is added at each timestep in the forward diffusion process.
    Different schedules provide different noise addition strategies:
    - Linear: Uniform noise increase from beta_start to beta_end
    - Cosine: Smooth cosine-based schedule that adds noise slowly at first, then faster

    Args:
        beta_start: Starting beta value (used only for linear schedule)
        beta_end: Ending beta value (used only for linear schedule)
        num_timesteps: Number of diffusion timesteps
        schedule_mode: "linear" or "cosine"

    Returns:
        torch.Tensor: Beta values for each timestep, shape (num_timesteps,)
    """
    # === Sanity checks ===
    if beta_start > beta_end:
        raise ValueError(f"beta_start must be less than or equal to beta_end, got {beta_start} > {beta_end}")

    # === Generate beta schedule ===
    if schedule_mode == "linear":
        # Simple linear interpolation from beta_start to beta_end
        return torch.linspace(beta_start, beta_end, num_timesteps)
    elif schedule_mode == "cosine":
        # Small offset to prevent division by zero and ensure smooth start
        s = 0.008
        # Extra step for boundary calculations
        steps = num_timesteps + 1

        # Create normalized time grid [0, 1]
        x = torch.linspace(0, num_timesteps, steps)

        # Generate cosine-based alpha cumulative product schedule
        # Maps time to [0, π/2] range and applies cosine for smooth decay
        alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2

        # Normalize so that alphas_cumprod[0] = 1 (no noise at t=0)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

        # Convert cumulative alphas to individual betas
        # β_t = 1 - α_t = 1 - (ᾱ_t / ᾱ_{t-1})
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

        # Clamp to prevent numerical instability
        return torch.clamp(betas, 0.0, 0.999)
    else:
        raise NotImplementedError(f"Beta schedule {schedule_mode} not implemented")
