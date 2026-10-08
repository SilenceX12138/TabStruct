from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from lightning.pytorch.callbacks import Callback, EarlyStopping

if TYPE_CHECKING:  # pragma: no cover - only used for typing
    from lightning.pytorch import LightningModule, Trainer


class AdaptiveKLBeta(Callback):
    """Dynamically decay the KL divergence weight (beta) when validation loss plateaus.

    This callback monitors a validation metric and reduces the beta coefficient
    in variational models when the metric stops improving for a specified number
    of epochs (patience). Beta is decayed multiplicatively and clamped to [beta_min, beta_max].

    Args:
        monitor: Name of the validation metric to track (e.g., "valid_metrics/vae_loss").
        beta_max: Maximum allowed value for beta coefficient (upper bound).
        beta_min: Minimum allowed value for beta coefficient (lower bound after decay).
        decay_rate: Multiplicative decay factor applied to beta when plateau is detected.
                   Should be in (0, 1), e.g., 0.7 means beta *= 0.7.
        patience_validation_epoch: Number of epochs without improvement before triggering decay.

    Attributes:
        best_metric: Best (lowest) metric value observed so far.
        epochs_without_improvement: Counter tracking epochs since last improvement.
    """

    def __init__(
        self,
        monitor: str,
        beta_max: float = 1e-2,
        beta_min: float = 1e-5,
        decay_rate: float = 0.7,
        patience_validation_epoch: int = 10,
    ) -> None:
        super().__init__()

        # Metric monitoring configuration
        self.monitor: str = monitor

        # Beta decay configuration
        self.beta_max: float = beta_max
        self.beta_min: float = beta_min
        self.decay_rate: float = decay_rate

        # Patience tracking: ensure at least 1 epoch
        self.patience_epochs: int = max(1, patience_validation_epoch)
        self.epochs_without_improvement: int = 0

        # Best metric tracking: None until first validation
        self.best_metric: Optional[float] = None

    def on_train_epoch_start(self, trainer, pl_module):
        # Ensure the pl_module has a 'beta' attribute
        if not hasattr(pl_module, "beta"):
            raise AttributeError("AdaptiveKLBeta callback requires 'pl_module' to have a 'beta' attribute.")

        for logger in trainer.loggers:
            logger.log_metrics({"kl_beta": pl_module.beta}, step=trainer.fit_loop.epoch_loop._batches_that_stepped)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Adjust ``pl_module.beta`` when the monitored validation metric plateaus.

        Decay logic:
        1. Skip during sanity checks or non-training phases.
        2. Track the best (lowest) metric value seen.
        3. Reset patience counter when metric improves.
        4. Increment counter when metric doesn't improve.
        5. When counter reaches patience threshold, decay beta and reset counter.
        """
        # Ensure the pl_module has a 'beta' attribute
        if not hasattr(pl_module, "beta"):
            raise AttributeError("AdaptiveKLBeta callback requires 'pl_module' to have a 'beta' attribute.")
        if not hasattr(pl_module, "train_vae"):
            raise AttributeError("AdaptiveKLBeta callback requires 'pl_module' to have a 'train_vae' attribute.")

        # Skip during sanity checks or testing phases
        if self._should_skip_check(trainer):
            return

        # Only adjust beta during VAE training phase
        if not pl_module.train_vae:
            return

        # Retrieve the monitored metric from trainer callbacks
        metric_tensor = trainer.callback_metrics.get(self.monitor)
        if metric_tensor is None:
            # Metric not yet available (e.g., first epoch); skip quietly
            return

        current_metric: float = float(metric_tensor)

        # Check if this is the best metric we've seen (lower is better)
        if self.best_metric is None or current_metric < self.best_metric:
            # Improvement detected: update best metric and reset patience
            self.best_metric = current_metric
            self.epochs_without_improvement = 0
        else:
            # No improvement: increment patience counter
            self.epochs_without_improvement += 1

            # Check if patience threshold reached
            if self.epochs_without_improvement >= self.patience_epochs:
                # Plateau detected: decay beta and clamp to valid range
                old_beta = pl_module.beta
                new_beta = old_beta * self.decay_rate
                pl_module.beta = max(self.beta_min, min(self.beta_max, new_beta))

                # Reset patience counter after decay
                self.epochs_without_improvement = 0

    def _should_skip_check(self, trainer: Trainer) -> bool:
        """Determine if the callback should skip processing this validation end.

        Returns:
            True if currently in sanity check mode or not in training phase.
            False if normal validation during training.
        """
        from lightning.pytorch.trainer.states import TrainerFn

        # Skip if not in fitting phase or during sanity validation
        return trainer.state.fn != TrainerFn.FITTING or trainer.sanity_checking


class EarlyStoppingVAE(Callback):
    def __init__(self, max_steps: int, monitor: str, beta_min: float, patience_validation_epoch: int):
        super().__init__()
        self.max_steps: int = max_steps
        self.monitor: str = monitor
        self.beta_min: float = beta_min

        self.patience_epochs: int = max(1, patience_validation_epoch)
        self.epochs_without_improvement: int = 0

        self.best_metric: Optional[float] = None

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not hasattr(pl_module, "train_vae"):
            raise AttributeError("EarlyStoppingVAE callback requires 'pl_module' to have a 'train_vae' attribute.")

        if trainer.fit_loop.epoch_loop._batches_that_stepped >= self.max_steps:
            pl_module.train_vae = False
            return

        if self._should_skip_check(trainer) or pl_module.beta > self.beta_min:
            return

        metric_tensor = trainer.callback_metrics.get(self.monitor)
        if metric_tensor is None:
            return

        current_metric: float = float(metric_tensor)

        if self.best_metric is None:
            self.best_metric = current_metric
            self.epochs_without_improvement = 0
            return

        if current_metric < self.best_metric:
            self.best_metric = current_metric
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= self.patience_epochs:
                pl_module.train_vae = False

    def _should_skip_check(self, trainer: Trainer) -> bool:
        """Determine if the callback should skip processing this validation end.

        Returns:
            True if currently in sanity check mode or not in training phase.
            False if normal validation during training.
        """
        from lightning.pytorch.trainer.states import TrainerFn

        # Skip if not in fitting phase or during sanity validation
        return trainer.state.fn != TrainerFn.FITTING or trainer.sanity_checking


class EarlyStoppingDiffusion(EarlyStopping):
    """Early stopping callback for diffusion model training phase.

    Extends Lightning's EarlyStopping to only monitor during diffusion training.
    """

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not hasattr(pl_module, "train_diffusion"):
            raise AttributeError(
                "EarlyStoppingDiffusion callback requires 'pl_module' to have a 'train_diffusion' attribute."
            )

        if not pl_module.train_diffusion:
            return

        super().on_validation_end(trainer, pl_module)
