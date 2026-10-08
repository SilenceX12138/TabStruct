from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from tabeval.plugins.core.models.tabular_encoder import TabularEncoder

from ...BaseGenerator import BaseLightningGenerationModule, BaseLitJointGenerator
from ..ddpm.utils.modules import MLPDiffusion, UniModVAE
from .utils.callback import AdaptiveKLBeta, EarlyStoppingDiffusion, EarlyStoppingVAE
from .utils.diffusion import PowerMeanNoise
from .utils.modules import DenoiseModel


class LitTabSyn(BaseLitJointGenerator):
    def __init__(self, args):
        super().__init__(args)

        if args.task not in ["classification", "regression", "unsupervision"]:
            raise ValueError(f"Task {args.task} is not supported for {self.name} model")

        self.model = _LitTabSyn(args)

    # ================================================================
    # =                     Model-specific                           =
    # ================================================================
    def _prepare_data_scalers(self):
        self.feature_scaler = TabularEncoder(
            categorical_encoder="onehot",
            cat_encoder_params={"sparse_output": False, "handle_unknown": "ignore"},
            continuous_encoder="quantile",
            cont_encoder_params={},
        )

        self.target_scaler = None
        if self.args.task != "unsupervision":
            self.target_scaler = TabularEncoder(
                categorical_encoder="onehot",
                cat_encoder_params={"sparse_output": False, "handle_unknown": "ignore"},
                continuous_encoder="quantile",
                cont_encoder_params={},
            )

    def create_lit_trainer(self):
        # ===== Prepare callbacks =====
        callbacks = self._prepare_lit_trainer_callbacks()

        # ===== Set up trainer =====
        trainer = L.Trainer(
            # Training
            # Note: max_steps means the step that optimizer.step() is called max_steps times
            # If using multiple optimizers, it counts only once for all optimizers
            max_steps=self.args.max_steps,  # VAE training phase + Diffusion training phase
            # logging
            logger=self.args.wandb_logger,  # lightning launches multiple wandb runs in DDP, while sub-processes do not have args ---> do not affect run retrieval
            log_every_n_steps=self.args.log_every_n_steps,
            check_val_every_n_epoch=self.args.check_val_every_n_epoch,
            callbacks=callbacks,
            # miscellaneous
            accelerator=self.args.accelerator,
            detect_anomaly=self.args.debugging,
            deterministic=self.args.deterministic,
            devices=(
                "auto" if self.args.train_num_samples_processed > 100000 else 1
            ),  # use DDP only when training on large dataset
            # used for debugging, but it may crash when validation is not performed before showing results
            # fast_dev_run=True,
        )

        return trainer

    def _prepare_lit_trainer_callbacks(self):
        callback_list = super()._prepare_lit_trainer_callbacks()

        callback_list.extend(
            [
                # Following official TabSyn, the adaptive beta is scheduled every validation epoch
                AdaptiveKLBeta(
                    monitor="valid_metrics/reconstruction_loss",
                    patience_validation_epoch=10,
                ),
                EarlyStoppingVAE(
                    max_steps=self.args.max_steps * 0.5,
                    monitor="valid_metrics/reconstruction_loss",
                    beta_min=self.args.model_params["architecture"]["beta_min"],
                    patience_validation_epoch=50,
                ),
                EarlyStoppingDiffusion(
                    monitor="valid_metrics/diffusion_loss",
                    mode="min",
                    patience=50,
                ),
            ]
        )

        return callback_list

    # ================================================================
    # =                      Hyperparams                             =
    # ================================================================
    @classmethod
    def _define_default_params(cls):
        params_arch = {
            "beta_max": 1e-2,
            "beta_min": 1e-5,
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
        params_arch = {
            # VAE
            "beta_max": 1e-2,
            "beta_min": 1e-5,
            # Diffusion
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
            "lr": 1e-3,  # Learning rate (EDM is fairly robust to this)
            "weight_decay": 1e-5,  # Regularization (light, as EDM already well-conditioned)
        }
        return {"architecture": params_arch, "optimization": params_optim}

    @classmethod
    def _define_test_params(cls):
        params_arch = {
            # VAE
            "beta_max": 1e-2,
            "beta_min": 1e-5,
            # Diffusion
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
            "lr": 1e-3,  # Learning rate (EDM is fairly robust to this)
            "weight_decay": 1e-5,  # Regularization (light, as EDM already well-conditioned)
        }

        return {"architecture": params_arch, "optimization": params_optim}

    @classmethod
    def _get_model_specific_scaler_config(cls):
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


class _LitTabSyn(BaseLightningGenerationModule):
    def __init__(self, args):
        super().__init__(args)

        self.beta = self.args.model_params["architecture"]["beta_max"]

        self.automatic_optimization = False

    # ================================================================
    # =                     Model-specific                           =
    # ================================================================
    def _create_torch_model(self):
        input_dim = len(self.args.full_feature_list_model) + len(self.args.full_target_list_model)

        vae_model = UniModVAE(
            d_numerical=len(self.args.full_indices_numerical_list_model),
            categories=self.args.train_cardinality_list_model,
            num_layers=2,
        )

        score_net = MLPDiffusion(
            input_dim=vae_model.latent_dim,
            hidden_dim=1024,
        )
        denoise_model = DenoiseModel(
            denoise_fn=score_net,
            # Official TabSyn trains diffusion directly on VAE latents without extra processing
            sigma_data=1.0,
            precond=True,
            net_conditioning="sigma",  # Use log(σ)/4 as timestep embedding
        )

        model = TabSyn(
            num_features=input_dim,
            numerical_idx=self.args.full_indices_numerical_list_model,
            categorical_idx=self.args.full_indices_categorical_list_model,
            train_cardinality_list=self.args.train_cardinality_list_model,
            sigma_min=self.args.model_params["architecture"]["sigma_min"],
            sigma_max=self.args.model_params["architecture"]["sigma_max"],
            rho=self.args.model_params["architecture"]["rho"],
            num_steps=self.args.model_params["architecture"]["num_steps"],
            sigma_data=self.args.model_params["architecture"]["sigma_data"],
            p_mean=self.args.model_params["architecture"]["p_mean"],
            p_std=self.args.model_params["architecture"]["p_std"],
            enable_heun_correction=self.args.model_params["architecture"]["enable_heun_correction"],
            vae_model=vae_model,
            denoise_model=denoise_model,
        )

        return model

    def _compute_loss(self, data_real: torch.Tensor, forward_dict: dict) -> dict:
        num_loss = forward_dict["numerical_loss"]
        cat_loss = forward_dict["categorical_loss"]
        kld_loss = forward_dict["kld_loss"]

        recon_loss = num_loss + cat_loss
        vae_loss = recon_loss + self.beta * kld_loss

        if self.train_diffusion or self.fitted_diffusion:
            diffusion_loss = forward_dict["diffusion_loss"]
        else:
            diffusion_loss = torch.tensor(1e5, device=self.device)
        num_loss_denoised = forward_dict["numerical_loss_denoised"]
        cat_loss_denoised = forward_dict["categorical_loss_denoised"]

        total_loss = vae_loss + diffusion_loss
        valid_loss = torch.tensor(0.0, device=self.device)
        if self.train_vae:
            valid_loss += vae_loss
        if self.train_diffusion:
            valid_loss += diffusion_loss

        return {
            "total_loss": total_loss,  # used for checkpoint
            "valid_loss": valid_loss,  # used for optimization
            "vae_loss": vae_loss,
            "diffusion_loss": diffusion_loss,
            "reconstruction_loss": recon_loss,
            "numerical_loss": num_loss,
            "categorical_loss": cat_loss,
            "kld_loss": kld_loss,
            "numerical_loss_denoised": num_loss_denoised,
            "categorical_loss_denoised": cat_loss_denoised,
        }

    def _generate(self, num_samples: int) -> torch.Tensor:
        return self.torch_model.sample(num_samples, device=self.device)

    # ================================================================
    # =                     Model training                           =
    # ================================================================
    def training_step(self, batch):
        total_loss = super().training_step(batch)
        loss_dict = self.training_step_output_list[-1]["loss_dict"]

        self.manual_backward(loss_dict["valid_loss"])

        optimizer_vae, optimizer_diffusion = self.optimizers()
        if self.train_vae:
            optimizer_vae.step()
            optimizer_vae.zero_grad()

            self.clip_gradients(optimizer_vae, gradient_clip_val=self.args.gradient_clip_val)

        if self.train_diffusion:
            optimizer_diffusion.step()
            optimizer_diffusion.zero_grad()

            self.clip_gradients(optimizer_diffusion, gradient_clip_val=self.args.gradient_clip_val)

        return total_loss

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()

        lr_scheduler_vae, lr_scheduler_diffusion = self.lr_schedulers()

        if self.train_vae:
            lr_scheduler_vae.step(self.trainer.callback_metrics["valid_metrics/vae_loss"])
        if self.train_diffusion:
            lr_scheduler_diffusion.step(self.trainer.callback_metrics["valid_metrics/diffusion_loss"])

    def configure_optimizers(self):
        optim_vae = self._optimizer_handler("adam", self.torch_model.vae_model.parameters())
        optim_diffusion = self._optimizer_handler("adam", self.torch_model.denoise_model.parameters())

        lr_scheduler_vae = self._lr_scheduler_handler("plateau", optimizer=optim_vae)
        lr_scheduler_diffusion = self._lr_scheduler_handler("plateau", optimizer=optim_diffusion)

        return [optim_vae, optim_diffusion], [lr_scheduler_vae, lr_scheduler_diffusion]

    @property
    def train_vae(self) -> bool:
        return self.torch_model.train_vae

    @train_vae.setter
    def train_vae(self, value: bool):
        self.torch_model.train_vae = value

        # If VAE training is turned off, enable diffusion training automatically
        if not value:
            self.train_diffusion = True
            self.save_train_embedding(self.trainer.datamodule)

    @property
    def train_diffusion(self) -> bool:
        return self.torch_model.train_diffusion

    @train_diffusion.setter
    def train_diffusion(self, value: bool):
        self.torch_model.train_diffusion = value

        if value:
            self.torch_model.fitted_diffusion = True

    @property
    def fitted_diffusion(self) -> bool:
        return self.torch_model.fitted_diffusion

    # ================================================================
    # =                                                              =
    # =                          Utils                               =
    # =                                                              =
    # ================================================================
    @torch.no_grad()
    def save_train_embedding(self, data_module):
        train_embedding_list = []

        # === Iterate over the training data loader to get embeddings ===
        for X_batch, y_batch, _ in data_module.train_dataloader():
            # Prepare the data batch
            data_batch = X_batch.to(self.device)
            if self.args.task != "unsupervision":
                y_batch = y_batch.unsqueeze(1) if len(y_batch.shape) == 1 else y_batch
                data_batch = torch.cat([data_batch, y_batch.to(self.device)], dim=1)

            # Get the embedding
            embedding_temp = self.torch_model.get_embedding(data_batch).detach()
            train_embedding_list.append(embedding_temp)

        train_embedding = torch.concat(train_embedding_list, dim=0)
        train_embedding = torch.tensor(train_embedding, device=self.device, requires_grad=False)
        self.torch_model.train_embedding = train_embedding
        self.torch_model.train_embedding_mean = torch.mean(train_embedding, dim=0)
        self.torch_model.train_embedding_std = torch.std(train_embedding, dim=0)


class TabSyn(nn.Module):
    def __init__(
        self,
        num_features: int,
        numerical_idx: list[int],
        categorical_idx: list[int],
        train_cardinality_list: list[int],
        sigma_min: float,
        sigma_max: float,
        rho: float,
        num_steps: int,
        sigma_data: float,
        p_mean: float,
        p_std: float,
        enable_heun_correction: bool,
        vae_model: UniModVAE,
        denoise_model: DenoiseModel,
    ):
        super().__init__()

        # === Data layout metadata ===
        self.num_features = int(num_features)
        self.numerical_idx = list(numerical_idx)
        self.categorical_idx = list(categorical_idx)
        self.num_numerical = len(self.numerical_idx)
        self.num_categorical = len(self.categorical_idx)
        self.train_cardinality_list = [int(c) for c in train_cardinality_list]
        self.enable_heun_correction = enable_heun_correction
        self.latent_dim = vae_model.latent_dim

        self.vae_model = vae_model
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
        self.sigma_schedule = PowerMeanNoise(self.sigma_min, self.sigma_max, self.rho)

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

        # Start with VAE training phase, later switch to diffusion
        self._train_vae = True
        self._train_diffusion = False
        self._fitted_diffusion = False

        train_embedding = torch.zeros(1, self.latent_dim)
        train_embedding_mean = torch.zeros(self.latent_dim)
        train_embedding_std = torch.ones(self.latent_dim)
        self.register_buffer("train_embedding", train_embedding)
        self.register_buffer("train_embedding_mean", train_embedding_mean)
        self.register_buffer("train_embedding_std", train_embedding_std)

    # =============================
    # Forward diffusion / training
    # =============================
    def forward(self, x0: torch.Tensor) -> dict:
        batch_size = x0.shape[0]
        device = x0.device

        x_num = self._extract_numerical(x0)
        x_cat = self._extract_categorical(x0)

        recon_dict = self.vae_model(x_num, x_cat)
        x_num_recon, x_cat_recon = recon_dict["x_num_recon"], recon_dict["x_cat_recon"]
        z_mu, z_log_var = recon_dict["z_mu"], recon_dict["z_log_var"]

        num_loss = self._numerical_loss(x_num_recon, x_num)
        cat_loss = self._categorical_loss(x_cat_recon, x_cat)
        kld_loss = self._kld_loss(z_mu, z_log_var)

        z_mu = self._preprocess_latent_variables(z_mu)
        sigma_z = self._build_forward_sigma(batch_size, device)
        # Following official TabSyn, diffusion is trained on VAE latents (mean/mu) directly
        z_t = self._apply_numerical_noise(z_mu, sigma_z)
        z_denoised = self._denoise(z_t, sigma_z)

        diffusion_loss = self._diffusion_loss(z_denoised, z_mu, sigma_z)

        # Compute the reconstruction loss using denoised latents instead of clean latents during diffusion training
        recon_dict_denoised = self.vae_model.decode(self.vae_model._reparameterize(z_denoised, z_log_var))
        x_num_recon_denoised, x_cat_recon_denoised = (
            recon_dict_denoised["x_num_recon"],
            recon_dict_denoised["x_cat_recon"],
        )
        num_loss_denoised = self._numerical_loss(x_num_recon_denoised, x_num)
        cat_loss_denoised = self._categorical_loss(x_cat_recon_denoised, x_cat)

        return {
            "numerical_loss": num_loss,
            "categorical_loss": cat_loss,
            "kld_loss": kld_loss,
            "diffusion_loss": diffusion_loss,
            "numerical_loss_denoised": num_loss_denoised,
            "categorical_loss_denoised": cat_loss_denoised,
        }

    # =============================
    # Sampling
    # =============================
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        sigma_z = self._build_sampling_sigma(device)

        z = self._init_latent_variables(num_samples, sigma_z, device)

        if self._fitted_diffusion:
            z = self._sample_diffusion(z, sigma_z)
            z = self._recover_latent_variables(z)

        samples = self._sample_vae(z, device)

        return samples

    # =============================
    # VAE utilities
    # =============================
    def _extract_numerical(self, x: torch.Tensor) -> torch.Tensor | None:
        x_num = None
        if self.num_numerical > 0:
            x_num = x[:, self.numerical_idx]

        return x_num

    def _extract_categorical(self, x: torch.Tensor) -> torch.Tensor | None:
        x_cat_ordinal = None
        if self.num_categorical > 0:
            # Convert to float for compatibility with torch operators
            x_cat_ordinal = x[:, self.categorical_idx].float()

        return x_cat_ordinal

    def _numerical_loss(
        self,
        x_num_syn: torch.Tensor | None,
        x_num_real: torch.Tensor | None,
    ) -> torch.Tensor:
        loss = torch.zeros(1, device=x_num_real.device)

        if self.num_numerical > 0:
            mse = F.mse_loss(x_num_syn, x_num_real, reduction="sum")
            loss = mse / x_num_real.shape[0]

        return loss

    def _categorical_loss(
        self,
        x_cat_onehot_syn: torch.Tensor | None,
        x_cat_onehot_real: torch.Tensor | None,
    ) -> torch.Tensor:
        loss = torch.zeros(1, device=x_cat_onehot_real.device)

        if self.num_categorical > 0:
            start_idx = 0
            total_cat_loss = 0.0
            for i, card in enumerate(self.train_cardinality_list):
                num_classes = card
                end_idx = start_idx + num_classes

                logits = x_cat_onehot_syn[:, start_idx:end_idx]
                target = x_cat_onehot_real[:, start_idx:end_idx]

                ce_loss = F.cross_entropy(logits, target, reduction="sum")
                total_cat_loss += ce_loss

                start_idx = end_idx

            loss = total_cat_loss / x_cat_onehot_real.shape[0]

        return loss

    def _kld_loss(
        self,
        z_mu: torch.Tensor,
        z_log_var: torch.Tensor,
    ) -> torch.Tensor:
        kld_loss = -0.5 * torch.sum(1 + z_log_var - z_mu.pow(2) - z_log_var.exp())

        # Average over batch
        kld_loss = kld_loss / z_mu.shape[0]

        return kld_loss

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        x_num = self._extract_numerical(x)
        x_cat = self._extract_categorical(x)

        # Following official TabSyn, use VAE mean (mu) as embedding
        recon_dict = self.vae_model(x_num, x_cat)
        z_mu = recon_dict["z_mu"]

        return z_mu

    def _sample_vae(self, z, device: torch.device) -> torch.Tensor:
        recon_dict = self.vae_model.decode(z)
        x_num = recon_dict["x_num_recon"]
        x_cat = recon_dict["x_cat_recon"]

        samples = torch.zeros(z.shape[0], self.num_features, device=device)
        if self.num_numerical > 0:
            samples[:, self.numerical_idx] = x_num
        if self.num_categorical > 0:
            samples[:, self.categorical_idx] = x_cat.float()

        return samples

    # =============================
    # Diffusion utilities
    # =============================
    def _build_forward_sigma(self, batch_size: int, device: torch.device) -> dict:
        eps = torch.randn(batch_size, 1, device=device)

        sigma = (self.noise_dist_params["P_mean"] + eps * self.noise_dist_params["P_std"]).exp()
        sigma = sigma.clamp(min=self.sigma_min, max=self.sigma_max)

        return sigma

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

    def _denoise(
        self,
        z_t: torch.Tensor | None,
        sigma: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        sigma_expanded = sigma.view(-1, 1)

        z_denoised, _ = self.denoise_model(x_num=z_t, x_cat=None, t=None, sigma=sigma_expanded)

        return z_denoised

    def _diffusion_loss(
        self,
        z_syn: torch.Tensor | None,
        z_real: torch.Tensor | None,
        sigma_z: torch.Tensor,
    ) -> torch.Tensor:
        loss = torch.zeros(1, device=z_real.device)

        sigma2 = sigma_z**2  # σ²
        sigma_data2 = self.edm_params["sigma_data"] ** 2  # σ_data²

        weight = (sigma2 + sigma_data2) / torch.clamp((sigma_z * self.edm_params["sigma_data"]) ** 2, min=1e-6)
        mse = F.mse_loss(z_syn, z_real, reduction="none")
        loss = (weight * mse).sum() / z_real.shape[0]

        return loss

    def _sample_diffusion(self, z: torch.Tensor, sigma_z: torch.Tensor) -> torch.Tensor:

        total_reverse_steps = len(sigma_z) - 1
        for i in range(total_reverse_steps):
            sigma_z_cur, sigma_z_next = sigma_z[i], sigma_z[i + 1]

            sigma_z_hat = self._build_sampling_sigma_churn(sigma_z_cur, total_reverse_steps)

            z = self._sample_step(z, sigma_z_cur, sigma_z_next, sigma_z_hat)

        return z

    # =============================
    # Sampling utilities
    # =============================
    def _preprocess_latent_variables(self, z: torch.Tensor) -> torch.Tensor:
        # Hyperparam from official TabSyn codebase
        z = (z - self.train_embedding_mean) / self.train_embedding_std

        return z

    def _recover_latent_variables(self, z: torch.Tensor) -> torch.Tensor:
        # Hyperparam from official TabSyn codebase
        z = z * self.train_embedding_std + self.train_embedding_mean

        return z

    def _init_latent_variables(
        self,
        num_samples: int,
        sigma_z: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        z = torch.randn(num_samples, self.latent_dim, device=device) * sigma_z[0]

        return z

    def _build_sampling_sigma(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.num_steps < 1:
            raise ValueError("num_steps must be a positive integer")

        # Linear ramp in time [1, 0] for reverse denoising
        ramp = torch.linspace(1, 0, self.num_steps, device=device)

        # Compute numerical noise levels via Karras schedule
        sigma_z = self.sigma_schedule.total_noise(ramp).reshape(*ramp.shape)

        # Append σ = 0 as final target (clean data)
        sigma_z = torch.cat([sigma_z, torch.zeros(1, device=device)])

        return sigma_z

    def _build_sampling_sigma_churn(
        self,
        sigma_z_cur: torch.Tensor,
        total_reverse_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sigma_z_hat = sigma_z_cur

        gamma = 0.0
        if (
            self.sampler_params["sigma_churn"] > 0
            and self.sampler_params["sigma_churn_min"] <= sigma_z_cur <= self.sampler_params["sigma_churn_max"]
        ):
            gamma = min(self.sampler_params["sigma_churn"] / total_reverse_steps, torch.sqrt(torch.tensor(2.0)) - 1.0)

            sigma_z_hat = sigma_z_cur * (1.0 + gamma)

        return sigma_z_hat

    def _sample_step(
        self,
        z_cur: torch.Tensor | None,
        sigma_z_cur: torch.Tensor,
        sigma_z_next: torch.Tensor,
        sigma_z_hat: torch.Tensor,
    ):
        # ============================================================
        # Step A: Churn injection (stochastic noise perturbation)
        # ============================================================
        z_hat = self._churn_injection(z_cur, sigma_z_cur, sigma_z_hat)

        # ============================================================
        # Step B: Denoise at perturbed state σ̂
        # ============================================================
        z_denoised = self._denoise(z_hat, sigma_z_hat)

        # ============================================================
        # Step C: Numerical update via Euler ODE integration
        # ============================================================
        z_next, d_cur = self._euler_step(z_hat, z_denoised, sigma_z_next, sigma_z_hat)

        # ============================================================
        # Step E: Second-order correction (Heun's method)
        # ============================================================
        if self.enable_heun_correction:
            # Refine numerical update with second-order correction
            # Improves accuracy of ODE integration step
            z_next = self._correction_step(z_hat, z_next, sigma_z_next, sigma_z_hat, d_cur)

        return z_next

    def _churn_injection(self, z_cur, sigma_z_cur, sigma_z_hat):
        z_hat = z_cur
        if z_hat is not None:
            noise_scale = torch.sqrt(torch.clamp(sigma_z_hat**2 - sigma_z_cur**2, min=0.0))
            if torch.any(noise_scale > 0):
                z_hat = z_hat + noise_scale * torch.randn_like(z_hat)

        return z_hat

    def _euler_step(
        self,
        z_hat: torch.Tensor | None,
        z_denoised: torch.Tensor | None,
        sigma_z_next: torch.Tensor,
        sigma_z_hat: torch.Tensor,
    ):
        # Numerical follows probability flow ODE: dx/dσ = (x - D(x,σ))/σ
        z_next = z_hat
        d_cur = None
        if z_hat is not None:
            # Compute derivative (gradient of log density)
            sigma_hat_safe = torch.clamp(sigma_z_hat, min=1e-8)
            d_cur = (z_hat - z_denoised) / sigma_hat_safe

            # Euler integration step: x_next = x̂ + Δσ * dx/dσ
            z_next = z_hat + (sigma_z_next - sigma_z_hat) * d_cur

        return z_next, d_cur

    def _correction_step(self, z_hat, z_next, sigma_z_next, sigma_z_hat, d_cur):
        if z_next is not None and d_cur is not None:
            # Re-evaluate network at predicted state x_next
            denoised_prime = self._denoise(z_next, sigma_z_next)

            # Compute derivative at predicted point
            sigma_next_safe = torch.clamp(sigma_z_next, min=1e-8)
            d_prime = (z_next - denoised_prime) / sigma_next_safe

            # Corrector step: use average of both derivatives for higher accuracy
            # This gives O(Δσ³) truncation error vs O(Δσ²) for Euler
            z_next = z_hat + 0.5 * (d_cur + d_prime) * (sigma_z_next - sigma_z_hat)

        return z_next

    @property
    def train_vae(self) -> bool:
        return self._train_vae

    @train_vae.setter
    def train_vae(self, value: bool):
        self._train_vae = value

        if value:
            self.vae_model.train()
        else:
            self.vae_model.eval()

    @property
    def train_diffusion(self) -> bool:
        return self._train_diffusion

    @train_diffusion.setter
    def train_diffusion(self, value: bool):
        self._train_diffusion = value

        if value:
            self.denoise_model.train()
        else:
            self.denoise_model.eval()

    @property
    def fitted_diffusion(self) -> bool:
        return self._fitted_diffusion

    @fitted_diffusion.setter
    def fitted_diffusion(self, value: bool):
        self._fitted_diffusion = value
