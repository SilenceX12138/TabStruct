from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd

from ...BaseGenerator import BaseJointGenerator


class TabFORGE(BaseJointGenerator):
    """Thin TabStruct adapter around the standalone TabFORGE generator."""

    # ================================================================
    # =                                                              =
    # =                       Initialisation                         =
    # =                                                              =
    # ================================================================

    def __init__(self, args: Any) -> None:
        """Validate TabStruct requirements and retain adapter arguments."""
        super().__init__(args)
        if args.task not in {"classification", "regression", "unsupervision"}:
            raise ValueError(f"Task {args.task} is not supported by {self.name}")
        if not args.model_specific_preprocessing:
            raise ValueError(f"Model-specific preprocessing is required for {self.name}")

    # ================================================================
    # =                                                              =
    # =                       Model fitting                          =
    # =                                                              =
    # ================================================================

    def _fit(self, data_module: Any) -> None:
        """Fit TabFORGE from raw training and optional validation splits."""
        # === Prepare raw training data ===
        # TabFORGE owns feature processing and schema reconstruction.
        X_train = data_module.X_train_df.copy(deep=True)
        y_train = None
        if self.args.task != "unsupervision":
            y_train = data_module.y_train_df.iloc[:, 0].to_numpy(copy=True)

        # === Fit the standalone estimator ===
        # TabStruct opts into validation through its scheduler argument.
        validation_data = self._validation_data(data_module)
        self.model = self._build_estimator().fit(
            X_train,
            y_train,
            validation_data=validation_data,
            checkpoint=self._checkpoint_source(),
            detokeniser=self.args.model_params["detokeniser"],
        )

    def _checkpoint_source(self) -> str | None:
        """Return the checkpoint selected by the model adapter."""
        return self.args.model_params["checkpoint"]

    def _validation_data(self, data_module: Any) -> Any:
        """Return the TabStruct validation split when scheduling is requested."""
        X_valid = data_module.X_valid_df.copy(deep=True)
        if self.args.task == "unsupervision":
            return X_valid
        y_valid = data_module.y_valid_df.iloc[:, 0].to_numpy(copy=True)
        return X_valid, y_valid

    def _fit_model(self, data_df: pd.DataFrame) -> None:
        """Reject the processed-data fitting path owned by the base class."""
        raise RuntimeError("TabFORGE is fitted directly from the raw DataModule split")

    # ================================================================
    # =                                                              =
    # =                     Model generation                         =
    # =                                                              =
    # ================================================================

    def _generate_model(self, generation_config: dict[str, Any]) -> pd.DataFrame | tuple[pd.DataFrame, Any]:
        """Generate rows and restore the TabStruct supervised table layout."""
        # === Generate raw synthetic rows ===
        output = self.model.generate(
            generation_config["num_samples_to_generate"],
            random_state=self._next_generation_seed(),
        )
        if self.args.task == "unsupervision":
            return output

        # === Reassemble supervised features and target ===
        X_syn, y_syn = output
        result = (
            X_syn.copy()
            if isinstance(X_syn, pd.DataFrame)
            else pd.DataFrame(X_syn, columns=self.args.full_feature_list_processed)
        )
        result[self.args.full_target_col_processed] = y_syn
        return result

    def _update_generation_config(self, generation_config: dict[str, Any]) -> dict[str, Any]:
        """Retain the sample count consumed by the standalone generator."""
        return {"num_samples_to_generate": generation_config["num_samples_to_generate"]}

    # ================================================================
    # =                                                              =
    # =                    Model configuration                       =
    # =                                                              =
    # ================================================================

    def _build_estimator(self) -> Any:
        """Build the standalone generator from nested TabStruct parameters."""
        # Use lazy imports to keep model discovery lightweight.
        from tabforge import TabFORGEEmbeddingConfig, TabFORGEGenerator

        # === Resolve nested TabStruct parameters ===
        params = self.args.model_params
        # === Build the standalone estimator ===
        return TabFORGEGenerator(
            task=self.args.task,
            categorical_features=tuple(self.args.cat_feature_list_original),
            embedding_config=TabFORGEEmbeddingConfig(**params["embedding"]),
            architecture_config=params["architecture"],
            diffusion_config=params["diffusion"],
            training_config=self._build_training_config(params["optimization"]),
            runtime_config=self._build_runtime_config(),
            random_state=self.args.seed,
            logger=(None if self.args.disable_wandb else self.args.wandb_logger),
            n_reference_neighbors=params["n_reference_neighbors"],
            observation_masking=params["observation_masking"],
        )

    def _build_training_config(self, params: dict[str, Any]) -> Any:
        """Map TabStruct optimization parameters to standalone training."""
        from tabforge import TabFORGETrainingConfig

        values = deepcopy(params)
        values["max_steps"] = int(self.args.max_steps)
        values["batch_size"] = int(self.args.batch_size)
        return TabFORGETrainingConfig(**values)

    def _build_runtime_config(self) -> Any:
        """Map TabStruct execution arguments to the standalone runtime."""
        from tabforge import TabFORGERuntimeConfig

        runtime = self.args.model_params["runtime"]
        return TabFORGERuntimeConfig(
            device=self.args.device,
            strategy=runtime["strategy"],
            gradient_accumulation=runtime["gradient_accumulation"],
            deterministic=self.args.deterministic,
        )

    @classmethod
    def _define_default_params(cls) -> dict[str, Any]:
        """Return the standalone TabFORGE default parameter groups."""
        from tabforge import (TabFORGEArchitectureConfig, TabFORGEDiffusionConfig, TabFORGEEmbeddingConfig,
                              TabFORGETrainingConfig)

        optimization = TabFORGETrainingConfig(
            unmasked_probability=1.0,
            target_mask_probability=0.0,
            random_feature_mask_probability=0.0,
        ).to_dict()
        return {
            "checkpoint": "pretrained",
            "detokeniser": "auto",
            "embedding": TabFORGEEmbeddingConfig().to_dict(),
            "architecture": TabFORGEArchitectureConfig().to_dict(),
            "diffusion": TabFORGEDiffusionConfig().to_dict(),
            "optimization": optimization,
            "runtime": {"strategy": "auto", "gradient_accumulation": 1},
            "n_reference_neighbors": 1,
            "observation_masking": False,
        }

    @classmethod
    def _define_optuna_params(cls, trial: Any) -> dict[str, Any]:
        """Sample a small supported TabFORGE tuning space."""
        params = deepcopy(cls._define_default_params())
        params["optimization"]["decoder_lr"] = trial.suggest_categorical("decoder_lr", [0.1, 0.05])
        params["optimization"]["diffusion_lr"] = trial.suggest_categorical("diffusion_lr", [1e-3, 5e-4])
        params["n_reference_neighbors"] = trial.suggest_categorical("n_reference_neighbors", [1, 2])
        return params

    @classmethod
    def _define_single_run_params(cls) -> dict[str, Any]:
        """Reduce embedding folds for ordinary TabStruct runs."""
        params = cls._define_default_params()
        params["embedding"]["n_folds"] = 2
        return params

    @classmethod
    def _define_test_params(cls) -> dict[str, Any]:
        """Use the explicit lightweight backend for regression tests."""
        params = cls._define_default_params()
        params["checkpoint"] = None
        params["embedding"]["model_path"] = "mock"
        params["embedding"]["n_folds"] = 0
        params["optimization"]["decoder_scheduler"]["patience_steps"] = None
        params["optimization"]["diffusion_scheduler"]["patience_steps"] = None
        return params

    @classmethod
    def _get_model_specific_scaler_config(cls) -> dict[str, dict[str, Any]]:
        """Disable TabStruct preprocessing because TabFORGE owns raw schemas."""
        return {
            "context": {"disable_preprocessing": True},
            "feature_scaler": {},
            "target_scaler": {},
        }
