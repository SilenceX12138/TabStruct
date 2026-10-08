from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from ..BasePredictor import BaseSklearnPredictor


class TabFORGE(BaseSklearnPredictor):
    """Thin prediction adapter around the standalone TabFORGE estimators."""

    def __init__(self, args: Any) -> None:
        super().__init__(args)
        if args.task not in {"classification", "regression"}:
            raise ValueError(f"Task {args.task} is not supported by {self.name}")
        if not args.model_specific_preprocessing:
            raise ValueError(f"Model-specific preprocessing is required for {self.name}")

    def _fit(self, data_module: Any) -> None:
        """Fit TabFORGE from raw feature frames and the validation split."""
        X_train = data_module.X_train_df.copy(deep=True)
        y_train = data_module.y_train_df.iloc[:, 0].to_numpy(copy=True)
        X_valid = data_module.X_valid_df.copy(deep=True)
        y_valid = data_module.y_valid_df.iloc[:, 0].to_numpy(copy=True)
        self.model = self._build_estimator().fit(
            X_train,
            y_train,
            validation_data=(X_valid, y_valid),
            checkpoint=self._checkpoint_source(),
            detokeniser=self.params["detokeniser"],
        )
        if self.args.task == "classification":
            self.class_to_index_ = {value: index for index, value in enumerate(self.model.classes_)}
            self._encode_data_module_targets(data_module)

    def _checkpoint_source(self) -> str | None:
        """Return the checkpoint selected by the model adapter."""
        return self.params["checkpoint"]

    def _predict(self, X: Any) -> np.ndarray:
        """Translate standalone class labels to TabStruct's integer contract."""
        values = self.model.predict(X)
        if self.args.task == "classification":
            return self.encode_target(values)
        return values

    def encode_target(self, values: Any) -> np.ndarray:
        """Encode raw class labels with the fitted standalone class order."""
        return np.array([self.class_to_index_[value] for value in values])

    def _encode_data_module_targets(self, data_module: Any) -> None:
        """Expose classifier targets through TabStruct's integer metric contract."""
        encoded = {}
        for split in ("train", "valid", "test"):
            values = getattr(data_module, f"y_{split}_df").iloc[:, 0].to_numpy(copy=True)
            encoded[split] = self.encode_target(values)
            setattr(data_module, f"y_{split}_df", encoded[split])
        classes = list(range(len(self.model.classes_)))
        counts = np.bincount(encoded["train"], minlength=len(classes))
        weights = len(encoded["train"]) / (len(classes) * counts)
        self.args.class_encoded_list = classes
        self.args.encoded2class = dict(enumerate(self.model.classes_))
        self.args.train_class_weight_list = weights
        self.args.train_class2weight = dict(zip(classes, weights))

    def _build_estimator(self) -> Any:
        """Translate prediction arguments into the standalone API."""
        from tabforge import (TabFORGEClassifier, TabFORGEDiffusionConfig, TabFORGEEmbeddingConfig, TabFORGERegressor,
                              TabFORGERuntimeConfig, TabFORGETrainingConfig)

        params = self.params
        estimator_type = TabFORGEClassifier if self.args.task == "classification" else TabFORGERegressor
        training = deepcopy(params["optimization"])
        runtime = params["runtime"]
        return estimator_type(
            categorical_features=tuple(self.args.cat_feature_list_original),
            embedding_config=TabFORGEEmbeddingConfig(**params["embedding"]),
            architecture_config=params["architecture"],
            diffusion_config=TabFORGEDiffusionConfig(**params["diffusion"]),
            training_config=TabFORGETrainingConfig(
                max_steps=int(self.args.max_steps),
                batch_size=int(self.args.batch_size),
                **{key: value for key, value in training.items() if key not in {"max_steps", "batch_size"}},
            ),
            runtime_config=TabFORGERuntimeConfig(
                device=self.args.device,
                strategy=runtime["strategy"],
                gradient_accumulation=runtime["gradient_accumulation"],
                deterministic=self.args.deterministic,
            ),
            random_state=self.args.seed,
            logger=(None if self.args.disable_wandb else self.args.wandb_logger),
            n_prediction_samples=params["n_prediction_samples"],
        )

    @classmethod
    def _define_default_params(cls) -> dict[str, Any]:
        """Return prediction defaults sourced from the standalone configs."""
        from tabforge import (TabFORGEArchitectureConfig, TabFORGEDiffusionConfig, TabFORGEEmbeddingConfig,
                              TabFORGETrainingConfig)

        optimization = TabFORGETrainingConfig(
            unmasked_probability=0.0,
            target_mask_probability=1.0,
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
            "n_prediction_samples": 32,
        }

    @classmethod
    def _define_single_run_params(cls) -> dict[str, Any]:
        params = deepcopy(cls._define_default_params())
        params["embedding"]["n_folds"] = 2
        return params

    @classmethod
    def _define_test_params(cls) -> dict[str, Any]:
        params = cls._define_default_params()
        params["checkpoint"] = None
        params["embedding"]["model_path"] = "mock"
        params["embedding"]["n_folds"] = 0
        params["optimization"]["decoder_scheduler"]["patience_steps"] = None
        params["optimization"]["diffusion_scheduler"]["patience_steps"] = None
        return params

    @classmethod
    def _define_optuna_params(cls, trial: Any) -> dict[str, Any]:
        """Sample a small supported TabFORGE prediction search space."""
        params = deepcopy(cls._define_default_params())
        params["optimization"]["decoder_lr"] = trial.suggest_categorical("decoder_lr", [0.1, 0.05])
        params["optimization"]["diffusion_lr"] = trial.suggest_categorical("diffusion_lr", [1e-3, 5e-4])
        params["n_prediction_samples"] = trial.suggest_categorical("n_prediction_samples", [16, 32])
        return params

    @classmethod
    def _get_model_specific_scaler_config(cls) -> dict[str, dict[str, Any]]:
        return {
            "context": {"disable_preprocessing": True},
            "feature_scaler": {},
            "target_scaler": {},
        }
