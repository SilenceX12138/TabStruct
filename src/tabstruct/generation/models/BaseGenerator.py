import copy
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
import torch
from tabeval.plugins.core.models.tabular_encoder import TabularEncoder
from tqdm.auto import tqdm

from src.tabstruct.common.data.DataHelper import DataHelper
from src.tabstruct.common.data.DataModule import DataModule
from src.tabstruct.common.model.BaseModel import BaseLightningModule, BaseModel, LitModelMixin


class BaseGenerator(BaseModel, metaclass=ABCMeta):
    """Base class for all generators."""

    def __init__(self, args):
        super().__init__(args)

    # ================================================================
    # =                                                              =
    # =                      Model fitting                           =
    # =                                                              =
    # ================================================================
    def _fit(self, data_module: DataModule):
        # === Prepare the training data ===
        data_df = pd.DataFrame(data_module.X_train, columns=self.args.full_feature_list_processed)
        if data_module.y_train is not None:
            data_df[self.args.full_target_col_processed] = data_module.y_train

        # === Prepare the condition ===
        cond_dict = self._prepare_cond(data_df)
        self.cond_for_fit = cond_dict["cond_for_fit"]
        self.cond_for_generation = cond_dict["cond_for_generation"]

        # === Fit the model ===
        self._fit_model(data_df)

    # ================================================================
    # =                                                              =
    # =                   Model generation                           =
    # =                                                              =
    # ================================================================
    def generate(self) -> pd.DataFrame:
        """Generate synthetic samples."""
        # Each top-level generation call should be reproducible while successive
        # memory-bounded batches receive distinct random streams.
        self._generation_seed_offset = 0

        # === Compute the generation configurations ===
        # Compute the number of synthetic samples for each class
        class2synthetic_samples = self.compute_class2synthetic_samples()

        # === Generate synthetic samples ===
        data_syn_df_raw = self._generate(class2synthetic_samples)

        return data_syn_df_raw

    def _next_generation_seed(self) -> int | None:
        """Return a deterministic seed unique to the next generation batch."""
        seed = getattr(self.args, "seed", None)
        if seed is None:
            return None

        offset = getattr(self, "_generation_seed_offset", 0)
        self._generation_seed_offset = offset + 1
        return int(seed) + offset

    def compute_class2synthetic_samples(self) -> dict:
        """Compute the number of synthetic samples to generate for each class."""
        # === Get the total number of samples to generate ===
        num_synthetic_samples_total = self.args.generation_num_samples
        if num_synthetic_samples_total is None:
            # If the total number of samples to generate is not provided, use the original training size
            num_synthetic_samples_total = int(self.args.train_num_samples_processed * self.args.generation_ratio)

        # === Get the class distribution to generate ===
        class2synthetic_distribution = self.compute_class2synthetic_distribution()

        # === Get the number of samples per class to generate ===
        class2synthetic_samples = {
            class_id: int(num_synthetic_samples_total * distribution)
            for class_id, distribution in class2synthetic_distribution.items()
        }

        # === align the number of samples per class to the total number of samples to generate ===
        class2synthetic_samples = self.align_class2samples(class2synthetic_samples, num_synthetic_samples_total)

        return class2synthetic_samples

    def compute_class2synthetic_distribution(self) -> dict:
        """Compute the distribution of synthetic samples to generate for each class.

        Raises:
            ValueError: If the generation mode is not supported

        Returns:
            dict: The distribution of synthetic samples to generate for each class
        """
        if self.args.task in ["regression", "unsupervision"]:
            class2synthetic_distribution = {
                "real": 1,  # class for Real samples
                "dummy": 0,  # Dummy class to skip generation
            }
        else:
            if self.args.generation_mode == "stratified":
                class2synthetic_distribution = self.args.train_class2distribution_processed
            elif self.args.generation_mode == "uniform":
                class2synthetic_distribution = {
                    class_id: 1 / self.args.full_num_classes_processed
                    for class_id in self.args.train_class2distribution_processed.keys()
                }
            else:
                raise ValueError(f"Generation mode {self.args.generation_mode} is not supported")

        return class2synthetic_distribution

    def align_class2samples(self, class2samples: dict, num_samples_total: int) -> dict:
        """Align the number of samples to generate for each class to the total number of samples to generate.

        Args:
            class2samples (dict): Class ID to the number of samples to generate
            num_samples_total (int): The total number of samples to generate

        Raises:
            ValueError: If the number of synthetic samples to generate does not match the total number of samples

        Returns:
            dict: The aligned number of samples to generate for each class
        """
        tol = 0
        while sum(class2samples.values()) != num_samples_total and tol < 100:
            # Pick a class randomly
            class_to_modify = np.random.choice(list(class2samples.keys()))
            # Increase or decrease the number of samples to generate for the class
            class2samples[class_to_modify] -= np.sign(sum(class2samples.values()) - num_samples_total)
            # Ensure that the number of samples to generate is positive
            class2samples[class_to_modify] = max(class2samples[class_to_modify], 1)
            # Update the tolerance
            tol += 1

        if sum(class2samples.values()) != num_samples_total:
            raise ValueError("The number of synthetic samples to generate does not match the total number of samples")

        return class2samples

    # ================================================================
    # =                                                              =
    # =              Utils to implement in sub class                 =
    # =                                                              =
    # ================================================================
    @abstractmethod
    def _fit_model(self, data: pd.DataFrame | DataModule):
        raise NotImplementedError("This method has to be implemented by the sub class")

    @abstractmethod
    def _prepare_cond(self, data_df: pd.DataFrame):
        raise NotImplementedError("This method has to be implemented by the sub class")

    @abstractmethod
    def _generate(self, class2synthetic_samples: dict) -> pd.DataFrame:
        raise NotImplementedError("This method has to be implemented by the sub class")

    @abstractmethod
    def _generate_model(self, generation_config: dict) -> dict | pd.DataFrame:
        """Generate samples with original APIs provided by the model."""
        raise NotImplementedError("This method has to be implemented by the sub class")

    @abstractmethod
    def _update_generation_config(self, generation_config: dict) -> dict:
        raise NotImplementedError("This method has to be implemented by the sub class")


# ================================================================
# =                                                              =
# =                    Generator Strategy Mixins                 =
# =                                                              =
# ================================================================
class JointGenerationMixin:
    """Mixin for joint generation strategy."""

    def _prepare_cond(self, data_df):
        cond_for_fit = None
        cond_for_generation = None

        return {
            "cond_for_fit": cond_for_fit,
            "cond_for_generation": cond_for_generation,
        }

    def _generate(self, class2synthetic_samples):
        # Update the dict every generation
        class2generated_samples = {}

        # === Generate synthetic samples in a GPU-memory-efficient way ===
        data_syn_temp_list = []
        num_synthetic_samples = sum(class2synthetic_samples.values())
        num_current_synthetic_samples = 0
        patience = 0
        max_patience = num_synthetic_samples // 1000 + 10
        progress_bar = tqdm(
            total=num_synthetic_samples,
            desc="Generating synthetic samples",
            unit="samples",
            leave=True,
        )
        try:
            while num_current_synthetic_samples < num_synthetic_samples and patience < max_patience:
                num_samples_to_generate = num_synthetic_samples - num_current_synthetic_samples
                num_samples_to_generate = min(num_samples_to_generate, 1000)

                generation_config = self._update_generation_config(
                    {
                        "num_samples_to_generate": num_samples_to_generate,
                    }
                )
                data_syn_temp_df = self._generate_model(generation_config)
                filter_dict = self._filter_generated_data(
                    data_syn_temp_df,
                    class2synthetic_samples,
                    class2generated_samples,
                )
                class2generated_samples.update(filter_dict["class2generated_samples"])
                data_syn_temp_list.append(filter_dict["data_syn_filtered"])

                generated_batch_size = data_syn_temp_list[-1].shape[0]
                num_current_synthetic_samples += generated_batch_size
                progress_bar.update(generated_batch_size)
                patience += 1
        finally:
            # Ensure the progress bar is cleared even if we break early
            if progress_bar.n < progress_bar.total:
                progress_bar.update(progress_bar.total - progress_bar.n)
            progress_bar.close()

        # Concatenate all generated samples
        data_syn_df = pd.concat(data_syn_temp_list, axis=0, ignore_index=True)

        # Upsample if unenough samples were generated
        if data_syn_df.shape[0] < num_synthetic_samples:
            data_syn_df = data_syn_df.sample(
                num_synthetic_samples,
                replace=True,
                random_state=self.args.seed,
            ).reset_index(drop=True)

        return data_syn_df

    def _filter_generated_data(self, data_syn_df, class2synthetic_samples, class2generated_samples):
        # For non-classification tasks, return all data without filtering
        if self.args.task != "classification":
            return {
                "data_syn_filtered": data_syn_df,
                "class2generated_samples": {},
            }

        filtered_df_list = []

        # Process each class separately
        for class_id, class_df in data_syn_df.groupby(self.args.full_target_col_processed):
            # Ensure class exists in generated samples counter
            current_generated = class2generated_samples.get(class_id, 0)

            # Calculate how many more samples we need for this class
            samples_needed = class2synthetic_samples[class_id] - current_generated

            if samples_needed > 0:
                # Take only what we need (or what's available)
                samples_to_keep = min(len(class_df), samples_needed)
                filtered_df_list.append(class_df.iloc[:samples_to_keep])

                # Update the counter
                class2generated_samples[class_id] = current_generated + samples_to_keep

        # Combine all filtered data or return empty DataFrame
        data_syn_filtered_df = (
            pd.concat(filtered_df_list, ignore_index=True)
            if filtered_df_list
            else pd.DataFrame(columns=data_syn_df.columns)
        )

        return {
            "data_syn_filtered": data_syn_filtered_df,
            "class2generated_samples": class2generated_samples,
        }


class ConditionalGenerationMixin:
    """Mixin for conditional generation strategy."""

    def _prepare_cond(self, data_df):
        cond_for_fit = None
        cond_for_generation = None

        if self.args.task != "unsupervision":
            cond_for_fit = data_df[self.args.full_target_col_processed]
            cond_for_generation = data_df[self.args.full_target_col_processed]

        return {
            "cond_for_fit": cond_for_fit,
            "cond_for_generation": cond_for_generation,
        }

    def _generate(self, class2synthetic_samples):
        # === Prepare lists to hold synthetic samples ===
        data_syn_class_df_list = []
        total_synthetic_samples = sum(class2synthetic_samples.values())
        progress_bar = tqdm(
            total=total_synthetic_samples,
            desc="Generating synthetic samples",
            unit="samples",
            leave=True,
        )
        try:
            # === Generate synthetic samples for each class ===
            for class_id, num_synthetic_samples in class2synthetic_samples.items():
                data_syn_temp_list = []

                # === Generate synthetic samples for the current class in a GPU-memory-efficient way ===
                num_current_synthetic_samples = 0
                patience = 0
                max_patience = num_synthetic_samples // 1000 + 10
                while num_current_synthetic_samples < num_synthetic_samples and patience < max_patience:
                    num_samples_to_generate = num_synthetic_samples - num_current_synthetic_samples
                    num_samples_to_generate = min(num_samples_to_generate, 1000)

                    generation_config = self._update_generation_config(
                        {
                            "class_id": class_id,
                            "num_samples_to_generate": num_samples_to_generate,
                        }
                    )
                    data_syn_temp_df = self._generate_model(generation_config)
                    data_syn_temp_list.append(data_syn_temp_df)

                    batch_size = len(data_syn_temp_df)
                    num_current_synthetic_samples += batch_size
                    progress_bar.update(batch_size)
                    patience += 1

                # Only consider classes with >0 samples, as empty arrays cannot be concatenated
                if len(data_syn_temp_list) == 0:
                    continue

                # Concatenate all generated samples for the current class
                data_syn_class_df = pd.concat(data_syn_temp_list, axis=0, ignore_index=True)

                # Upsample if unenough samples were generated
                if data_syn_class_df.shape[0] < num_synthetic_samples:
                    data_syn_class_df = data_syn_class_df.sample(
                        num_synthetic_samples,
                        replace=True,
                        random_state=self.args.seed,
                    ).reset_index(drop=True)

                # Append to the list of all synthetic samples
                data_syn_class_df_list.append(data_syn_class_df)
        finally:
            if progress_bar.n < progress_bar.total:
                progress_bar.update(progress_bar.total - progress_bar.n)
            progress_bar.close()

        # Concatenate all synthetic samples
        data_syn_df = pd.concat(data_syn_class_df_list, axis=0, ignore_index=True)

        return data_syn_df


class ClassFocusedGenerationMixin:
    """Mixin for joint generation strategy."""

    def _prepare_cond(self, data_df):
        cond_for_fit = None
        cond_for_generation = None

        return {
            "cond_for_fit": cond_for_fit,
            "cond_for_generation": cond_for_generation,
        }

    def _fit(self, data_module):
        # === Save the data ===
        if self.args.task == "classification":
            self.X_train = data_module.X_train
            self.y_train = data_module.y_train
        elif self.args.task == "regression":
            self.X_train = np.concatenate([data_module.X_train, data_module.y_train.reshape(-1, 1)], axis=1)
            # The samples are sorted by class id (0->real data, 1->dummy data)
            self.y_train = np.zeros(self.X_train.shape[0], dtype=np.int64)
        elif self.args.task == "unsupervision":
            self.X_train = data_module.X_train
            # The samples are sorted by class id (0->real data, 1->dummy data)
            self.y_train = np.zeros(self.X_train.shape[0], dtype=np.int64)

        self._fit_model(data_module)

    def _generate(self, class2synthetic_samples):
        # Generate synthetic samples with original APIs provided by the model
        synthetic_data_dict = self._generate_model(class2synthetic_samples)

        # Create a dataframe from the synthetic data
        data_syn_df = pd.DataFrame(synthetic_data_dict["X_syn"], columns=self.args.full_feature_list_processed)
        if self.args.task != "unsupervision":
            data_syn_df[self.args.full_target_col_processed] = synthetic_data_dict["y_syn"]

        return data_syn_df

    def _update_generation_config(self, generation_config: dict) -> dict:
        return generation_config


# ================================================================
# =                                                              =
# =                 TabStruct Generator Classes                  =
# =                                                              =
# ================================================================
class BaseJointGenerator(JointGenerationMixin, BaseGenerator):
    """Base class for joint generators (non-TabEval)."""

    def __init__(self, args):
        super().__init__(args)


class BaseConditionalGenerator(ConditionalGenerationMixin, BaseGenerator):
    """Base class for conditional generators (non-TabEval)."""

    def __init__(self, args):
        super().__init__(args)


class BaseClassFocusedGenerator(ClassFocusedGenerationMixin, BaseGenerator):
    """Generators initially designed for classification tasks.
    We further extend them to be applicable for other tasks like regression and unsupervision.
    """

    def __init__(self, args):
        super().__init__(args)


# ================================================================
# =                                                              =
# =                    TabEval Generator Classes                 =
# =                                                              =
# ================================================================
class TabEvalGenerationMixin:
    """Mixin for TabEval-specific functionality."""

    def _fit_model(self, data_df):
        """TabEval-specific model fitting."""
        self.model.fit(data_df, cond=self.cond_for_fit)

    def _generate_model(self, generation_config: dict):
        syn_data_loader = self.model.generate(
            count=generation_config["num_samples_to_generate"],
            cond=generation_config["cond_for_generation"],
        )
        data_syn_temp_df = syn_data_loader.dataframe()

        return data_syn_temp_df

    @classmethod
    def _get_model_specific_scaler_config(cls):
        """Get TabEval-specific scaler configuration."""
        scaler_config_dict = {
            "context": {
                # Though TabEval has inherent data preprocessing strategies, they are limited and fragile
                # Thus, we still need basic preprocessing in the pipeline
                "disable_preprocessing": False,
            },
            "feature_scaler": {
                "categorical_transform": "ordinal",
                "categorical_as_numerical": False,
            },
            "target_scaler": {},
        }

        return scaler_config_dict


class BaseTabEvalJointGenerator(JointGenerationMixin, TabEvalGenerationMixin, BaseGenerator):
    """TabEval generator with joint generation strategy."""

    def __init__(self, args):
        super().__init__(args)

    def _update_generation_config(self, generation_config: dict) -> dict:
        generation_config = {
            "num_samples_to_generate": generation_config["num_samples_to_generate"],
            "cond_for_generation": None,
        }

        return generation_config


class BaseTabEvalConditionalGenerator(ConditionalGenerationMixin, TabEvalGenerationMixin, BaseGenerator):
    """TabEval generator with conditional generation strategy."""

    def __init__(self, args):
        super().__init__(args)

    def _update_generation_config(self, generation_config: dict) -> dict:
        # Parse generation conditions
        cond_for_generation = self._update_generation_cond(
            generation_config["class_id"], generation_config["num_samples_to_generate"]
        )

        # Update generation config
        generation_config = {
            "num_samples_to_generate": generation_config["num_samples_to_generate"],
            "cond_for_generation": cond_for_generation,
        }

        return generation_config

    def _update_generation_cond(self, class_id, num_samples_to_generate):
        cond_for_generation = self.cond_for_generation
        if self.args.task == "classification":
            cond_for_generation = [class_id] * num_samples_to_generate
        else:
            cond_for_generation = self.cond_for_generation
            if (
                cond_for_generation is not None
                and (not isinstance(cond_for_generation, str))
                and (len(cond_for_generation) != num_samples_to_generate)
            ):
                cond_for_generation = np.resize(cond_for_generation, num_samples_to_generate)

        return cond_for_generation


# ================================================================
# =                                                              =
# =               Pytorch Lightning Generator Classes            =
# =                                                              =
# ================================================================
class LitGenerationMixin:
    # ================================================================
    # =                                                              =
    # =                        General                               =
    # =                                                              =
    # ================================================================
    def _fit(self, data_module):
        # Build the trainer
        trainer = self.create_lit_trainer()

        # Model-specific preparations before fitting the model, such as preprocessing the data etc.
        data_module_dict = self._prepare_data_module(data_module)
        DataHelper.log_data_properties_runtime(self.args, data_module_dict, stage="model")

        # Train the model
        data_module = data_module_dict["data_module"]
        self.train_lit_model(data_module, trainer)

        # Model-specific special operations for fitting the model, such as saving the embeddings etc.
        self._fit_model(data_module)

    # ================================================================
    # =                                                              =
    # =                     Model-specific                           =
    # =                                                              =
    # ================================================================
    def _prepare_data_module(self, data_module):
        return {
            "data_module": data_module,
            "feature_scaler": None,
            "target_scaler": None,
        }

    def _fit_model(self, data_module):
        pass


class BaseLitJointGenerator(LitGenerationMixin, JointGenerationMixin, LitModelMixin, BaseGenerator):
    """Base class for all PyTorch Lightning models used in the generation task with joint generation strategy.
    This class provides a common interface for training, validation, and testing of lightning models.
    """

    def __init__(self, args):
        super().__init__(args)

    def _update_generation_config(self, generation_config):
        generation_config = {
            "num_samples_to_generate": generation_config["num_samples_to_generate"],
        }

        return generation_config

    # ================================================================
    # =                                                              =
    # =                     Model-specific                           =
    # =                                                              =
    # ================================================================
    def _prepare_data_module(self, data_module):
        # Deep copy the data module to avoid changing the original one
        data_module = copy.deepcopy(data_module)

        # === Prepare the data scalers ===
        self._prepare_data_scalers()

        # === Encode the features ===
        self.feature_scaler = self.feature_scaler.fit(data_module.X_train_df)
        data_module.X_train_df = self.feature_scaler.transform(data_module.X_train_df)
        data_module.X_valid_df = self.feature_scaler.transform(data_module.X_valid_df)
        data_module.X_test_df = self.feature_scaler.transform(data_module.X_test_df)

        # === Encode the target ===
        if self.args.task != "unsupervision":
            self.target_scaler = self.target_scaler.fit(data_module.y_train_df)
            data_module.y_train_df = self.target_scaler.transform(data_module.y_train_df)
            data_module.y_valid_df = self.target_scaler.transform(data_module.y_valid_df)
            data_module.y_test_df = self.target_scaler.transform(data_module.y_test_df)

        return {
            "data_module": data_module,
            "feature_scaler": self.feature_scaler,
            "target_scaler": self.target_scaler,
        }

    def _prepare_data_scalers(self):
        # Feature scaler
        self.feature_scaler = TabularEncoder(categorical_encoder="onehot", continuous_encoder="bayesian_gmm")

        # Target scaler
        self.target_scaler = None
        if self.args.task != "unsupervision":
            self.target_scaler = TabularEncoder(categorical_encoder="onehot", continuous_encoder="bayesian_gmm")

    def _fit_model(self, data_module):
        pass

    def _generate_model(self, generation_config):
        # Generate synthetic samples according to the config
        data_syn_df = self.model.generate(
            num_samples=generation_config["num_samples_to_generate"],
        )
        data_syn_df = pd.DataFrame(data_syn_df.detach().cpu())

        # Post-process the generated samples to original format
        data_syn_df = self._postprocess_generated_data(data_syn_df)

        return data_syn_df

    def _postprocess_generated_data(self, data_syn_df):
        X_syn_df = data_syn_df
        y_syn_df = None
        if self.args.task != "unsupervision":
            count_features = self.feature_scaler.n_features()
            X_syn_df = data_syn_df.iloc[:, :count_features]
            y_syn_df = data_syn_df.iloc[:, count_features:]

        data_syn_df = self.feature_scaler.inverse_transform(X_syn_df)
        if self.args.task != "unsupervision":
            y_syn_df = self.target_scaler.inverse_transform(y_syn_df)
            data_syn_df = pd.concat([data_syn_df, y_syn_df], axis=1)

        return data_syn_df


class BaseLightningGenerationModule(BaseLightningModule, metaclass=ABCMeta):
    """Base class for all PyTorch Lightning models used in the generation task.
    This class provides a common interface for training, validation, and testing of lightning models.
    """

    def __init__(self, args):
        super().__init__(args)

    # ================================================================
    # =                                                              =
    # =                         General                              =
    # =                                                              =
    # ================================================================
    def _step(self, X, y_true):
        data_real = X
        if self.args.task != "unsupervision":
            # If y_true is 1D, convert to 2D for concatenation
            y_true = y_true.unsqueeze(1) if len(y_true.shape) == 1 else y_true
            data_real = torch.cat([X, y_true], dim=1)

        # Generate synthetic data
        forward_dict = self.torch_model(data_real)

        # Compute losses
        loss_dict = self.compute_loss(data_real, forward_dict)

        return {
            "total_loss": loss_dict["total_loss"],
            "loss_dict": loss_dict,
        }

    def _compute_metric(self, step_dict_list):
        metric_dict = {}

        return metric_dict

    def generate(self, num_samples: int) -> torch.Tensor:
        return self._generate(num_samples)

    # ================================================================
    # =                                                              =
    # =                     Model-specific                           =
    # =                                                              =
    # ================================================================
    @abstractmethod
    def _create_torch_model(self):
        """Creates the PyTorch model.

        Args:
            args (Namespace): The arguments for the model.
        """
        raise NotImplementedError("This method has to be implemented by the sub class")

    @abstractmethod
    def _compute_loss(self, data_real: torch.Tensor, forward_dict: dict):
        raise NotImplementedError("This method has to be implemented by the sub class")

    @abstractmethod
    def _generate(self, num_samples: int) -> torch.Tensor:
        raise NotImplementedError("This method has to be implemented by the sub class")
