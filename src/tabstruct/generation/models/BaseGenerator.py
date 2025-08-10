from abc import abstractmethod

import numpy as np
import pandas as pd
from tabcamel.data.transform import CategoryTransform, SimpleImputeTransform
from tabeval.plugins.core.constraints import Constraints

from src.tabstruct.common.model.BaseModel import BaseModel
from src.tabstruct.common.runtime.log.TerminalIO import TerminalIO


class BaseGenerator(BaseModel):

    def __init__(self, args):
        super().__init__(args)

    def generate(self):
        """Generate synthetic samples."""
        # === Compute the generation configurations ===
        # Compute the number of synthetic samples for each class
        class2synthetic_samples = self.compute_class2synthetic_samples()

        # === Generate synthetic samples ===
        synthetic_data_dict = self._generate(class2synthetic_samples)

        # === Convert the synthetic samples to unified type ===
        synthetic_data_dict["X_syn"] = np.array(synthetic_data_dict["X_syn"], dtype=np.float32)
        synthetic_data_dict["y_syn"] = np.array(
            synthetic_data_dict["y_syn"], dtype=np.int64 if self.args.task == "classification" else np.float32
        )

        return synthetic_data_dict

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
        if self.args.task == "regression":
            class2synthetic_distribution = {
                "real": 1,  # Real samples to fit SMOTE/TabEBM
                "dummy": 0,  # Dummy samples to fit SMOTE/TabEBM
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

    @abstractmethod
    def _generate(self, class2synthetic_samples):
        raise NotImplementedError("This method has to be implemented by the sub class")


# ================================================================
# =                                                              =
# =                        Imblearn                              =
# =                                                              =
# ================================================================
class BaseImblearnGenerator(BaseGenerator):

    def __init__(self, args):
        super().__init__(args)


# ================================================================
# =                                                              =
# =                        TabEval                               =
# =                                                              =
# ================================================================
class BaseTabEvalGenerator(BaseGenerator):

    def __init__(self, args):
        super().__init__(args)

        self.disable_feature_encoding = self.args.categorical_transform == "onehot" and self.args.model not in [
            "tabddpm"
        ]

    def _fit(self, data_module):
        # Some TabEval models has intrinsic data preprocessing, so we should reverse the one-hot encoding for categorical features
        if self.disable_feature_encoding:
            train_data_dict = self.format_train_data(data_module)
        else:
            train_data_dict = {
                "X_train": data_module.X_train,
                "y_train": data_module.y_train,
            }

        self._fit_tabeval(train_data_dict)

    def format_train_data(self, data_module):
        # Valid and Test data are not used for training, so we only need to format the training data
        X_train = pd.DataFrame(data_module.X_train, columns=self.args.full_feature_col_list_processed)
        y_train = pd.DataFrame(data_module.y_train, columns=[self.args.full_target_col_processed])

        # === Reverse the onehot encoding ===
        # TabEval does not handle string categories, and thus we need to apply ordinal encoding
        for scaler_idx, feature_scaler in enumerate(self.args.feature_scaler_list[::-1]):
            if isinstance(feature_scaler, CategoryTransform):
                X_train = feature_scaler.inverse_transform(X_train)
                for i in range(scaler_idx):
                    if isinstance(self.args.feature_scaler_list[i], SimpleImputeTransform):
                        X_train = self.args.feature_scaler_list[i].transform(X_train)
                        break

        # === Apply ordinal encoding ===
        self.ordinal_feature_scaler = CategoryTransform(
            categorical_feature_list=feature_scaler.categorical_feature_list,
            strategy="ordinal",
        )
        self.ordinal_feature_scaler.fit(X_train)
        X_train = self.ordinal_feature_scaler.transform(X_train)

        # === Align the columns of the training data with the original data ===
        X_train = X_train[self.args.full_feature_col_list_original]
        y_train = y_train[self.args.full_target_col_original]

        return {
            "X_train": X_train,
            "y_train": y_train,
        }

    def _generate(self, class2synthetic_samples):
        synthetic_data_dict = self._generate_tabeval(class2synthetic_samples)

        # TabEval has intrinsic data preprocessing, so we should one-hot encode the synthetic data for uniform evaluation
        if self.disable_feature_encoding:
            synthetic_data_dict = self.format_synthetic_data(synthetic_data_dict)

        return synthetic_data_dict

    def format_synthetic_data(self, synthetic_data_dict):
        X_syn = pd.DataFrame(synthetic_data_dict["X_syn"], columns=self.args.full_feature_col_list_original)
        y_syn = pd.DataFrame(synthetic_data_dict["y_syn"], columns=[self.args.full_target_col_original])

        # === Reverse the ordinal encoding ===
        if self.ordinal_feature_scaler is not None:
            X_syn = self.ordinal_feature_scaler.inverse_transform(X_syn)

        # === Apply one-hot encoding ===
        for feature_scaler in self.args.feature_scaler_list:
            if isinstance(feature_scaler, CategoryTransform):
                X_syn = feature_scaler.transform(X_syn)

        # === Align the columns of the synthetic data with the original data ===
        X_syn = X_syn[self.args.full_feature_col_list_processed]
        y_syn = y_syn[self.args.full_target_col_processed]

        return {
            "X_syn": X_syn.to_numpy(),
            "y_syn": y_syn.to_numpy(),
        }

    @abstractmethod
    def _fit_tabeval(self, data_module):
        raise NotImplementedError("This method has to be implemented by the sub class")

    @abstractmethod
    def _generate_tabeval(self, class2synthetic_samples):
        raise NotImplementedError("This method has to be implemented by the sub class")


class BaseTabEvalConditionalGenerator(BaseTabEvalGenerator):

    def __init__(self, args):
        super().__init__(args)

    def _fit_tabeval(self, train_data_dict):
        data_df = pd.DataFrame(train_data_dict["X_train"])
        data_df[self.args.full_target_col_processed] = train_data_dict["y_train"]

        # === Prepare the conditions ===
        cond = self.prepare_cond(train_data_dict)
        cond_for_fit = cond["cond_for_fit"]
        self.cond_for_generation = cond["cond_for_generation"]

        # === Fit the model ===
        self.model.fit(data_df, cond=cond_for_fit)

    def prepare_cond(self, train_data_dict):
        cond_for_generation = train_data_dict["y_train"]

        if self.args.model in ["great"]:
            cond_for_fit = self.args.full_target_col_processed
        elif self.args.model in ["ctgan", "tvae"] and self.args.task == "regression":
            # CTGAN does not support conditional generation for regression tasks
            cond_for_fit = None
            cond_for_generation = None
        else:
            # For other models, we need to prepare the condition for generation
            cond_for_fit = train_data_dict["y_train"]

        return {
            "cond_for_fit": cond_for_fit,
            "cond_for_generation": cond_for_generation,
        }

    def _generate_tabeval(self, class2synthetic_samples):
        X_syn_list = []
        y_syn_list = []

        for class_id, num_synthetic_samples in class2synthetic_samples.items():
            num_current_synthetic_samples = 0
            patience = 0
            max_patience = num_synthetic_samples // 1000 + 10
            while num_current_synthetic_samples < num_synthetic_samples and patience < max_patience:
                num_samples_to_generate = num_synthetic_samples - num_current_synthetic_samples
                num_samples_to_generate = min(num_samples_to_generate, 1000)

                # Compute the generation configurations
                if self.args.task == "classification":
                    cond = [class_id] * num_samples_to_generate
                    constraints = Constraints(
                        rules=[
                            (self.args.full_target_col_processed, "==", class_id),
                            # Note: When `strict=False`, the dtype of the constrained column is set to float by default,
                            # which may cause issues when performaing matching as the target column is expected to be int for classification tasks.
                            (self.args.full_target_col_processed, "dtype", "int"),
                        ]
                    )
                else:
                    cond = self.cond_for_generation
                    if cond is not None and (not isinstance(cond, str)) and (len(cond) != num_samples_to_generate):
                        cond = np.resize(cond, num_samples_to_generate)
                    constraints = None

                syn_data_loader = self.model.generate(
                    count=num_samples_to_generate,
                    cond=cond,
                    constraints=constraints,
                )
                X_syn_temp, y_syn_temp = syn_data_loader.unpack()
                X_syn_list.append(X_syn_temp)
                y_syn_list.append(y_syn_temp)

                num_current_synthetic_samples += len(X_syn_temp)
                patience += 1

            X_syn = pd.concat(X_syn_list).to_numpy()
            y_syn = pd.concat(y_syn_list).to_numpy()
            if X_syn.shape[0] < num_synthetic_samples:
                upsample_index = np.random.choice(X_syn.shape[0], num_synthetic_samples)
                X_syn = X_syn[upsample_index]
                y_syn = y_syn[upsample_index]

        return {
            "X_syn": X_syn,
            "y_syn": y_syn,
        }


class BaseTabEvalJointGenerator(BaseTabEvalGenerator):

    def __init__(self, args):
        super().__init__(args)

    def _fit_tabeval(self, train_data_dict):
        data_df = pd.DataFrame(train_data_dict["X_train"])
        data_df[self.args.full_target_col_processed] = train_data_dict["y_train"]
        data_df = self.process_with_corner_case(data_df)
        self.model.fit(data_df)

    def _generate_tabeval(self, class2synthetic_samples):
        X_syn = pd.DataFrame()
        y_syn = pd.Series()

        for class_id, num_synthetic_samples in class2synthetic_samples.items():
            # Compute the generation configurations
            constraints = None
            if self.args.task == "classification":
                constraints = Constraints(
                    rules=[
                        (self.args.full_target_col_processed, "==", class_id),
                        # Note: When `strict=False`, the dtype of the constrained column is set to float by default,
                        # which may cause issues when performaing matching as the target column is expected to be int for classification tasks.
                        (self.args.full_target_col_processed, "dtype", "int"),
                    ]
                )

            # Generate synthetic samples for the current class in a GPU-memory-efficient way
            num_current_synthetic_samples = 0
            patience = 0
            max_patience = num_synthetic_samples // 1000 + 10
            while num_current_synthetic_samples < num_synthetic_samples and patience < max_patience:
                num_samples_to_generate = num_synthetic_samples - num_current_synthetic_samples
                num_samples_to_generate = min(num_samples_to_generate, 1000)
                syn_data_loader = self.model.generate(
                    count=num_samples_to_generate,
                    constraints=constraints,
                )
                X_syn_temp, y_syn_temp = syn_data_loader.unpack()
                X_syn = pd.concat([X_syn, X_syn_temp], axis=0)
                y_syn = pd.concat([y_syn, y_syn_temp], axis=0)

                num_current_synthetic_samples += len(X_syn_temp)
                patience += 1

            if X_syn.shape[0] < num_synthetic_samples:
                upsample_index = np.random.choice(X_syn.shape[0], num_synthetic_samples)
                X_syn = X_syn.iloc[upsample_index]
                y_syn = y_syn.iloc[upsample_index]

        return {
            "X_syn": X_syn.to_numpy(),
            "y_syn": y_syn.to_numpy(),
        }

    def process_with_corner_case(self, data_df):
        if self.args.model in ["bn"]:
            # Discretize all columns to up to 100 bins per column
            for col in data_df.columns:
                if self.args.full_feature_col2type_original.get(col, None) == "numerical":
                    data_df[col] = pd.cut(data_df[col], bins=100, labels=False, duplicates="drop")
            # BN cannot scale up to more than 10k samples
            data_df = data_df.loc[:10000, :]
        elif self.args.model in ["nflow"]:
            # NFlow is very unstable with more than 2k samples
            data_df = data_df.loc[:2000, :]

        return data_df


class BaseMixedGenerator(BaseGenerator):

    def __init__(self, args):
        super().__init__(args)

        self.patience_for_generation = 10

    def prepare_data(self, data_module):
        # === Prepare the data according to feature type ===
        X_train, y_train = data_module.X_train, data_module.y_train
        X_valid, y_valid = data_module.X_valid, data_module.y_valid

        # For classification, we save a sample per class to duplicate when the generator cannot generate for some classes.
        if self.args.task == "classification":
            example_indices = [np.where(y_train == class_id)[0][0] for class_id in np.unique(y_train)]
            self.X_train_example = X_train[example_indices, :]
            self.y_train_example = y_train[example_indices]

        # By default, feature preprocessing will sort all features as numerical -> categorical (stable sort)
        num_col_count = len(self.args.num_feature_col_list_processed)
        X_train_num, X_train_cat = X_train[:, :num_col_count], X_train[:, num_col_count:]
        X_valid_num, X_valid_cat = X_valid[:, :num_col_count], X_valid[:, num_col_count:]
        if self.args.task == "regression":
            X_train_num = np.concatenate([X_train_num, y_train.reshape(-1, 1)], axis=1)
            X_valid_num = np.concatenate([X_valid_num, y_valid.reshape(-1, 1)], axis=1)
        elif self.args.task == "classification":
            X_train_cat = np.concatenate([X_train_cat, y_train.reshape(-1, 1)], axis=1)
            X_valid_cat = np.concatenate([X_valid_cat, y_valid.reshape(-1, 1)], axis=1)
        X_train_cat = X_train_cat.astype(np.int64)
        X_valid_cat = X_valid_cat.astype(np.int64)

        # Get the bounds of the numerical features
        self.num_lower_bound = np.min(X_train_num, axis=0)
        self.num_upper_bound = np.max(X_train_num, axis=0)

        return {
            "X_train_num": X_train_num,
            "X_train_cat": X_train_cat,
            "X_valid_num": X_valid_num,
            "X_valid_cat": X_valid_cat,
        }

    def _generate(self, class2synthetic_samples):
        patience = 0
        X_syn_df = pd.DataFrame()
        y_syn_df = pd.DataFrame()
        while patience < self.patience_for_generation:
            # === Generate synthetic data ===
            while X_syn_df.shape[0] < sum(class2synthetic_samples.values()):
                syn_df_dict = self._generate_single_run()
                X_syn_df = pd.concat([X_syn_df, syn_df_dict["X_syn_df"]], axis=0)
                y_syn_df = pd.concat([y_syn_df, syn_df_dict["y_syn_df"]], axis=0)

            # Re-sampling to satisfy the class2synthetic_samples
            if self.args.task == "classification":
                if pd.unique(y_syn_df.iloc[:, -1]).size != len(class2synthetic_samples.keys()):
                    TerminalIO.print(
                        f"Warning: The number of classes in the generated data ({pd.unique(y_syn_df.iloc[:, -1]).size}) "
                        f"does not match the number of classes in the original data ({len(class2synthetic_samples)})",
                        color=TerminalIO.WARNING,
                    )
                    patience += 1
                    if patience < self.patience_for_generation - 1:
                        continue

                    TerminalIO.print(
                        f"Patience {self.patience_for_generation} exceeded. "
                        f"The model cannot synthesise samples for some classes. Thus we will use example samples instead.",
                        color=TerminalIO.WARNING,
                    )
                    X_syn_df = pd.concat(
                        [X_syn_df, pd.DataFrame(self.X_train_example, columns=X_syn_df.columns)], axis=0
                    )
                    y_syn_df = pd.concat(
                        [y_syn_df, pd.DataFrame(self.y_train_example, columns=y_syn_df.columns)], axis=0
                    )

                # Resample the synthetic data
                X_syn = X_syn_df.to_numpy()
                y_syn = y_syn_df.to_numpy()

                resample_indices = np.concatenate(
                    [
                        np.random.choice(np.where(y_syn == class_id)[0], num_synthetic_samples, replace=True)
                        for class_id, num_synthetic_samples in class2synthetic_samples.items()
                    ]
                ).reshape(-1)
                X_syn = X_syn[resample_indices, :]
                y_syn = y_syn[resample_indices, :]
                break
            else:
                X_syn = X_syn_df.to_numpy()
                y_syn = y_syn_df.to_numpy()
                break

        return {
            "X_syn": X_syn,
            "y_syn": y_syn,
        }

    def _generate_single_run(self):
        synthetic_data_dict = self._model_generate()

        # === Parse the generated data ===
        syn_num = synthetic_data_dict["syn_num"]
        syn_cat = synthetic_data_dict["syn_cat"]

        # Set the bounds of the numerical features
        syn_num = np.clip(syn_num, self.num_lower_bound, self.num_upper_bound)

        # Parse the target column
        y_syn = None
        if self.args.task == "regression":
            y_syn = syn_num[:, -1]
            syn_num = syn_num[:, :-1]
        elif self.args.task == "classification":
            y_syn = syn_cat[:, -1]
            syn_cat = syn_cat[:, :-1]

        # Order the columns of generated samples
        syn_num_df = pd.DataFrame(syn_num, columns=self.args.num_feature_col_list_processed)
        syn_cat_df = pd.DataFrame(syn_cat, columns=self.args.cat_feature_col_list_processed)
        y_syn_df = pd.DataFrame(y_syn, columns=[self.args.full_target_col_processed])
        X_syn_df = pd.concat([syn_num_df, syn_cat_df], axis=1)

        return {
            "X_syn_df": X_syn_df,
            "y_syn_df": y_syn_df,
        }

    @abstractmethod
    def _model_generate(self):
        """Generate synthetic data using a specific model.

        Returns:
            dict: The generated synthetic data
        """
        raise NotImplementedError("This method has to be implemented by the sub class")
