import numpy as np
import pandas as pd
import wandb
from sklearn.utils import compute_class_weight
from tabcamel.data.dataset import TabularDataset
from tabcamel.data.transform import CategoryTransform, NumericTransform, SimpleImputeTransform, TargetTransform

from ..runtime.log.TerminalIO import TerminalIO
from .DataModule import DataModule


class DataHelper:
    """Helper class to prepare data"""

    # ================================================================
    # =                                                              =
    # =                    Data preparation                          =
    # =                                                              =
    # ================================================================
    @classmethod
    @TerminalIO.trace_func
    def create_data_module(cls, args):
        """Create a data module for training, validation and testing."""
        # === Load the dataset ===
        dataset_dict_original = cls.load_dataset(args)
        data_info_dict = cls.log_data_properties(args, dataset_dict_original, {}, stage="original")

        # === Split the dataset ===
        split_dict = cls.split_full_dataset(args, dataset_dict_original["full_set"])
        data_info_dict = cls.log_data_properties(args, split_dict, data_info_dict, stage="split")

        # === Process the dataset ===
        # Curate the dataset
        split_dict_curated = cls.curate_dataset(args, split_dict)
        # Preprocess the dataset
        split_dict_processed = cls.preprocess_dataset(args, split_dict_curated)
        # Log the data properties
        data_info_dict = cls.log_data_properties(args, split_dict_processed, data_info_dict, stage="processed")
        wandb.log({"data_info": wandb.Table(dataframe=pd.DataFrame(data_info_dict).T)})

        # ===== Create the data module =====
        # Extract processed datasets
        train_set = split_dict_processed["train_set"]
        valid_set = split_dict_processed["valid_set"]
        test_set = split_dict_processed["test_set"]

        # Create data module with numpy arrays for better performance
        data_module = DataModule(
            args,
            train_set.X_df.to_numpy(),
            train_set.y_s.to_numpy(),
            valid_set.X_df.to_numpy(),
            valid_set.y_s.to_numpy(),
            test_set.X_df.to_numpy(),
            test_set.y_s.to_numpy(),
        )

        return data_module

    @classmethod
    def log_data_properties(cls, args, dataset_dict, data_info_dict, stage):
        """Log the dataset properties for display and runtime reference.

        Args:
            args (Namespace): arguments
            dataset_dict (dict): dictionary of datasets/splits
            data_info_dict (dict): dictionary of data properties at different stages
            stage (str): stage of the data preparation

        Returns:
            dict: updated data_info_dict
        """
        # === Log the dataset properties for runtime reference ===
        cls.log_data_properties_runtime(args, dataset_dict, stage=stage)

        # === Log the dataset properties for display ===
        data_info_dict = cls.log_data_properties_display(dataset_dict, data_info_dict, stage=stage)

        return data_info_dict

    @staticmethod
    def log_data_properties_display(dataset_dict, data_info, stage):
        """Log the dataset properties for display and debugging."""
        split2attr = {
            "full": [
                "num_samples",
                "class2samples",
                "class2distribution",
                "num_features",
                "num_classes",
                "metafeature_dict",
            ],
            "train": [
                "num_samples",
                "class2samples",
                "class2distribution",
            ],
            "valid": [
                "num_samples",
                "class2samples",
                "class2distribution",
            ],
            "test": [
                "num_samples",
                "class2samples",
                "class2distribution",
            ],
        }
        data_info[stage] = {}
        # Wandb ignores the index when converting to wandb.Table, thus make the first column as index
        data_info[stage]["stage"] = stage
        for split, attr_list in split2attr.items():
            for attr in attr_list:
                data_info[stage][f"{split}_{attr}"] = str(getattr(dataset_dict.get(f"{split}_set", None), attr, None))

        return data_info

    @staticmethod
    def log_data_properties_runtime(args, dataset_dict, stage):
        """Log the dataset properties for runtime reference."""
        if stage == "original":
            # === Feature names before transformation ===
            feature_list = list(dataset_dict["full_set"].col2type.keys())
            feature_list.remove(dataset_dict["full_set"].target_col)
            args.full_feature_col_list_original = feature_list
            args.full_feature_col2type_original = dataset_dict["full_set"].col2type
            args.full_target_col_original = dataset_dict["full_set"].target_col
            args.num_feature_col_list_original = [
                col
                for col in args.full_feature_col_list_original
                if args.full_feature_col2type_original[col].name == "numerical"
            ]
        elif stage == "split":
            args.train_num_samples_split = dataset_dict["train_set"].num_samples
        elif stage == "processed":
            split2attr = {
                "full": [
                    "num_features",
                    "num_classes",
                    "target_col",
                ],
                "train": [
                    "num_samples",
                    "class2samples",
                    "class2distribution",
                ],
                "valid": [
                    "num_samples",
                ],
                "test": [
                    "num_samples",
                ],
            }
            for split, attr_list in split2attr.items():
                for attr in attr_list:
                    setattr(args, f"{split}_{attr}_{stage}", getattr(dataset_dict[f"{split}_set"], attr))

            # ===== Log special properties =====
            # === The feature names after transformation ===
            feature_list = list(dataset_dict["full_set"].col2type.keys())
            # Note: target column is not considered as a feature
            feature_list.remove(dataset_dict["full_set"].target_col)
            args.full_feature_col_list_processed = feature_list
            args.num_feature_col_list_processed = (
                feature_list
                if args.categorical_as_numerical
                else [
                    col
                    for col in args.full_feature_col2type_original.keys()
                    if args.full_feature_col2type_original[col].name == "numerical" and col in feature_list
                ]
            )
            args.cat_feature_col_list_processed = list(
                set(feature_list).difference(args.num_feature_col_list_processed)
            )

            # === The scaler list for transformation ===
            args.feature_scaler_list = dataset_dict["feature_scaler_list"]
            args.target_scaler_list = dataset_dict["target_scaler_list"]
            if args.task == "classification":
                # === Record the mapping from encoded labels to original labels ===
                for scaler in dataset_dict["target_scaler_list"]:
                    if isinstance(scaler, TargetTransform):
                        args.encoded2class = scaler.encoded2class
                # === Compute class weights to balance datasets ===
                """
                `compute_class_weight()` returns weights in the order of `classes` argument
                Use np.unique() to ensure the weight order is consistent with the class_id order in the model
                >>> y = [1, 1, 1, 1, 0, 0]
                >>> np.unique(y)
                array([0, 1])
                >>> pd.Series(y).unique()
                array([1, 0])
                >>> compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
                array([1.5 , 0.75])
                >>> compute_class_weight(class_weight="balanced", classes=pd.Series(y).unique(), y=y)
                array([0.75, 1.5])
                """
                args.class_encoded_list = list(args.encoded2class.keys())
                class_weight_list = compute_class_weight(
                    class_weight="balanced", classes=np.array(args.class_encoded_list), y=dataset_dict["train_set"].y_s
                )
                args.train_class_weight_list = class_weight_list
                args.train_class2weight = {i: val for i, val in zip(args.class_encoded_list, class_weight_list)}

            # === Feature properties after transformation ===
            args.cat_feature_cardinality_list_processed = TabularDataset.get_cardinality_list(
                dataset_dict["full_set"],
                args.cat_feature_col_list_processed,
            )

    # ================================================================
    # =                                                              =
    # =                       Data loading                           =
    # =                                                              =
    # ================================================================
    @staticmethod
    def load_dataset(args) -> dict:
        full_set = TabularDataset(
            dataset_name=args.dataset,
            task_type=args.task,
        )

        # ===== Special options to gurantee the data is legitimate w.r.t. class numbers etc. =====
        # === Drop some samples ===
        if args.task == "classification":
            # === Drop classes with very few samples ===
            if args.min_sample_per_class is not None:
                full_set.drop_low_sample_class(min_sample_per_class=args.min_sample_per_class)

            # === Drop some classes (e.g., very hard/easy samples) ===
            if args.drop_class_id is not None:
                full_set.drop_class(class_id=args.drop_class_id)

        # === Drop some features ===
        # If a feature is constant, it should be removed for simplicity
        full_set = TabularDataset.drop_constant_feature(full_set)

        return {
            "full_set": full_set,
        }

    # ================================================================
    # =                                                              =
    # =                       Data split                             =
    # =                                                              =
    # ================================================================
    @staticmethod
    def split_full_dataset(args, full_set) -> dict:
        # ===== Split the dataset into Train/Val/Test =====
        split_dict_test = full_set.split(
            split_mode=args.split_mode,
            test_size=args.test_size,
            random_state=args.test_id,
        )
        train_val_set, test_set = split_dict_test["train_set"], split_dict_test["test_set"]
        indices_test = split_dict_test["indices_test"]

        split_dict_val = train_val_set.split(
            split_mode=args.split_mode,
            test_size=args.valid_size,
            random_state=args.valid_id,
        )
        train_set, valid_set = split_dict_val["train_set"], split_dict_val["test_set"]
        indices_train, indices_valid = split_dict_val["indices_train"], split_dict_val["indices_test"]

        return {
            # Samples
            "full_set": full_set,
            "train_set": train_set,
            "valid_set": valid_set,
            "test_set": test_set,
            # Indices
            "indices_train": indices_train,
            "indices_valid": indices_valid,
            "indices_test": indices_test,
        }

    # ================================================================
    # =                                                              =
    # =                      Data curation                           =
    # =                                                              =
    # ================================================================
    @classmethod
    def curate_dataset(cls, args, dataset_dict) -> dict:
        match args.curate_mode:
            case None:
                return dataset_dict
            case "sharing":
                # === Data sharing ===
                curate_set = cls.load_curate_data(args)
                dataset_dict["train_set"] = curate_set
            case _:
                raise NotImplementedError(f"Curate mode '{args.curate_mode}' is not implemented.")

        return dataset_dict

    @staticmethod
    def load_curate_data(args):
        # === Load the curate data ===
        curate_set = TabularDataset(
            dataset_name=args.curate_data_path,
            task_type=args.task,
            target_col=args.full_target_col_original,
            metafeature_dict={
                "col2type": args.full_feature_col2type_original,
            },
        )

        # === Subsample the curate data ===
        subsample_dict = curate_set.sample(
            sample_mode=args.split_mode,
            sample_size=int(args.curate_ratio * args.train_num_samples_split),
        )
        curate_set = subsample_dict["dataset_sampled"]

        return curate_set

    # ================================================================
    # =                                                              =
    # =                   Data preprocessing                         =
    # =                                                              =
    # ================================================================
    @classmethod
    def preprocess_dataset(cls, args, dataset_dict) -> dict:
        target_col = dataset_dict["full_set"].target_col

        # === Preprocess the features ===
        X_dict = cls.preprocess_features(args, dataset_dict)

        # === Preprocess the target ===
        y_dict = cls.preprocess_target(args, dataset_dict)

        # === Build datasets with preprocessed features and target ===
        # The preprocessing preserves the index of the original data
        train_set = TabularDataset(
            dataset_name=args.dataset,
            task_type=args.task,
            target_col=target_col,
            data_df=pd.concat([X_dict["X_train"], y_dict["y_train"]], axis=1),
            metafeature_dict={
                "is_tensor": True,
            },
        )
        valid_set = TabularDataset(
            dataset_name=args.dataset,
            task_type=args.task,
            target_col=target_col,
            data_df=pd.concat([X_dict["X_valid"], y_dict["y_valid"]], axis=1),
            metafeature_dict={
                "is_tensor": True,
            },
        )
        test_set = TabularDataset(
            dataset_name=args.dataset,
            task_type=args.task,
            target_col=target_col,
            data_df=pd.concat([X_dict["X_test"], y_dict["y_test"]], axis=1),
            metafeature_dict={
                "is_tensor": True,
            },
        )
        full_set = TabularDataset(
            dataset_name=args.dataset,
            task_type=args.task,
            target_col=target_col,
            data_df=pd.concat([train_set.data_df, valid_set.data_df, test_set.data_df], axis=0),
            metafeature_dict={
                "is_tensor": True,
            },
        )

        return {
            # Datasets
            "full_set": full_set,
            "train_set": train_set,
            "valid_set": valid_set,
            "test_set": test_set,
            # Scalers
            "feature_scaler_list": X_dict["feature_scaler_list"],
            "target_scaler_list": y_dict["target_scaler_list"],
        }

    @staticmethod
    def preprocess_features(args, dataset_dict: dict) -> dict:
        categorical_feature_list = dataset_dict["full_set"].categorical_feature_list
        numerical_feature_list = dataset_dict["full_set"].numerical_feature_list

        # === Get data splits ===
        X_train = dataset_dict["train_set"].X_df
        X_valid = dataset_dict["valid_set"].X_df
        X_test = dataset_dict["test_set"].X_df

        # === Preprocess the features ===
        feature_scaler_list = [
            # Impute missing values
            SimpleImputeTransform(
                categorical_feature_list=categorical_feature_list,
                numerical_feature_list=numerical_feature_list,
                strategy_categorical=args.categorical_impute,
                strategy_numerical=args.numerical_impute,
            ),
            # Encode the categorical features
            CategoryTransform(
                categorical_feature_list=categorical_feature_list,
                strategy=args.categorical_transform,
            ),
            # Encode the numerical features
            NumericTransform(
                numerical_feature_list=numerical_feature_list,
                strategy=args.numerical_transform,
                include_categorical=args.categorical_as_numerical,
                train_num_samples=X_train.shape[0],
            ),
        ]
        for scaler in feature_scaler_list:
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_valid = scaler.transform(X_valid)
            X_test = scaler.transform(X_test)

        return {
            "X_train": X_train,
            "X_valid": X_valid,
            "X_test": X_test,
            "feature_scaler_list": feature_scaler_list,
        }

    @staticmethod
    def preprocess_target(args, dataset_dict: dict) -> dict:
        # === Dataset properties ===
        target_col = dataset_dict["full_set"].target_col

        # === Get data splits ===
        y_train = dataset_dict["train_set"].data_df[[target_col]]
        y_valid = dataset_dict["valid_set"].data_df[[target_col]]
        y_test = dataset_dict["test_set"].data_df[[target_col]]

        # === Preprocess the target ===
        target_scaler_list = [
            # Impute the target
            SimpleImputeTransform(
                categorical_feature_list=[target_col] if args.task == "classification" else [],
                numerical_feature_list=[target_col] if args.task != "classification" else [],
                strategy_categorical="most_frequent",
                strategy_numerical="mean",
            ),
            # Encode or standardise the target
            TargetTransform(
                task=args.task,
                target_feature=target_col,
            ),
        ]
        for scaler in target_scaler_list:
            scaler.fit(y_train)
            y_train = scaler.transform(y_train)
            y_valid = scaler.transform(y_valid)
            y_test = scaler.transform(y_test)

        return {
            "y_train": y_train,
            "y_valid": y_valid,
            "y_test": y_test,
            "target_scaler_list": target_scaler_list,
        }

    @classmethod
    def recover_original_data(cls, args, X_processed: np.ndarray, y_processed: np.ndarray) -> dict:
        # === Convert the processed samples to DataFrame ===
        X_original = pd.DataFrame(X_processed, columns=args.full_feature_col_list_processed)
        y_original = pd.DataFrame(y_processed, columns=[args.full_target_col_processed])

        # === Inverse transform the synthetic features ===
        for feature_scaler in args.feature_scaler_list[::-1]:
            X_original = feature_scaler.inverse_transform(X_original)
        for target_scaler in args.target_scaler_list[::-1]:
            y_original = target_scaler.inverse_transform(y_original)

        # === Align the order of the columns ===
        X_original = X_original[args.full_feature_col_list_original]
        y_original = y_original[args.full_target_col_original]

        return {
            "X_original": X_original,
            "y_original": y_original,
        }
