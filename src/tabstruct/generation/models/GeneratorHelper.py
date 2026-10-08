import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from tabcamel.data.dataset import TabularDataset
from tabeval.plugins.core.dataloader import GenericDataLoader

from src.tabstruct.common import LOG_DIR, WANDB_PROJECT
from src.tabstruct.common.data.DataHelper import DataHelper
from src.tabstruct.common.model.BaseModelHelper import BaseModelHelper
from src.tabstruct.common.runtime.log.FileHelper import FileHelper

from .utils.evaluation import compute_all_metrics


class GeneratorHelper(BaseModelHelper):
    """Helper class for training and evaluating models."""

    # ================================================================
    # =                                                              =
    # =                     Initialisation                           =
    # =                                                              =
    # ================================================================
    @classmethod
    def _model_handler(cls, model):
        # Use lazy imports to accelerate the startup time
        match model:
            case "smote":
                from .imblearn.SMOTE import SMOTE

                model_class = SMOTE
            case "ctgan":
                from .synth.CTGAN import CTGAN

                model_class = CTGAN
            case "tvae":
                from .synth.TVAE import TVAE

                model_class = TVAE
            case "bn":
                from .synth.BN import BN

                model_class = BN
            case "goggle":
                from .synth.GOGGLE import GOGGLE

                model_class = GOGGLE
            case "tabddpm":
                from .synth.TabDDPM import TabDDPM

                model_class = TabDDPM
            case "arf":
                from .synth.ARF import ARF

                model_class = ARF
            case "nflow":
                from .synth.NFLOW import NFLOW

                model_class = NFLOW
            case "great":
                from .custom.great.GReaT import GReaT

                model_class = GReaT
            case "tabebm":
                from .custom.tabebm.TabEBM import TabEBM

                model_class = TabEBM
            case "real" | "real-test":
                from .custom.real.Real import Real

                model_class = Real
            case "nrgboost":
                from .custom.nrgboost.NRGBoost import NRGBoost

                model_class = NRGBoost
            case "tabular-argn":
                from .custom.argn.TabularARGN import TabularARGN

                model_class = TabularARGN
            case "ae":
                from .custom.ae.LitAE import LitAE

                model_class = LitAE
            case "vae":
                from .custom.ae.LitVAE import LitVAE

                model_class = LitVAE
            case "ddpm":
                from .custom.ddpm.LitDDPM import LitDDPM

                model_class = LitDDPM
            case "tddpm":
                from .custom.ddpm.LitTDDPM import LitTDDPM

                model_class = LitTDDPM
            case "tabdiff":
                from .custom.cdm.LitTabDiff import LitTabDiff

                model_class = LitTabDiff
            case "vesde":
                from .custom.cdm.LitVESDE import LitVESDE

                model_class = LitVESDE
            case "vpsde":
                from .custom.cdm.LitVPSDE import LitVPSDE

                model_class = LitVPSDE
            case "edm":
                from .custom.cdm.LitEDM import LitEDM

                model_class = LitEDM
            case "tabsyn":
                from .custom.cdm.LitTabSyn import LitTabSyn

                model_class = LitTabSyn
            case "tabforge":
                from .custom.tabforge.TabFORGE import TabFORGE

                model_class = TabFORGE
            case _:
                raise NotImplementedError(f"Model {model} is not implemented.")

        return model_class

    # ================================================================
    # =                                                              =
    # =                       Evaluation                             =
    # =                                                              =
    # ================================================================
    @classmethod
    def _eval_model(cls, args, data_module, model):
        # === Prepare synthetic samples ===
        synthetic_data_path = args.synthetic_data_path
        if synthetic_data_path is None:
            data_syn_df_raw = cls.inference(data_module, model)
            # Save synthetic samples
            synthetic_data_path = cls.save_synthetic_data(args, data_syn_df_raw)

        # === Evaluate the synthetic data ===
        metric_dict = {}
        if not args.generation_only:
            # Load real and synthetic data
            loader_dict = cls.prepare_data_loader(args, data_module, synthetic_data_path)

            # Compute all metrics
            metric_dict = compute_all_metrics(args, loader_dict["real"], loader_dict["syn"])

        # === Custom logging ===
        # Record the computation cost
        cls.log_computation_cost(data_module, model)

        return metric_dict

    @classmethod
    def _inference(cls, data_module, model):
        data_syn_df_raw = model.generate()

        return data_syn_df_raw

    # ================================================================
    # =                                                              =
    # =                        Data Ops                              =
    # =                                                              =
    # ================================================================
    @classmethod
    def save_synthetic_data(cls, args, data_syn_df_raw: dict | pd.DataFrame):
        # ===  Format the output to a unified format ===
        formatted_output_dict = cls._format_synthetic_data(args, data_syn_df_raw)
        X_processed = formatted_output_dict["X_syn"]
        y_processed = formatted_output_dict["y_syn"]

        # === Convert the synthetic samples to DataFrame ===
        original_data_dict = DataHelper.recover_original_data(args, X_processed, y_processed)

        # === Save the synthetic samples ===
        synthetic_samples = original_data_dict["X_original"].copy(deep=True)
        # Only add target column for supervised tasks
        if args.task != "unsupervision":
            synthetic_samples[args.full_target_col_original] = original_data_dict["y_original"]

        # Create the directory to save synthetic data
        synthetic_data_dir = cls._generate_path_to_save_synthetic_data()
        synthetic_data_path = os.path.join(synthetic_data_dir, "synthetic_samples.csv")

        # Save to CSV file
        FileHelper.save_to_csv_file(synthetic_samples, synthetic_data_path)
        wandb.run.summary["generated_data_path"] = synthetic_data_path

        return synthetic_data_path

    @classmethod
    def prepare_data_loader(cls, args, data_module, synthetic_data_path):
        """Prepare data loaders for real and synthetic data.
        Note: The preprocessing is independent of the model training, such as "model_specific_preprocessing",
        thus we can fairly compare the synthetic data from generators with different preprocessing steps.

        Args:
            args (argparse.Namespace): The arguments parsed from the command line.
            data_module (DataModule): The data module containing the dataset.
            synthetic_data_path (str): The path to the synthetic data file.

        Returns:
            dict: A dictionary containing the prepared data loaders.
        """
        # Load real data
        loader_dict_real = cls.load_real_data(args, data_module)
        DataHelper.log_data_properties_runtime(args, loader_dict_real, stage="eval")

        # Load synthetic data
        if args.model == "real":
            loader_dict_syn = loader_dict_real["train"]
        elif args.model == "real-test":
            loader_dict_syn = loader_dict_real["test"]
        else:
            loader_dict_syn = cls.load_synthetic_data(args, synthetic_data_path)

        return {"real": loader_dict_real, "syn": loader_dict_syn}

    @classmethod
    def load_real_data(cls, args, data_module):
        """Load real data according to the specified arguments.

        Args:
            args (argparse.Namespace): The arguments parsed from the command line.
            data_module (DataModule): The data module containing the dataset.

        Returns:
            dict: A dictionary containing the loaded data.
        """
        # === Load the real data ===
        # Load full data
        full_set = TabularDataset(
            dataset_name=args.dataset,
            # X and y are **not** treated differently when evaluating synthetic data
            # Thus, we can reuse the code for unsupervised tasks during preprocessing
            task_type="unsupervision",
        )
        # Split according to the specified indices
        train_set = full_set.sample(sample_mode="fixed", sample_indices=data_module.indices_train)["dataset_sampled"]
        valid_set = full_set.sample(sample_mode="fixed", sample_indices=data_module.indices_valid)["dataset_sampled"]
        test_set = full_set.sample(sample_mode="fixed", sample_indices=data_module.indices_test)["dataset_sampled"]
        original_data_dict = {
            "full_set": full_set,
            "train_set": train_set,
            "valid_set": valid_set,
            "test_set": test_set,
        }

        # === Preprocess the real data ===
        # Process onehot and ordinal transformations in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both transformation tasks concurrently
            # "all_onehot": all features + target are one-hot encoded when applicable
            onehot_future = executor.submit(
                DataHelper.preprocess_dataset,
                cls._create_preprocessing_args_for_eval(args, "onehot"),
                original_data_dict,
            )
            # "all_ordinal": all features + target are ordinal encoded when applicable
            ordinal_future = executor.submit(
                DataHelper.preprocess_dataset,
                cls._create_preprocessing_args_for_eval(args, "ordinal"),
                original_data_dict,
            )

            # Wait for both transformations to complete
            onehot_data_dict = onehot_future.result()
            ordinal_data_dict = ordinal_future.result()

        # === Prepare data loaders ===
        data_splits = ["train", "valid", "test"]
        data_dicts = {
            "all_original": original_data_dict,
            "all_onehot": onehot_data_dict,
            "all_ordinal": ordinal_data_dict,
        }

        loader_dict = {}
        for split in data_splits:
            loader_dict[split] = {
                data_type: GenericDataLoader(data_dict[f"{split}_set"].data_df)
                for data_type, data_dict in data_dicts.items()
            }

        return {
            # Data loaders
            **loader_dict,
            # Metafeatures
            "col_list": full_set.data_df.columns.tolist(),
            "col2type": full_set.col2type,
            # Scalers
            "onehot_scaler_list": onehot_data_dict["feature_scaler_list"],
            "ordinal_scaler_list": ordinal_data_dict["feature_scaler_list"],
        }

    @classmethod
    def load_synthetic_data(cls, args, synthetic_data_path):
        # === Load the synthetic samples ===
        # Ensure the synthetic data goes through all meta-processing and preprocessing steps
        full_synthetic_dataset = TabularDataset(
            dataset_name=synthetic_data_path,
            task_type=args.task,
            metafeature_dict={
                "col2type": args.full_col2type_original,
            },
        )
        # Subsample the synthetic data to align with data curation
        synthetic_dataset = full_synthetic_dataset
        if args.curate_ratio != 1.0:
            sample_dict = full_synthetic_dataset.sample(
                sample_mode=args.split_mode,
                sample_size=int(args.curate_ratio * args.train_num_samples_split),
            )
            synthetic_dataset = sample_dict["dataset_sampled"]
        # Update the synthetic dataset into unsupervision task to reuse preprocessing
        synthetic_dataset = TabularDataset(
            dataset_name=synthetic_data_path,
            task_type="unsupervision",
            data_df=synthetic_dataset.data_df,
            metafeature_dict={
                "col2type": args.full_col2type_eval,
            },
        )

        # === Preprocess the synthetic samples with scalers fitted on the real training data ===
        original_df = synthetic_dataset.data_df

        # Process onehot and ordinal transformations in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both transformation tasks concurrently
            onehot_future = executor.submit(cls._apply_scalers, original_df.copy(), args.onehot_scaler_list_eval)
            ordinal_future = executor.submit(cls._apply_scalers, original_df.copy(), args.ordinal_scaler_list_eval)

            # Wait for both transformations to complete
            onehot_df = onehot_future.result()
            ordinal_df = ordinal_future.result()

        return {
            "all_original": GenericDataLoader(original_df),
            "all_onehot": GenericDataLoader(onehot_df),
            "all_ordinal": GenericDataLoader(ordinal_df),
        }

    # ================================================================
    # =                                                              =
    # =                  Utils within class                          =
    # =                                                              =
    # ================================================================
    @staticmethod
    def _format_synthetic_data(args, data_syn_df_raw: pd.DataFrame) -> dict:
        """Format the data types of synthetic samples.

        Args:
            synthetic_data_dict (dict): Dictionary containing X_syn and y_syn keys

        Returns:
            dict: The formatted synthetic data dictionary
        """
        # Parse the raw output
        data_syn = data_syn_df_raw.to_numpy()
        X_syn = data_syn[:, :-1]
        y_syn = data_syn[:, -1][..., np.newaxis]
        if args.task == "unsupervision":
            X_syn = np.concatenate([X_syn, y_syn], axis=1)
            y_syn = None
        synthetic_data_dict = {
            "X_syn": X_syn,
            "y_syn": y_syn,
        }

        # Format X_syn
        synthetic_data_dict["X_syn"] = np.asarray(synthetic_data_dict["X_syn"])
        if not args.disable_preprocessing:
            synthetic_data_dict["X_syn"] = synthetic_data_dict["X_syn"].astype(np.float32)

        # Format y_syn based on task type
        if args.task == "unsupervision":
            synthetic_data_dict["y_syn"] = None
        else:
            synthetic_data_dict["y_syn"] = np.asarray(synthetic_data_dict["y_syn"])
            if not args.disable_preprocessing:
                target_type = np.int64 if args.task == "classification" else np.float32
                synthetic_data_dict["y_syn"] = synthetic_data_dict["y_syn"].astype(target_type)

        return synthetic_data_dict

    @staticmethod
    def _generate_path_to_save_synthetic_data():
        synthetic_data_path = os.path.join(LOG_DIR, WANDB_PROJECT, wandb.run.id)
        Path(synthetic_data_path).mkdir(parents=True, exist_ok=True)

        return synthetic_data_path

    @staticmethod
    def _create_preprocessing_args_for_eval(args, categorical_transform):
        """Create preprocessing arguments with specified categorical transform.

        Args:
            args (argparse.Namespace): The arguments parsed from the command line.
            categorical_transform (str): The type of categorical transformation to apply.

        Returns:
            argparse.Namespace: The modified arguments for preprocessing.
        """
        args_eval = argparse.Namespace()

        # Context
        args_eval.dataset = args.dataset
        args_eval.task = "unsupervision"
        args_eval.disable_preprocessing = False

        # Features
        args_eval.categorical_impute = "most_frequent"
        args_eval.numerical_impute = "mean"
        args_eval.categorical_transform = categorical_transform
        args_eval.numerical_transform = "standard"
        args_eval.categorical_as_numerical = False

        return args_eval

    @staticmethod
    def _apply_scalers(df, scaler_list):
        """Apply a list of scalers to a dataframe using functional programming approach."""
        if not scaler_list:
            return df
        return reduce(lambda data, scaler: scaler.transform(data), scaler_list, df)
