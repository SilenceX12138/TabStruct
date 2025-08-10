import os
from pathlib import Path

import numpy as np
import wandb
from tabcamel.data.dataset import TabularDataset

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
            case "scm":
                from .synth.SCM import SCM

                model_class = SCM
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
                from .synth.GReaT import GReaT

                model_class = GReaT
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
            generation_dict = cls.inference(data_module, model)
            # Save synthetic samples
            synthetic_data_path = cls.save_synthetic_data(args, generation_dict)

        # === Compute metrics ===
        metric_dict = {}
        if not args.generation_only:
            if args.model == "real":
                generation_dict = None
            else:
                generation_dict = cls.load_synthetic_data(args, synthetic_data_path)
            metric_dict = compute_all_metrics(args, data_module, generation_dict)

        # === Custom logging ===
        # Record the computation cost
        cls.log_computation_cost(data_module, model)

        return metric_dict

    @classmethod
    def _inference(cls, data_module, model):
        generation_dict = model.generate()

        return generation_dict

    # ================================================================
    # =                                                              =
    # =                        Data Ops                              =
    # =                                                              =
    # ================================================================
    @classmethod
    def save_synthetic_data(cls, args, generation_dict):
        # === Convert the synthetic samples to DataFrame ===
        X_syn = generation_dict["X_syn"]
        y_syn = generation_dict["y_syn"]
        original_data_dict = DataHelper.recover_original_data(args, X_syn, y_syn)

        # === Save the synthetic samples ===
        synthetic_samples = original_data_dict["X_original"].copy(deep=True)
        synthetic_samples[args.full_target_col_original] = original_data_dict["y_original"]
        synthetic_data_dir = cls.generate_path_to_save_synthetic_data()
        synthetic_data_path = os.path.join(synthetic_data_dir, "synthetic_samples.csv")
        FileHelper.save_to_csv_file(synthetic_samples, synthetic_data_path)
        wandb.run.summary["generated_data_path"] = synthetic_data_path

        return synthetic_data_path

    @staticmethod
    def load_synthetic_data(args, synthetic_data_path):
        # === Load the synthetic samples ===
        # Ensure the synthetic data goes through all meta-processing and preprocessing steps
        synthetic_dataset = TabularDataset(
            dataset_name=synthetic_data_path,
            task_type=args.task,
            target_col=args.full_target_col_original,
            metafeature_dict={
                "col2type": args.full_feature_col2type_original,
            },
        )
        # Subsample the synthetic data to align with data curation
        sample_dict = synthetic_dataset.sample(
            sample_mode=args.split_mode,
            sample_size=int(args.curate_ratio * args.train_num_samples_split),
        )
        synthetic_dataset = sample_dict["dataset_sampled"]
        synthetic_data_df = synthetic_dataset.data_df
        X_syn_original = synthetic_data_df[args.full_feature_col_list_original]
        y_syn_original = synthetic_data_df[[args.full_target_col_original]]
        X_syn = X_syn_original.copy(deep=True)
        y_syn = y_syn_original.copy(deep=True)

        # === Preprocess the synthetic samples with scalers fitted on the real training data ===
        for feature_scaler in args.feature_scaler_list:
            X_syn = feature_scaler.transform(X_syn)
        for target_scaler in args.target_scaler_list:
            y_syn = target_scaler.transform(y_syn)
        X_syn = X_syn[args.full_feature_col_list_processed].to_numpy().astype(np.float32)
        y_syn = y_syn[args.full_target_col_processed].to_numpy()

        # === Return the processed synthetic samples ===
        return {
            "X_syn": X_syn,
            "y_syn": y_syn,
            "X_syn_original": X_syn_original,
            "y_syn_original": y_syn_original,
        }

    @staticmethod
    def generate_path_to_save_synthetic_data():
        synthetic_data_path = os.path.join(LOG_DIR, WANDB_PROJECT, wandb.run.id)
        Path(synthetic_data_path).mkdir(parents=True, exist_ok=True)

        return synthetic_data_path
