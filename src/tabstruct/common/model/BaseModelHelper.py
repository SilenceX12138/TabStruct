import os
from abc import abstractmethod
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from thop import profile

from src.tabstruct.common.data.DataHelper import DataHelper
from src.tabstruct.common.runtime.config.argument import fix_tentative_training_args
from src.tabstruct.common.runtime.log.WandbHelper import WandbHelper

from .. import LOG_DIR, WANDB_PROJECT
from ..runtime.log.FileHelper import FileHelper
from ..runtime.log.TerminalIO import TerminalIO


class BaseModelHelper:
    """Helper class for training and evaluating models."""

    # The models that do not need to be saved
    non_trainable_model_list = ["knn", "smote", "tabebm"]

    # ================================================================
    # =                                                              =
    # =               Top-level APIs (class methods)                 =
    # =                                                              =
    # ================================================================
    @classmethod
    def benchmark_model(cls, args):
        """Benchmark the model.

        Args:
            args (Namespace): The arguments for the experiment.

        Returns:
            dict: The training, validation, and test metrics.
        """
        # === Prepare data and fix tentative arguments ===
        data_module = DataHelper.create_data_module(args)
        fix_tentative_training_args(args)

        # === Fit the model ===
        model = cls.fit_model(args, data_module)

        # === Unified evaluationg of all possible kinds of models ===
        metric_dict = cls.eval_model(args, data_module, model)

        return metric_dict

    @classmethod
    @WandbHelper.trace_time(summary_key="computation/fit_duration")
    @TerminalIO.trace_func
    def fit_model(cls, args, data_module):
        """Fit the model.

        Args:
            args (Namespace): The arguments for the experiment.
            data_module (DataModule): The data module.

        Returns:
            BaseModel: The fitted model.
        """
        # === No need to fit the generative model if synthetic data is provided ===
        if args.pipeline == "generation" and args.synthetic_data_path is not None:
            return None

        # === Fit the model ===
        # Prepare the model
        # Try loading checkpoint except those who store all data
        if args.saved_checkpoint_path and args.model not in cls.non_trainable_model_list:
            model = cls.load_model(args.saved_checkpoint_path)
            # Note that the args in model is deprecated and thus we reload the new args
            model.args = args
        else:
            model = cls.create_model(args)

        # Get fitted model for evaluation
        if not args.eval_only or args.model in cls.non_trainable_model_list:
            model.fit(data_module)

        # Save the model (except those who stores all data)
        if args.save_model:
            cls.save_model(model)

        return model

    @classmethod
    @WandbHelper.trace_time(summary_key="computation/create_duration")
    def create_model(cls, args):
        """Create the model.

        Args:
            args (Namespace): The arguments for the experiment.

        Returns:
            BaseModel: The model instance.
        """
        # === Select the model class ===
        model_class = cls._model_handler(args.model)

        # === Define the model parameters ===
        # Only define the model parameters when it is not provided
        if not hasattr(args, "model_params"):
            args.model_params = model_class.define_params(reg_test=args.reg_test, dev="dev" in args.tags)

        # === Create the model ===
        return model_class(args)

    @classmethod
    def model_handler(cls, model):
        """Handle the model selection.

        Args:
            model (str): model name.

        Raises:
            NotImplementedError: If the custom model handler is not implemented.
        """
        model_class = cls._model_handler(model)

        return model_class

    @classmethod
    @WandbHelper.trace_time(summary_key="computation/eval_duration")
    @TerminalIO.trace_func
    def eval_model(cls, args, data_module, model):
        """Evaluate a scikit-learn style model.

        Args:
            args (argparse): The arguments for the experiment.
            data_module (dataset.DatasetModule): The dataset module.
            model: The trained model.

        Raises:
            NotImplementedError: If the custom analysis model is not implemented.

        Returns:
            dict: The training, validation, and test metrics.
        """
        metric_dict = cls._eval_model(args, data_module, model)

        return metric_dict

    @classmethod
    @WandbHelper.trace_time(summary_key="computation/inference_duration")
    def inference(cls, data_module, model):
        """Inference of the model.

        Args:
            data_module (DataModule): The data module.
            model: The trained model.

        Returns:
            dict: The inference results.
        """
        inference_dict = cls._inference(data_module, model)

        return inference_dict

    @classmethod
    def log_computation_cost(cls, data_module, model):
        """Log the computation cost.

        Args:
            data_module (DataModule): The data module.
            model: The trained model.
        """
        # Record the GPU usage if available
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.max_memory_reserved(0) / 1e9
            wandb.run.summary["computation/gpu_memory_used (GB)"] = gpu_memory_used

        # Record the model size and FLOPs
        if isinstance(model, nn.Module):
            flops, params = profile(
                model, inputs=torch.tensor(data_module.X_train[[[0]]], dtype=torch.float32, device=model.device)
            )
            wandb.run.summary["computation/FLOPs"] = flops
            wandb.run.summary["computation/total_params"] = params

    @classmethod
    def save_model(cls, model):
        """Save the model to disk. Default to use pkl file."""
        model_dir = cls.generate_path_to_save()
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        model_path = os.path.join(model_dir, f"{model.args.model}.pkl")

        # Save appropriate content based on model type
        content = (
            "Non-trainable model, thus no checkpoint is saved."
            if model.args.model in cls.non_trainable_model_list
            else model
        )
        FileHelper.save_to_pickle_file(content, model_path)

        wandb.run.summary["best_model_path"] = model_path
        return model_path

    @classmethod
    def load_model(cls, checkpoint_path):
        """Load the model from a checkpoint. Default to use pkl file."""
        model = FileHelper.load_from_pickle_file(checkpoint_path)

        return model

    @classmethod
    def generate_path_to_save(cls):
        return os.path.join(LOG_DIR, WANDB_PROJECT, wandb.run.id)

    # ================================================================
    # =                                                              =
    # =                            Utils                             =
    # =                                                              =
    # ================================================================
    @classmethod
    @abstractmethod
    def _model_handler(cls, model):
        """Handle the model selection.

        Args:
            model (str): model name.

        Raises:
            NotImplementedError: If the custom model handler is not implemented.
        """
        raise NotImplementedError("This method has to be implemented by the sub class")

    @classmethod
    @abstractmethod
    def _eval_model(cls, args, data_module, model):
        """Evaluate a scikit-learn style model.

        Args:
            args (argparse): The arguments for the experiment.
            data_module (dataset.DatasetModule): The dataset module.
            model: The trained model.

        Raises:
            NotImplementedError: If the custom analysis model is not implemented.

        Returns:
            dict: The training, validation, and test metrics.
        """
        raise NotImplementedError("This method has to be implemented by the sub class")

    @classmethod
    @abstractmethod
    def _inference(cls, data_module, model):
        """Inference of the model.

        Args:
            data_module (DatasetModule): The data module.
            model: The trained model.

        Raises:
            NotImplementedError: If the custom inference model is not implemented.
        """
        raise NotImplementedError("This method has to be implemented by the sub class")
