import os

from be_great import GReaT as GReaTOfficial

import wandb
from src.tabstruct.common import LOG_DIR, WANDB_PROJECT
from src.tabstruct.common.runtime.error.ManualStopError import ManualStopError

from ...BaseGenerator import BaseConditionalGenerator


class GReaT(BaseConditionalGenerator):

    def __init__(self, args):
        super().__init__(args)

        if args.task not in ["classification", "regression", "unsupervision"]:
            raise ValueError(f"Task {args.task} is not supported by {args.model}")

        if not args.model_specific_preprocessing:
            raise ValueError(f"Model-specific preprocessing is required for {self.name}")

        if args.full_num_features_processed > 50:
            raise ManualStopError("GReaT does not support more than 50 features.")

        self.model = GReaTOfficial(
            llm="distilgpt2",
            experiment_dir=os.path.join(LOG_DIR, WANDB_PROJECT, wandb.run.id, "trainer_great"),
            epochs=args.model_params["optimization"]["epochs"],
            batch_size=64,
            efficient_finetuning="lora",
            # Additional hyperparameters added to the TrainingArguments used by the HuggingFaceLibrary
            fp16=True,
            dataloader_num_workers=4,
        )

    def _fit_model(self, data_df):
        # Fit the model
        self.model.fit(
            data=data_df,
            column_names=data_df.columns.tolist(),
            conditional_col=self.cond_for_fit,
        )

    def _prepare_cond(self, data_df):
        cond_for_fit = None
        cond_for_generation = None

        if self.args.task != "unsupervision":
            cond_for_fit = self.args.full_target_col_processed
            cond_for_generation = data_df[self.args.full_target_col_processed].tolist()

        return {
            "cond_for_fit": cond_for_fit,
            "cond_for_generation": cond_for_generation,
        }

    def _generate_model(self, generation_config: dict):
        data_syn_temp_df = self.model.sample(
            n_samples=generation_config["num_samples_to_generate"],
            start_col=generation_config["cond_for_fit"],
            start_col_dist=generation_config["cond_for_generation"],
        )

        return data_syn_temp_df

    def _update_generation_config(self, generation_config: dict) -> dict:
        cond_for_generation = self._update_generation_cond(generation_config["class_id"])
        generation_config = {
            "num_samples_to_generate": generation_config["num_samples_to_generate"],
            "cond_for_fit": self.cond_for_fit,
            "cond_for_generation": cond_for_generation,
        }

        return generation_config

    def _update_generation_cond(self, class_id):
        cond_for_generation = None
        if self.args.task == "classification":
            cond_for_generation = {
                class_id: 1,
            }
        elif self.args.task == "regression":
            cond_for_generation = self.cond_for_generation

        return cond_for_generation

    @classmethod
    def _define_default_params(cls):
        params_arch = {}

        params_optim = {
            "epochs": 100,
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_optuna_params(cls, trial):
        params_arch = {}

        params_optim = {
            "epochs": trial.suggest_int("epochs", 50, 500),
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_single_run_params(cls):
        params_arch = {}

        params_optim = {
            "epochs": 5,
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_test_params(cls):
        params_arch = {}

        params_optim = {
            "epochs": 1,
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _get_model_specific_scaler_config(cls):
        scaler_config_dict = {
            "context": {
                "disable_preprocessing": True,
            },
            "feature_scaler": {},
            "target_scaler": {},
        }

        return scaler_config_dict
