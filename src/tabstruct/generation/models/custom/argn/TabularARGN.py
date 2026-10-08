import pandas as pd
from mostlyai.sdk import MostlyAI

from ...BaseGenerator import BaseConditionalGenerator


class TabularARGN(BaseConditionalGenerator):

    def __init__(self, args):
        super().__init__(args)

        if args.task not in ["classification", "regression", "unsupervision"]:
            raise ValueError(f"Task {args.task} is not supported by {self.name}")

        if not args.model_specific_preprocessing:
            raise ValueError(f"Model-specific preprocessing is required for {self.name}")

        self.model_class = MostlyAI(local=True)

    def _fit_model(self, data_df):
        self.model = self.model_class.train(
            config={
                "tables": [
                    {
                        "name": "data",
                        "data": data_df,
                        "enable_data_report": False,
                    }
                ],
                "random_state": self.args.seed,
            }
        )

    def _prepare_cond(self, data_df):
        cond_for_fit = None
        cond_for_generation = None

        if self.args.task != "unsupervision":
            # TabularARGN takes seed_df as the generation condition
            cond_for_generation = data_df[[self.args.full_target_col_processed]]

        return {
            "cond_for_fit": cond_for_fit,
            "cond_for_generation": cond_for_generation,
        }

    def _generate_model(self, generation_config: dict):
        synthetic_data = self.model_class.generate(
            generator=self.model,
            config={
                "tables": [
                    {
                        "name": "data",
                        "configuration": {
                            "sample_size": generation_config["num_samples_to_generate"],
                            "enable_data_report": False,
                        },
                    }
                ],
                "random_state": self._next_generation_seed(),
            },
            seed=generation_config["cond_for_generation"],
        )

        return synthetic_data.data()

    def _update_generation_config(self, generation_config: dict) -> dict:
        cond_for_generation = self._update_generation_cond(
            generation_config["class_id"],
            generation_config["num_samples_to_generate"],
        )
        generation_config = {
            "num_samples_to_generate": generation_config["num_samples_to_generate"],
            "cond_for_generation": cond_for_generation,
        }

        return generation_config

    def _update_generation_cond(self, class_id, num_samples_to_generate):
        cond_for_generation = self.cond_for_generation
        if self.args.task == "classification":
            cond_for_generation = [class_id] * num_samples_to_generate
            cond_for_generation = pd.DataFrame({self.args.full_target_col_processed: cond_for_generation})
        elif self.args.task == "regression":
            cond_for_generation = self.cond_for_generation.sample(
                n=num_samples_to_generate,
                replace=True,
            ).reset_index(drop=True)

        return cond_for_generation

    @classmethod
    def _define_default_params(cls):
        params_arch = {}

        params_optim = {}

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_optuna_params(cls, trial):
        params_arch = {}

        params_optim = {}

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_single_run_params(cls):
        params_arch = {}

        params_optim = {}

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_test_params(cls):
        params_arch = {}

        params_optim = {}

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
