import numpy as np
import torch
from tabebm.TabEBM import TabEBM as TabEBMOfficial
from tqdm.auto import tqdm

from src.tabstruct.common.runtime.error.ManualStopError import ManualStopError

from ...BaseGenerator import BaseClassFocusedGenerator


class TabEBM(BaseClassFocusedGenerator):

    def __init__(self, args):
        super().__init__(args)

        if args.task not in ["classification", "regression", "unsupervision"]:
            raise ValueError(f"Task {args.task} is not supported by {args.model}")

        if args.train_num_samples_processed > 10000:
            raise ManualStopError("TabEBM does not support more than 10000 samples.")

        if args.full_num_features_processed > 500:
            raise ManualStopError("TabEBM does not support more than 500 features.")

        self.model = TabEBMOfficial()

    def _fit_model(self, data_df):
        pass

    def _generate_model(self, class2synthetic_samples):
        # === Prepare the data and generatation configurations ===
        if self.args.task in ["regression", "unsupervision"]:
            class2synthetic_samples = {
                0: class2synthetic_samples["real"],
                1: class2synthetic_samples["dummy"],
            }
        num_synthetic_samples_max = max(class2synthetic_samples.values())

        # === Generate the synthetic data ===
        current_synthetic_samples = 0
        synthetic_data_dict = {}
        progress_bar = tqdm(
            total=num_synthetic_samples_max,
            desc="Generating synthetic samples",
            unit="samples",
            leave=True,
        )
        try:
            while current_synthetic_samples < num_synthetic_samples_max:
                # If num_synthetic_samples_max is too large, we limit the number of samples to generate per iteration
                num_samples_to_generate = num_synthetic_samples_max - current_synthetic_samples
                num_samples_to_generate = min(num_samples_to_generate, 1000)
                with torch.enable_grad():
                    synthetic_data_temp = self.model.generate(
                        X=self.X_train,
                        y=self.y_train,
                        num_samples=num_samples_to_generate,
                        sgld_noise_std=self.args.model_params["sgld_noise_std"],
                        sgld_steps=self.args.model_params["sgld_steps"],
                    )
                for class_id in synthetic_data_temp.keys():
                    new_samples = synthetic_data_temp[class_id][:num_samples_to_generate]
                    if class_id not in synthetic_data_dict:
                        synthetic_data_dict[class_id] = new_samples
                    else:
                        synthetic_data_dict[class_id] = np.concatenate(
                            [synthetic_data_dict[class_id], new_samples], axis=0
                        )
                current_synthetic_samples += num_samples_to_generate
                progress_bar.update(num_samples_to_generate)
        finally:
            if progress_bar.n < progress_bar.total:
                progress_bar.update(progress_bar.total - progress_bar.n)
            progress_bar.close()

        # === Format the synthetic data for return ===
        if self.args.task == "classification":
            X_syn = np.concatenate(
                [
                    synthetic_data_dict[f"class_{class_id}"][
                        np.random.choice(synthetic_data_dict[f"class_{class_id}"].shape[0], num_synthetic_samples)
                    ]
                    for class_id, num_synthetic_samples in class2synthetic_samples.items()
                ],
                axis=0,
            )
            y_syn = np.concatenate(
                [
                    [class_id] * num_synthetic_samples
                    for class_id, num_synthetic_samples in class2synthetic_samples.items()
                ]
            ).reshape(-1)
        elif self.args.task == "regression":
            X_syn = synthetic_data_dict["class_0"]
            X_syn = X_syn[:, :-1]
            y_syn = X_syn[:, -1]
        elif self.args.task == "unsupervision":
            X_syn = synthetic_data_dict["class_0"]
            y_syn = None
        else:
            raise ValueError(f"Task {self.args.task} is not supported by {self.args.model}")

        return {
            "X_syn": X_syn,
            "y_syn": y_syn,
        }

    @classmethod
    def _define_default_params(cls):
        params = {
            "sgld_steps": 100,
            "sgld_noise_std": 0.04,
        }

        return params

    @classmethod
    def _define_optuna_params(cls, trial):
        params = {}

        return params

    @classmethod
    def _define_single_run_params(cls):
        params = {
            "sgld_steps": 200,
            "sgld_noise_std": 0.04,
        }

        return params

    @classmethod
    def _define_test_params(cls):
        params = {
            "sgld_steps": 1,
            "sgld_noise_std": 0.04,
        }

        return params
