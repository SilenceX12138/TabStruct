import numpy as np
from imblearn.over_sampling import SMOTE as ImlearnSMOTE

from src.tabstruct.common.runtime.error.ManualStopError import ManualStopError

from ..BaseGenerator import BaseClassFocusedGenerator


class SMOTE(BaseClassFocusedGenerator):

    def __init__(self, args):
        super().__init__(args)

        if args.task not in ["classification", "regression", "unsupervision"]:
            raise ValueError(f"Task {args.task} is not supported by {args.model}")

    def _fit_model(self, data_df):
        # === Check if number of neighbors is valid ===
        if self.args.task == "classification":
            min_num_samples_per_class = min(self.args.train_class2samples_processed.values())
            if self.args.model_params["k_neighbors"] > min_num_samples_per_class - 1:
                # Optuna will not proceed on the same split if an exception is raised
                # i.e., Optuna will move on to the next split
                raise ManualStopError(
                    f"Number of neighbors ({self.k_neighbors}) is too high for some classes (min: {min_num_samples_per_class})"
                )

    def _generate_model(self, class2synthetic_samples):
        # === Prepare the data and generatation configurations ===
        X_train = self.X_train
        y_train = self.y_train
        if self.args.task in ["regression", "unsupervision"]:
            # The samples are sorted by class id (0->real data, 1->dummy data)
            X_dummy = np.random.rand(X_train.shape[0], X_train.shape[1])
            y_dummy = np.ones(self.X_train.shape[0], dtype=np.int64)
            X_train = np.concatenate([X_train, X_dummy], axis=0)
            y_train = np.concatenate([y_train, y_dummy], axis=0)

            # Add number of synthetic samples to the dict
            class2total_samples = {
                0: self.X_train.shape[0] + class2synthetic_samples["real"],
                1: X_dummy.shape[0] + class2synthetic_samples["dummy"],
            }
        else:
            # Add number of real samples to the dict
            class2total_samples = {
                class_id: num_synthetic_samples + self.args.train_class2samples_processed[class_id]
                for (class_id, num_synthetic_samples) in class2synthetic_samples.items()
            }

        # === Fit the SMOTE model and generate synthetic samples ===
        self.model = ImlearnSMOTE(
            sampling_strategy=class2total_samples,
            k_neighbors=self.args.model_params["k_neighbors"],
            # random_state is not needed if we set seed in the main script
            # random_state=self.args.seed,
        )
        X_syn_full, y_syn_full = self.model.fit_resample(X_train, y_train)

        # === Drop the original data ===
        num_train_samples = X_train.shape[0]
        if self.args.task == "regression":
            X_syn = X_syn_full[num_train_samples:, :-1]
            y_syn = X_syn_full[num_train_samples:, -1]
        elif self.args.task == "unsupervision":
            X_syn = X_syn_full[num_train_samples:]
            y_syn = None
        else:
            X_syn = X_syn_full[num_train_samples:]
            y_syn = y_syn_full[num_train_samples:]

        return {
            "X_syn": X_syn,
            "y_syn": y_syn,
        }

    @classmethod
    def _define_default_params(cls):
        params = {
            "k_neighbors": 5,
        }

        return params

    @classmethod
    def _define_optuna_params(cls, trial):
        params = {
            "k_neighbors": trial.suggest_categorical("k_neighbors", list(range(3, 42, 2))),
        }

        return params

    @classmethod
    def _define_single_run_params(cls):
        params = {
            "k_neighbors": 5,
        }

        return params

    @classmethod
    def _define_test_params(cls):
        params = {
            "k_neighbors": 3,
        }

        return params
