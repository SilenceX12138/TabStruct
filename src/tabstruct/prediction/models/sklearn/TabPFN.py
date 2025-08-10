import numpy as np
from tabpfn import TabPFNClassifier, TabPFNRegressor

from src.tabstruct.common.runtime.error.ManualStopError import ManualStopError

from ..BasePredictor import BaseSklearnPredictor


class TabPFN(BaseSklearnPredictor):

    def __init__(self, args):
        super().__init__(args)

        if args.task == "classification" and args.full_num_classes_processed > 10:
            raise ManualStopError("TabPFN does not support more than 10 classes.")
        if args.full_num_features_processed > 500:
            raise ManualStopError("TabPFN does not support more than 500 features.")

        if args.task == "regression":
            self.model = TabPFNRegressor(
                **args.model_params,
                ignore_pretraining_limits=True,
            )
        elif args.task == "classification":
            self.model = TabPFNClassifier(
                **args.model_params,
                ignore_pretraining_limits=True,
            )

    def fit(self, data_module):
        if data_module.X_train.shape[0] > 10000:
            subsample_index = np.random.choice(
                data_module.X_train.shape[0],
                10000,
                replace=False,
            )
            self.model.fit(data_module.X_train[subsample_index], data_module.y_train[subsample_index])
        else:
            self.model.fit(data_module.X_train, data_module.y_train)

    @classmethod
    def _define_default_params(cls):
        params = {
            "n_estimators": 3,
        }
        return params

    @classmethod
    def _define_optuna_params(cls, trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 1, 5),
        }
        return params

    @classmethod
    def _define_single_run_params(cls):
        params = {
            "n_estimators": 3,
        }
        return params

    @classmethod
    def _define_test_params(cls):
        params = {
            "n_estimators": 3,
        }
        return params
