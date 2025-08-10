import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

from ..BasePredictor import BaseSklearnPredictor
from ..utils.evaluation import gate_bin2dec


class RandomForest(BaseSklearnPredictor):

    def __init__(self, args):
        super().__init__(args)

        if args.task == "regression":
            self.model = RandomForestRegressor(
                **args.model_params,
                random_state=args.seed,
            )
        elif args.task == "classification":
            self.model = RandomForestClassifier(
                **args.model_params,
                class_weight=args.train_class2weight,
                random_state=args.seed,
            )
        else:
            raise ValueError(f"Task {args.task} is not supported by {args.model}")

        self.model = SelectFromModel(self.model)

    def predict(self, X):
        y_pred = self.model.estimator_.predict(X)

        return y_pred

    def predict_proba(self, X):
        if self.args.task == "classification":
            y_hat = self.model.estimator_.predict_proba(X)
        else:
            y_hat = None

        return y_hat

    def _feature_selection(self, X=None):
        gate = np.asarray(self.model.get_support(), dtype=int)
        gate_all = gate_bin2dec(gate)
        num_selected_features = gate.sum()

        return [gate_all], num_selected_features

    @classmethod
    def _define_default_params(cls):
        params = {
            "n_estimators": 100,
            "max_depth": 5,
        }
        return params

    @classmethod
    def _define_optuna_params(cls, trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 5, 100, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 12, log=True),
        }
        return params

    @classmethod
    def _define_single_run_params(cls):
        params = {
            "n_estimators": 50,
            "max_depth": 5,
        }
        return params

    @classmethod
    def _define_test_params(cls):
        params = {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
        }
        return params
