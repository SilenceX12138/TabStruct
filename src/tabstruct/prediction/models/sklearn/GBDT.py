import numpy as np
import xgboost as xgb

from ..BasePredictor import BaseSklearnPredictor


class XGBoost(BaseSklearnPredictor):

    def __init__(self, args):
        super().__init__(args)

        self.params["device"] = args.device

        if args.task == "regression":
            self.params["objective"] = "reg:squarederror"
            self.params["eval_metric"] = "rmse"
        elif args.task == "classification":
            self.params["objective"] = "multi:softprob"
            self.params["num_class"] = args.full_num_classes_processed
            self.params["eval_metric"] = "mlogloss"
        else:
            raise ValueError(f"Task {args.task} is not supported by {args.model}")

    def fit(self, data_module):
        X_train = data_module.X_train
        y_train = data_module.y_train
        X_val = data_module.X_valid
        y_val = data_module.y_valid

        train = xgb.DMatrix(X_train, label=y_train)
        val = xgb.DMatrix(X_val, label=y_val)
        eval_list = [(val, "eval")]
        self.model = xgb.train(
            params=self.params,
            dtrain=train,
            num_boost_round=100,
            evals=eval_list,
            early_stopping_rounds=5,
        )

    def predict(self, X):
        # For classification, "Booster" outputs probabilities rather than specific labels
        y_hat = self.predict_proba(X)
        if self.args.task == "classification":
            y_pred = np.argmax(y_hat, axis=1)
        else:
            y_pred = y_hat

        return y_pred

    def predict_proba(self, X):
        X = xgb.DMatrix(X)
        return self.model.predict(X)

    @classmethod
    def _define_default_params(cls):
        params = {
            "max_depth": 5,
            "alpha": 1e-4,
            "lambda": 1e-4,
            "eta": 0.08,
        }
        return params

    @classmethod
    def _define_optuna_params(cls, trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 12, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
        }
        return params

    @classmethod
    def _define_single_run_params(cls):
        params = {
            "max_depth": 5,
            "alpha": 1e-4,
            "lambda": 1e-4,
            "eta": 0.08,
        }
        return params

    @classmethod
    def _define_test_params(cls):
        params = {
            "max_depth": 5,
            "alpha": 1e-4,
            "lambda": 1e-4,
            "eta": 0.08,
        }
        return params
