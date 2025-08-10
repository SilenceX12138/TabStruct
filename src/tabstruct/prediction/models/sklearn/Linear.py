from sklearn.linear_model import LinearRegression, LogisticRegression

from ..BasePredictor import BaseSklearnPredictor


class LinearModel(BaseSklearnPredictor):

    def __init__(self, args):
        super().__init__(args)

        if args.task == "regression":
            self.model = LinearRegression()
        elif args.task == "classification":
            self.model = LogisticRegression(**args.model_params)
        else:
            raise ValueError(f"Task {args.task} is not supported by {args.model}")

    @classmethod
    def _define_default_params(cls):
        params = {
            "C": 1.0,
        }
        return params

    @classmethod
    def _define_optuna_params(cls, trial):
        params = {
            "C": trial.suggest_float("C", 0.1, 10.0, log=True),
        }
        return params

    @classmethod
    def _define_single_run_params(cls):
        params = {}
        return params

    @classmethod
    def _define_test_params(cls):
        params = {
            "C": 1.1,
        }
        return params
