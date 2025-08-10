from sklearn.neural_network import MLPClassifier, MLPRegressor

from ..BasePredictor import BaseSklearnPredictor


class MLPSklearn(BaseSklearnPredictor):

    def __init__(self, args):
        super().__init__(args)

        if args.task == "regression":
            self.model = MLPRegressor(
                **args.model_params,
                max_iter=1000,
                early_stopping=True,
                verbose=True,
            )
        elif args.task == "classification":
            self.model = MLPClassifier(
                **args.model_params,
                max_iter=1000,
                early_stopping=True,
                verbose=True,
            )
        else:
            raise ValueError(f"Task {args.task} is not supported by {args.model}")

    @classmethod
    def _define_default_params(cls):
        params = {
            "learning_rate_init": 0.001,
            "hidden_layer_sizes": [100],
        }
        return params

    @classmethod
    def _define_optuna_params(cls, trial):
        hidden_dim = trial.suggest_int("hidden_dim", 10, 100, log=True)
        n_layers = trial.suggest_int("n_layers", 1, 5)
        params = {
            "learning_rate_init": trial.suggest_float("learning_rate_init", 5e-4, 1e-3, log=True),
            "hidden_layer_sizes": n_layers * [hidden_dim],
        }

        return params

    @classmethod
    def _define_single_run_params(cls):
        params = {
            "learning_rate_init": 0.001,
            "hidden_layer_sizes": [100],
        }
        return params

    @classmethod
    def _define_test_params(cls):
        params = {
            "learning_rate_init": 0.001,
            "hidden_layer_sizes": [100],
        }
        return params
