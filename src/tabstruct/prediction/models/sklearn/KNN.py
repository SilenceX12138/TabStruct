from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from ..BasePredictor import BaseSklearnPredictor


class KNN(BaseSklearnPredictor):

    def __init__(self, args):
        super().__init__(args)

        if args.task == "regression":
            self.model = KNeighborsRegressor(**args.model_params)
        elif args.task == "classification":
            self.model = KNeighborsClassifier(**args.model_params)
        else:
            raise ValueError(f"Task {args.task} is not supported by {args.model}")

    @classmethod
    def _define_default_params(cls):
        params = {
            "n_neighbors": 5,
            "algorithm": "auto",
            "leaf_size": 30,
        }
        return params

    @classmethod
    def _define_optuna_params(cls, trial):
        params = {
            "n_neighbors": trial.suggest_categorical("n_neighbors", list(range(3, 42, 2))),
            "algorithm": trial.suggest_categorical("algorithm", ["auto", "kd_tree", "ball_tree"]),
            "leaf_size": trial.suggest_int("leaf_size", 30, 300, log=True),
        }
        return params

    @classmethod
    def _define_single_run_params(cls):
        params = {
            "n_neighbors": 5,
            "algorithm": "auto",
            "leaf_size": 30,
        }
        return params

    @classmethod
    def _define_test_params(cls):
        params = {
            "n_neighbors": 3,
        }
        return params
