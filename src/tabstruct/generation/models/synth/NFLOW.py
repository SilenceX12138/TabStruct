from tabeval.plugins.generic.plugin_nflow import NormalizingFlowsPlugin

from ..BaseGenerator import BaseTabEvalJointGenerator


class NFLOW(BaseTabEvalJointGenerator):

    def __init__(self, args):
        super().__init__(args)

        self.model = NormalizingFlowsPlugin(
            # Architecture
            n_layers_hidden=args.model_params["architecture"]["n_layers_hidden"],
            n_units_hidden=args.model_params["architecture"]["n_units_hidden"],
            linear_transform_type=args.model_params["architecture"]["linear_transform_type"],
            base_transform_type=args.model_params["architecture"]["base_transform_type"],
            dropout=args.model_params["architecture"]["dropout"],
            batch_norm=args.model_params["architecture"]["batch_norm"],
            # Optimization
            lr=args.model_params["optimization"]["lr"],
            n_iter=args.model_params["optimization"]["n_iter"],
        )

    @classmethod
    def _define_default_params(cls):
        params_arch = {
            "n_layers_hidden": 1,
            "n_units_hidden": 100,
            "linear_transform_type": "permutation",
            "base_transform_type": "rq-autoregressive",
            "dropout": 0.1,
            "batch_norm": False,
        }

        params_optim = {
            "lr": 1e-3,
            "n_iter": 1000,
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_optuna_params(cls, trial):
        params_arch = {
            "n_layers_hidden": trial.suggest_int("n_layers_hidden", 1, 10),
            "n_units_hidden": trial.suggest_int("n_units_hidden", 10, 100),
            "linear_transform_type": trial.suggest_categorical("linear_transform_type", ["lu", "permutation", "svd"]),
            "base_transform_type": trial.suggest_categorical(
                "base_transform_type",
                [
                    "affine-coupling",
                    "quadratic-coupling",
                    "rq-coupling",
                    "affine-autoregressive",
                    "quadratic-autoregressive",
                    "rq-autoregressive",
                ],
            ),
            "dropout": trial.suggest_float("dropout", 0.0, 0.2),
            "batch_norm": trial.suggest_categorical("batch_norm", [False, True]),
        }

        params_optim = {
            "lr": trial.suggest_float("lr", 2e-4, 1e-3, log=True),
            "n_iter": trial.suggest_int("n_iter", 100, 5000),
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_single_run_params(cls):
        params_arch = {
            "n_layers_hidden": 1,
            "n_units_hidden": 100,
            "linear_transform_type": "permutation",
            "base_transform_type": "rq-autoregressive",
            "dropout": 0.1,
            "batch_norm": False,
        }

        params_optim = {
            "lr": 1e-3,
            "n_iter": 10,
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_test_params(cls):
        params_arch = {
            "n_layers_hidden": 1,
            "n_units_hidden": 100,
            "linear_transform_type": "permutation",
            "base_transform_type": "rq-autoregressive",
            "dropout": 0.1,
            "batch_norm": False,
        }

        params_optim = {
            "lr": 1e-3,
            "n_iter": 10,
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }
