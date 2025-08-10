from tabeval.plugins.generic.plugin_ddpm import TabDDPMPlugin

from ..BaseGenerator import BaseTabEvalConditionalGenerator


class TabDDPM(BaseTabEvalConditionalGenerator):

    def __init__(self, args):
        super().__init__(args)

        self.model = TabDDPMPlugin(
            # Architecture
            # Optimization
            n_iter=args.model_params["optimization"]["n_iter"],
            lr=args.model_params["optimization"]["lr"],
            weight_decay=args.model_params["optimization"]["weight_decay"],
            num_timesteps=args.model_params["optimization"]["num_timesteps"],
            # Misc
            is_classification=(args.task == "classification"),
            strict=False,
        )

    @classmethod
    def _define_default_params(cls):
        params_arch = {}

        params_optim = {
            "n_iter": 1000,
            "lr": 2e-3,
            "weight_decay": 1e-4,
            "num_timesteps": 1e3,
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_optuna_params(cls, trial):
        params_arch = {}

        params_optim = {
            "n_iter": trial.suggest_int("n_iter", 1e3, 1e4),
            "lr": trial.suggest_float("lr", 1e-5, 1e-1, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-3, log=True),
            "num_timesteps": trial.suggest_int("num_timesteps", 10, 1e3),
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_single_run_params(cls):
        params_arch = {}

        params_optim = {
            "n_iter": 1,
            "lr": 2e-3,
            "weight_decay": 1e-4,
            "num_timesteps": 1e3,
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_test_params(cls):
        params_arch = {}

        params_optim = {
            "n_iter": 1,
            "lr": 2e-3,
            "weight_decay": 1e-4,
            "num_timesteps": 1e3,
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }
