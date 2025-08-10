from tabeval.plugins.generic.plugin_great import GReaTPlugin

from src.tabstruct.common.runtime.error.ManualStopError import ManualStopError

from ..BaseGenerator import BaseTabEvalConditionalGenerator


class GReaT(BaseTabEvalConditionalGenerator):

    def __init__(self, args):
        super().__init__(args)

        if args.full_num_features_processed > 50:
            raise ManualStopError("GReaT does not support more than 50 features.")

        self.model = GReaTPlugin(
            # Architecture
            # Optimization
            n_iter=args.model_params["optimization"]["n_iter"],
            # Misc.
            sampling_patience=1,
            strict=True,
        )

    @classmethod
    def _define_default_params(cls):
        params_arch = {}

        params_optim = {
            "n_iter": 5,
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_optuna_params(cls, trial):
        params_arch = {}

        params_optim = {
            "n_iter": trial.suggest_int("n_iter", 50, 500),
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_single_run_params(cls):
        params_arch = {}

        params_optim = {
            "n_iter": 5,
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
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }
