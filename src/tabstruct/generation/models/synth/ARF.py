from tabeval.plugins.generic.plugin_arf import ARFPlugin

from ..BaseGenerator import BaseTabEvalJointGenerator


class ARF(BaseTabEvalJointGenerator):

    def __init__(self, args):
        super().__init__(args)

        self.model = ARFPlugin(
            # Architecture
            num_trees=args.model_params["architecture"]["num_trees"],
            min_node_size=args.model_params["architecture"]["min_node_size"],
            # Optimization
            max_iters=args.model_params["optimization"]["max_iters"],
            early_stop=args.model_params["optimization"]["early_stop"],
            # Misc.
            strict=False,
        )

    @classmethod
    def _define_default_params(cls):
        params_arch = {
            "num_trees": 30,
            "min_node_size": 5,
        }

        params_optim = {
            "max_iters": 10,
            "early_stop": True,
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_optuna_params(cls, trial):
        params_arch = {
            "num_trees": trial.suggest_int("num_trees", 10, 100),
            "min_node_size": trial.suggest_int("min_node_size", 2, 20),
        }

        params_optim = {
            "max_iters": trial.suggest_int("max_iters", 1, 5),
            "early_stop": trial.suggest_categorical("early_stop", [False, True]),
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_single_run_params(cls):
        params_arch = {
            "num_trees": 30,
            "min_node_size": 5,
        }

        params_optim = {
            "max_iters": 10,
            "early_stop": True,
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_test_params(cls):
        params_arch = {
            "num_trees": 30,
            "min_node_size": 5,
        }

        params_optim = {
            "max_iters": 1,
            "early_stop": True,
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }
