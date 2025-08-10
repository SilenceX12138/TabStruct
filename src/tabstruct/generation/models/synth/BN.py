from tabeval.plugins.generic.plugin_bayesian_network import BayesianNetworkPlugin

from ..BaseGenerator import BaseTabEvalJointGenerator


class BN(BaseTabEvalJointGenerator):

    def __init__(self, args):
        super().__init__(args)

        self.model = BayesianNetworkPlugin(
            # Architecture
            struct_learning_search_method=args.model_params["architecture"]["struct_learning_search_method"],
            # Optimization
            struct_learning_n_iter=args.model_params["optimization"]["struct_learning_n_iter"],
            struct_learning_score=args.model_params["optimization"]["struct_learning_score"],
            # Misc.
            strict=False,
        )

    @classmethod
    def _define_default_params(cls):
        params_arch = {
            "struct_learning_search_method": "tree_search",
        }

        params_optim = {
            "struct_learning_n_iter": 1000,
            "struct_learning_score": "k2",
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_optuna_params(cls, trial):
        params_arch = {
            "struct_learning_search_method": trial.suggest_categorical(
                "struct_learning_search_method", ["hillclimb", "pc", "tree_search"]
            ),
        }

        params_optim = {
            "struct_learning_n_iter": trial.suggest_int("n_iter", 100, 1000),
            "struct_learning_score": trial.suggest_categorical("struct_learning_score", ["k2", "bdeu", "bic", "bds"]),
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_single_run_params(cls):
        params_arch = {
            "struct_learning_search_method": "tree_search",
        }

        params_optim = {
            "struct_learning_n_iter": 1000,
            "struct_learning_score": "k2",
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_test_params(cls):
        params_arch = {
            "struct_learning_search_method": "tree_search",
        }

        params_optim = {
            "struct_learning_n_iter": 100,
            "struct_learning_score": "k2",
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }
