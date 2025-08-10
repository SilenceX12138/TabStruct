from tabeval.plugins.generic.plugin_ctgan import CTGANPlugin

from ..BaseGenerator import BaseTabEvalConditionalGenerator


class CTGAN(BaseTabEvalConditionalGenerator):

    def __init__(self, args):
        """Note: CTGAN only supports using categorical features as conditioning variables."""
        super().__init__(args)

        self.model = CTGANPlugin(
            # Architecture
            generator_n_layers_hidden=args.model_params["architecture"]["generator_n_layers_hidden"],
            generator_n_units_hidden=args.model_params["architecture"]["generator_n_units_hidden"],
            generator_nonlin=args.model_params["architecture"]["generator_nonlin"],
            discriminator_n_layers_hidden=args.model_params["architecture"]["discriminator_n_layers_hidden"],
            discriminator_n_units_hidden=args.model_params["architecture"]["discriminator_n_units_hidden"],
            discriminator_nonlin=args.model_params["architecture"]["discriminator_nonlin"],
            # Optimization
            n_iter=args.model_params["optimization"]["n_iter"],  # epochs (in tabeval)
            discriminator_n_iter=args.model_params["optimization"]["discriminator_n_iter"],
            lr=args.model_params["optimization"]["lr"],
            weight_decay=args.model_params["optimization"]["weight_decay"],
        )

    @classmethod
    def _define_default_params(cls):
        params_arch = {
            "generator_n_layers_hidden": 2,
            "generator_n_units_hidden": 500,
            "generator_nonlin": "relu",
            "discriminator_n_layers_hidden": 2,
            "discriminator_n_units_hidden": 500,
            "discriminator_nonlin": "leaky_relu",
        }

        params_optim = {
            "n_iter": 100,
            "discriminator_n_iter": 1,
            "lr": 1e-3,
            "weight_decay": 1e-3,
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_optuna_params(cls, trial):
        params_arch = {
            "generator_n_layers_hidden": trial.suggest_int("generator_n_layers_hidden", 1, 4),
            "generator_n_units_hidden": trial.suggest_int("generator_n_units_hidden", 50, 150),
            "generator_nonlin": trial.suggest_categorical("generator_nonlin", ["relu", "leaky_relu", "tanh", "elu"]),
            "discriminator_n_layers_hidden": trial.suggest_int("discriminator_n_layers_hidden", 1, 4),
            "discriminator_n_units_hidden": trial.suggest_int("discriminator_n_units_hidden", 50, 150),
            "discriminator_nonlin": trial.suggest_categorical(
                "discriminator_nonlin", ["relu", "leaky_relu", "tanh", "elu"]
            ),
        }

        params_optim = {
            "n_iter": trial.suggest_int("n_iter", 100, 1000),
            "discriminator_n_iter": trial.suggest_int("discriminator_n_iter", 1, 5),
            "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-3, log=True),
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_single_run_params(cls):
        params_arch = {
            "generator_n_layers_hidden": 2,
            "generator_n_units_hidden": 500,
            "generator_nonlin": "relu",
            "discriminator_n_layers_hidden": 2,
            "discriminator_n_units_hidden": 500,
            "discriminator_nonlin": "leaky_relu",
        }

        params_optim = {
            "n_iter": 10,
            "discriminator_n_iter": 1,
            "lr": 1e-3,
            "weight_decay": 1e-3,
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_test_params(cls):
        params_arch = {
            "generator_n_layers_hidden": 2,
            "generator_n_units_hidden": 500,
            "generator_nonlin": "relu",
            "discriminator_n_layers_hidden": 2,
            "discriminator_n_units_hidden": 500,
            "discriminator_nonlin": "leaky_relu",
        }

        params_optim = {
            "n_iter": 1,
            "discriminator_n_iter": 1,
            "lr": 1e-3,
            "weight_decay": 1e-3,
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }
