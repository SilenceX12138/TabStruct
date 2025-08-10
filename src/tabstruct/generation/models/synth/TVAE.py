from tabeval.plugins.generic.plugin_tvae import TVAEPlugin

from ..BaseGenerator import BaseTabEvalConditionalGenerator


class TVAE(BaseTabEvalConditionalGenerator):

    def __init__(self, args):
        """Note: TVAE only supports categorical features as conditional variables."""
        super().__init__(args)

        self.model = TVAEPlugin(
            # Architecture
            encoder_n_layers_hidden=args.model_params["architecture"]["encoder_n_layers_hidden"],
            encoder_n_units_hidden=args.model_params["architecture"]["encoder_n_units_hidden"],
            encoder_nonlin=args.model_params["architecture"]["encoder_nonlin"],
            n_units_embedding=args.model_params["architecture"]["n_units_embedding"],
            decoder_n_layers_hidden=args.model_params["architecture"]["decoder_n_layers_hidden"],
            decoder_n_units_hidden=args.model_params["architecture"]["decoder_n_units_hidden"],
            decoder_nonlin=args.model_params["architecture"]["decoder_nonlin"],
            # Optimization
            n_iter=args.model_params["optimization"]["n_iter"],
            lr=args.model_params["optimization"]["lr"],
            weight_decay=args.model_params["optimization"]["weight_decay"],
        )

    @classmethod
    def _define_default_params(cls):
        params_arch = {
            "encoder_n_layers_hidden": 3,
            "encoder_n_units_hidden": 500,
            "encoder_nonlin": "leaky_relu",
            "n_units_embedding": 500,
            "decoder_n_layers_hidden": 3,
            "decoder_n_units_hidden": 500,
            "decoder_nonlin": "leaky_relu",
        }

        params_optim = {
            "n_iter": 100,
            "lr": 1e-3,
            "weight_decay": 1e-5,
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_optuna_params(cls, trial):
        params_arch = {
            "encoder_n_layers_hidden": trial.suggest_int("encoder_n_layers_hidden", 1, 5),
            "encoder_n_units_hidden": trial.suggest_int("encoder_n_units_hidden", 50, 500),
            "encoder_nonlin": trial.suggest_categorical("encoder_nonlin", ["relu", "leaky_relu", "tanh", "elu"]),
            "n_units_embedding": trial.suggest_int("n_units_embedding", 50, 500),
            "decoder_n_layers_hidden": trial.suggest_int("decoder_n_layers_hidden", 1, 5),
            "decoder_n_units_hidden": trial.suggest_int("decoder_n_units_hidden", 50, 500),
            "decoder_nonlin": trial.suggest_categorical("decoder_nonlin", ["relu", "leaky_relu", "tanh", "elu"]),
        }

        params_optim = {
            "n_iter": trial.suggest_int("n_iter", 100, 1000),
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
            "encoder_n_layers_hidden": 3,
            "encoder_n_units_hidden": 500,
            "encoder_nonlin": "leaky_relu",
            "n_units_embedding": 500,
            "decoder_n_layers_hidden": 3,
            "decoder_n_units_hidden": 500,
            "decoder_nonlin": "leaky_relu",
        }

        params_optim = {
            "n_iter": 10,
            "lr": 1e-3,
            "weight_decay": 1e-5,
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_test_params(cls):
        params_arch = {
            "encoder_n_layers_hidden": 3,
            "encoder_n_units_hidden": 500,
            "encoder_nonlin": "leaky_relu",
            "n_units_embedding": 500,
            "decoder_n_layers_hidden": 3,
            "decoder_n_units_hidden": 500,
            "decoder_nonlin": "leaky_relu",
        }

        params_optim = {
            "n_iter": 1,
            "lr": 1e-3,
            "weight_decay": 1e-5,
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }
