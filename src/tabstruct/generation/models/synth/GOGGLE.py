from tabeval.plugins.generic.plugin_goggle import GOGGLEPlugin

from src.tabstruct.common.runtime.error.ManualStopError import ManualStopError

from ..BaseGenerator import BaseTabEvalJointGenerator


class GOGGLE(BaseTabEvalJointGenerator):

    def __init__(self, args):
        super().__init__(args)

        if args.dataset in ["ARTH150__570"]:
            raise ManualStopError("GOGGLE does not support ARTH150__570 dataset due to DGL implementation.")

        self.model = GOGGLEPlugin(
            # Architecture
            encoder_dim=args.model_params["architecture"]["encoder_dim"],
            encoder_l=args.model_params["architecture"]["encoder_l"],
            decoder_dim=args.model_params["architecture"]["decoder_dim"],
            decoder_arch=args.model_params["architecture"]["decoder_arch"],
            # Optimization
            n_iter=args.model_params["optimization"]["n_iter"],
            learning_rate=args.model_params["optimization"]["learning_rate"],
            weight_decay=args.model_params["optimization"]["weight_decay"],
            alpha=args.model_params["optimization"]["alpha"],
            beta=args.model_params["optimization"]["beta"],
            iter_opt=args.model_params["optimization"]["iter_opt"],
            threshold=args.model_params["optimization"]["threshold"],
            # Misc.
            sampling_patience=1,
            strict=False,
        )

    @classmethod
    def _define_default_params(cls):
        params_arch = {
            "encoder_dim": 64,
            "encoder_l": 2,
            "decoder_dim": 64,
            "decoder_arch": "gcn",
        }

        params_optim = {
            "n_iter": 50,
            "learning_rate": 5e-3,
            "weight_decay": 1e-3,
            "alpha": 0.1,
            "beta": 0.1,
            "iter_opt": True,
            "threshold": 0.1,
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_optuna_params(cls, trial):
        params_arch = {
            "encoder_dim": trial.suggest_int("encoder_dim", 32, 128),
            "encoder_l": trial.suggest_int("encoder_l", 1, 5),
            "decoder_dim": trial.suggest_int("decoder_dim", 32, 128),
            "decoder_arch": trial.suggest_categorical("decoder_arch", ["gcn", "het", "sage"]),
        }

        params_optim = {
            "n_iter": trial.suggest_int("n_iter", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-3, log=True),
            "alpha": trial.suggest_float("alpha", 0.0, 1.0),
            "beta": trial.suggest_float("beta", 0.0, 1.0),
            "iter_opt": trial.suggest_categorical("iter_opt", [True, False]),
            "threshold": trial.suggest_float("threshold", 0.0, 1.0),
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_single_run_params(cls):
        params_arch = {
            "encoder_dim": 64,
            "encoder_l": 2,
            "decoder_dim": 64,
            "decoder_arch": "gcn",
        }

        params_optim = {
            "n_iter": 10,
            "learning_rate": 5e-3,
            "weight_decay": 1e-3,
            "alpha": 0.1,
            "beta": 0.1,
            "iter_opt": True,
            "threshold": 0.1,
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_test_params(cls):
        params_arch = {
            "encoder_dim": 64,
            "encoder_l": 2,
            "decoder_dim": 64,
            "decoder_arch": "gcn",
        }

        params_optim = {
            "n_iter": 1,
            "learning_rate": 5e-3,
            "weight_decay": 1e-3,
            "alpha": 0.1,
            "beta": 0.1,
            "iter_opt": True,
            "threshold": 0.1,
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }
