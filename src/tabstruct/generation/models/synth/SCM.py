import pandas as pd
from tabeval.plugins.generic.plugin_scm import SCMPlugin

from ..BaseGenerator import BaseTabEvalJointGenerator


class SCM(BaseTabEvalJointGenerator):

    def __init__(self, args):
        super().__init__(args)

        self.model = SCMPlugin(cd_method=args.model_params["architecture"]["cd_method"], strict=False)

    def _fit(self, data_module):
        data_df = pd.DataFrame(data_module.X_train)
        data_df[self.args.full_target_col_processed] = data_module.y_train
        self.model.fit(data_df, task_type=self.args.task)

    @classmethod
    def _define_default_params(cls):
        params_arch = {
            "cd_method": "direct-lingam",
        }

        params_optim = {}

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_optuna_params(cls, trial):
        params_arch = {
            "cd_method": trial.suggest_categorical("cd_method", ["direct-lingam", "lim"]),
        }

        params_optim = {}

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_single_run_params(cls):
        params_arch = {
            "cd_method": "direct-lingam",
        }

        params_optim = {}

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_test_params(cls):
        params_arch = {
            "cd_method": "direct-lingam",
        }

        params_optim = {}

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }
