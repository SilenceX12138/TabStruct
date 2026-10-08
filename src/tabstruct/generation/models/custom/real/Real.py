from ...BaseGenerator import BaseGenerator


class Real(BaseGenerator):

    def __init__(self, args):
        super().__init__(args)

    def _fit(self, data_module):
        pass

    def _generate(self, class2synthetic_samples):
        pass

    @classmethod
    def _define_default_params(cls):
        params = {}

        return params

    @classmethod
    def _define_optuna_params(cls, trial):
        params = {}

        return params

    @classmethod
    def _define_single_run_params(cls):
        params = {}

        return params

    @classmethod
    def _define_test_params(cls):
        params = {}

        return params
