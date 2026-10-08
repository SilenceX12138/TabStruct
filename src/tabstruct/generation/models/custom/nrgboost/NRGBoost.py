from nrgboost import Dataset, NRGBooster

from ...BaseGenerator import BaseJointGenerator


class NRGBoost(BaseJointGenerator):

    def __init__(self, args):
        super().__init__(args)

        if args.task not in ["classification", "regression", "unsupervision"]:
            raise ValueError(f"Task {args.task} is not supported by {self.name}")

        if not args.model_specific_preprocessing:
            raise ValueError(f"Model-specific preprocessing is required for {self.name}")

        self.model_class = NRGBooster
        self.model_params = args.model_params

    def _fit_model(self, data_df):
        # Prepare the training data
        train_ds = Dataset(data_df)

        # Fit the model
        self.model = self.model_class.fit(train_ds, self.model_params, seed=42)

    def _generate_model(self, generation_config: dict):
        # Generate synthetic samples according to the config
        data_syn_temp_df = self.model.sample(
            num_samples=generation_config["num_samples_to_generate"],
            seed=self._next_generation_seed(),
        )

        return data_syn_temp_df

    def _update_generation_config(self, generation_config: dict) -> dict:
        generation_config = {
            "num_samples_to_generate": generation_config["num_samples_to_generate"],
        }

        return generation_config

    @classmethod
    def _define_default_params(cls):
        params = {
            "num_trees": 200,
            "shrinkage": 0.15,
            "max_leaves": 256,
            "max_ratio_in_leaf": 2,
            "num_model_samples": 80_000,
            "p_refresh": 0.1,
            "num_chains": 16,
            "burn_in": 100,
        }

        return params

    @classmethod
    def _define_optuna_params(cls, trial):
        params = {
            "num_trees": trial.suggest_int("num_trees", 100, 500),
        }

        return params

    @classmethod
    def _define_single_run_params(cls):
        params = {
            "num_trees": 1,
            "shrinkage": 0.15,
            "max_leaves": 256,
            "max_ratio_in_leaf": 2,
            "num_model_samples": 80_000,
            "p_refresh": 0.1,
            "num_chains": 16,
            "burn_in": 100,
        }

        return params

    @classmethod
    def _define_test_params(cls):
        params = {
            "num_trees": 1,
            "shrinkage": 0.15,
            "max_leaves": 256,
            "max_ratio_in_leaf": 2,
            "num_model_samples": 80_000,
            "p_refresh": 0.1,
            "num_chains": 16,
            "burn_in": 100,
        }

        return params

    @classmethod
    def _get_model_specific_scaler_config(cls):
        scaler_config_dict = {
            "context": {
                "disable_preprocessing": False,
            },
            "feature_scaler": {
                "categorical_transform": "onehot",
                "categorical_as_numerical": False,
            },
            "target_scaler": {},
        }

        return scaler_config_dict
