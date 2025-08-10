import numpy as np
import torch
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

from ..BasePredictor import BaseSklearnPredictor
from ..utils.evaluation import gate_bin2dec


class TabNet(BaseSklearnPredictor):

    def __init__(self, args):
        super().__init__(args)

        self.unsupervised_model = None
        if args.model_params["pretrain"]:
            self.unsupervised_model = TabNetPretrainer(
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=2e-2),
                mask_type="entmax",
            )

        if args.task == "regression":
            model_class = TabNetRegressor
        elif args.task == "classification":
            model_class = TabNetClassifier
        else:
            raise ValueError(f"Task {args.task} is not supported by {args.model}")

        self.model = model_class(**args.model_params)

    def fit(self, data_module):
        virtual_batch_size = max(self.args.batch_size // 2, 1)
        # Drop last only if last batch has only one sample
        drop_last = self.args.train_num_samples_processed % self.args.batch_size == 1

        # === Pretrain the model ===
        if self.unsupervised_model is not None:
            self.unsupervised_model.fit(
                X_train=data_module.X_train,
                eval_set=[data_module.X_valid],
                pretraining_ratio=0.8,
                batch_size=data_module.batch_size,
                max_epochs=1,
                drop_last=drop_last,
                virtual_batch_size=virtual_batch_size,
            )

        # === Train the model ===
        self.model.fit(
            data_module.X_train,
            data_module.y_train,
            eval_set=[(data_module.X_valid, data_module.y_valid)],
            # TabNet does not support weighted loss with dynamic weights
            # eval_metric=[WeightedCrossEntropy()],
            # WeightedCrossEntropy() has different input order to PyTorch loss functions
            loss_fn=torch.nn.CrossEntropyLoss(
                weight=torch.tensor(self.args.train_class_weight_list, dtype=torch.float32, device=self.args.device)
            ),
            batch_size=self.args.batch_size,
            virtual_batch_size=virtual_batch_size,
            max_epochs=max(
                int(self.args.max_steps // (self.args.train_num_samples_processed // self.args.batch_size)), 1
            ),
            drop_last=drop_last,
            from_unsupervised=self.unsupervised_model,
        )

    def _feature_selection(self, X=None):
        feature_importance, _ = self.model.explain(X)
        gate_all_mat = np.asarray((feature_importance != 0), dtype=int)
        num_selected_features = gate_all_mat.sum(axis=1).mean()
        gate_all = []
        for gate in gate_all_mat:
            gate_dec_str = gate_bin2dec(gate)
            gate_all.append(gate_dec_str)

        return gate_all, num_selected_features

    @classmethod
    def _define_default_params(cls):
        params = {
            "pretrain": False,
            "n_d": 24,
            "n_steps": 5,
            "gamma": 1.5,
            "cat_emb_dim": 2,
            "n_independent": 3,
            "n_shared": 3,
            "momentum": 0.015,
            "mask_type": "sparsemax",
        }
        return params

    @classmethod
    def _define_optuna_params(cls, trial):
        params = {
            "pretrain": False,
            "n_d": trial.suggest_int("n_d", 8, 64),
            "n_steps": trial.suggest_int("n_steps", 3, 10),
            "gamma": trial.suggest_float("gamma", 1.0, 2.0),
            "cat_emb_dim": trial.suggest_int("cat_emb_dim", 1, 3),
            "n_independent": trial.suggest_int("n_independent", 1, 5),
            "n_shared": trial.suggest_int("n_shared", 1, 5),
            "momentum": trial.suggest_float("momentum", 0.001, 0.4, log=True),
            "mask_type": trial.suggest_categorical("mask_type", ["sparsemax", "entmax"]),
        }

        return params

    @classmethod
    def _define_single_run_params(cls):
        params = {
            "pretrain": False,
            "n_d": 24,
            "n_steps": 5,
            "gamma": 1.5,
            "cat_emb_dim": 2,
            "n_independent": 3,
            "n_shared": 3,
            "momentum": 0.015,
            "mask_type": "sparsemax",
        }

        return params

    @classmethod
    def _define_test_params(cls):
        params = {
            "pretrain": False,
            "n_d": 8,
            "n_steps": 3,
            "gamma": 1.3,
            "cat_emb_dim": 1,
            "n_independent": 2,
            "n_shared": 2,
            "momentum": 0.02,
            "mask_type": "sparsemax",
            "lambda_sparse": 1e-3,
        }

        return params
