import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import confusion_matrix

from src.tabstruct.common.model.BaseModelHelper import BaseModelHelper

from .utils.evaluation import compute_all_metrics


class PredictorHelper(BaseModelHelper):
    """Helper class for training and evaluating models."""

    @classmethod
    def _model_handler(cls, model):
        match model:
            case "xgb":
                from .sklearn.GBDT import XGBoost

                model_class = XGBoost
            case "knn":
                from .sklearn.KNN import KNN

                model_class = KNN
            case "lr":
                from .sklearn.Linear import LinearModel

                model_class = LinearModel
            case "rf":
                from .sklearn.Tree import RandomForest

                model_class = RandomForest
            case "tabpfn":
                from .sklearn.TabPFN import TabPFN

                model_class = TabPFN
            case "mlp-sklearn":
                from .sklearn.MLPSklearn import MLPSklearn

                model_class = MLPSklearn
            case "tabnet":
                from .sklearn.TabNet import TabNet

                model_class = TabNet
            case "mlp":
                from .lit.LitMLP import LitMLP

                model_class = LitMLP
            case "ft-transformer":
                from .lit.LitFTTransformer import LitFTTransformer

                model_class = LitFTTransformer
            case _:
                raise NotImplementedError(f"Model {model} is not implemented.")

        return model_class

    @classmethod
    def _eval_model(cls, args, data_module, model):
        # === Make predictions ===
        pred_dict = cls.inference(data_module, model)

        # === Compute metrics for each split ===
        splits = ["train", "valid", "test"]
        metrics_dict = {}

        for split in splits:
            y_true = getattr(data_module, f"y_{split}")
            y_pred = pred_dict[split]["y_pred"]
            y_hat = pred_dict[split]["y_hat"]

            metrics = compute_all_metrics(args, y_true, y_pred, y_hat)
            metrics_dict[f"{split}_metrics"] = metrics

            # Log metrics to wandb
            for metric, value in metrics.items():
                wandb.run.summary[f"{split}_metrics/{metric}"] = value

        # === Custom logging ===
        # Record the computation cost
        cls.log_computation_cost(data_module, model)

        # Record class-wise recall for classification
        if args.task == "classification":
            y_true = data_module.y_test
            y_pred = pred_dict["test"]["y_pred"]

            # Compute class-wise recall
            weights = [args.train_class2weight[i.item()] for i in y_true]
            C = confusion_matrix(y_true, y_pred, sample_weight=weights)
            with np.errstate(divide="ignore", invalid="ignore"):
                recall_per_class = np.diag(C) / C.sum(axis=1)

            # Log recall statistics
            recall_range = recall_per_class.max() - recall_per_class.min()
            wandb.run.summary["test/recall_per_class_range"] = recall_range

            class2recall = {f"class_{i}": [recall] for i, recall in enumerate(recall_per_class)}
            wandb.log({"test_recall_per_class": wandb.Table(dataframe=pd.DataFrame(class2recall))})

        # Record feature selection
        if args.model in ["rf", "tabnet"]:
            gate_all, num_selected_features = model.feature_selection(data_module.X_test)
            wandb.run.summary["test/num_selected_features"] = num_selected_features
            wandb.log({"test_gate_all": wandb.Table(dataframe=pd.DataFrame(gate_all))})

        return metrics_dict

    @classmethod
    def _inference(cls, data_module, model):
        """Run inference on all splits with DRY code."""
        result = {}

        for split in ["train", "valid", "test"]:
            X = getattr(data_module, f"X_{split}")
            result[split] = {
                "y_pred": model.predict(X),
                "y_hat": model.predict_proba(X),
            }

        return result
