import numpy as np
import pandas as pd
import wandb
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from tabcamel.data.transform import CategoryTransform, SimpleImputeTransform
from tabeval.metrics.eval_density import HighOrderMetrics, LowOrderMetrics
from tabeval.metrics.eval_privacy import DCR
from tabeval.metrics.eval_structure import UtilityPerFeature
from tabeval.plugins.core.dataloader import GenericDataLoader
from tqdm import tqdm

from src.tabstruct.common.data.DataHelper import DataHelper
from src.tabstruct.common.runtime.error.ManualStopError import ManualStopError


# ================================================================
# =                                                              =
# =                   Data operations                            =
# =                                                              =
# ================================================================
def compute_all_metrics(args, data_module, generation_dict) -> dict:
    metric_dict = {}

    pbar = tqdm(["train", "valid", "test"])
    for split in pbar:
        pbar.set_description(f"Evaluating on {split} split")
        temp_metric_dict = {}

        # === Prepare the data ===
        # Recover the original data for real samples
        # Note that imputation is not allowed for evaluating synthetic data, and thus we cannot use `recover_original_data`
        # Instead, we directly use the loaded synthetic data without any imputation
        original_data_dict = DataHelper.recover_original_data(
            args,
            getattr(data_module, f"X_{split}"),
            getattr(data_module, f"y_{split}"),
        )
        # Use real training samples when evaluating the "real" generator
        if args.model == "real" and split == "train":
            generation_dict = {
                "X_syn": getattr(data_module, "X_train"),
                "y_syn": getattr(data_module, "y_train"),
                "X_syn_original": original_data_dict["X_original"],
                "y_syn_original": original_data_dict["y_original"],
            }

        # Compute TabEval metrics
        real_data_df = pd.concat([original_data_dict["X_original"], original_data_dict["y_original"]], axis=1)
        synthetic_data_df = pd.concat([generation_dict["X_syn_original"], generation_dict["y_syn_original"]], axis=1)
        temp_metric_dict |= compute_tabeval_metrics(
            args,
            real_data_df,
            synthetic_data_df,
            getattr(data_module, f"X_{split}"),
            getattr(data_module, f"y_{split}"),
            generation_dict["X_syn"],
            generation_dict["y_syn"],
            split=split,
        )

        # Log metrics
        for metric, value in temp_metric_dict.items():
            wandb.run.summary[f"{split}_metrics/{metric}"] = value

        metric_dict[f"{split}_metrics"] = temp_metric_dict

    return metric_dict


def compute_tabeval_metrics(
    args,
    real_data_df: pd.DataFrame,
    synthetic_data_df: pd.DataFrame,
    X_real: np.ndarray,
    y_real: np.ndarray,
    X_syn: np.ndarray,
    y_syn: np.ndarray,
    split: str,
) -> dict:
    # === Preprocess the data ===
    loader_dict_real = prepare_loader_for_eval(args, X_real, y_real)
    loader_dict_syn = prepare_loader_for_eval(args, X_syn, y_syn)
    loader_dict_real["all_original"] = GenericDataLoader(data=real_data_df, target_column=args.full_target_col_original)
    loader_dict_syn["all_original"] = GenericDataLoader(
        data=synthetic_data_df, target_column=args.full_target_col_original
    )

    # === Initialize the metric dictionary ===
    # All sanity and stat metrics are computed with one-hot encoded features + labetls
    # See prior work: https://github.com/amazon-science/tabsyn/tree/main/eval
    metric_dict = {}

    # === Statistical metrics ===
    if args.eval_density:
        metric_dict |= compute_density_metrics(args, loader_dict_real, loader_dict_syn)

    # === Privacy metrics ===
    if args.eval_privacy:
        metric_dict |= compute_privacy_metrics(args, loader_dict_real, loader_dict_syn)

    # === Structure metrics ===
    if args.eval_structure and split == "test":
        metric_dict |= compute_structure_metrics(args, loader_dict_real, loader_dict_syn)

    return metric_dict


def prepare_loader_for_eval(args, X: np.ndarray, y: np.ndarray) -> dict:
    # === Build Dataloader for group 1 ===
    data = pd.DataFrame(X, columns=args.full_feature_col_list_processed)
    data[args.full_target_col_processed] = y
    loader_all = GenericDataLoader(data=data, target_column=args.full_target_col_processed)
    loader_cat = loader_all.drop(columns=args.num_feature_col_list_processed)
    loader_num = loader_all.drop(columns=args.cat_feature_col_list_processed)
    # For classification, locader_num contains the target column, as target column not in cat_feature_col_list_processed
    if args.task == "classification":
        loader_num = loader_num.drop(columns=[args.full_target_col_processed])
    # For regression, loader_cat contains the target column, as target column not in num_feature_col_list_processed
    elif args.task == "regression":
        loader_cat = loader_cat.drop(columns=[args.full_target_col_processed])

    # === Build Dataloader for group 2 ===
    loader_all_onehot = loader_all
    loader_cat_onehot = loader_cat
    if args.task == "classification":
        data = pd.DataFrame(X, columns=args.full_feature_col_list_processed)
        onehot_encoder = OneHotEncoder(sparse_output=False, categories=[args.class_encoded_list]).fit(y.reshape(-1, 1))
        y_onehot = onehot_encoder.transform(y.reshape(-1, 1))
        y_onehot_df = pd.DataFrame(
            y_onehot, columns=onehot_encoder.get_feature_names_out([args.full_target_col_processed])
        )
        data = pd.concat([data, y_onehot_df], axis=1)
        loader_all_onehot = GenericDataLoader(data=data, target_column=None)
        loader_cat_onehot = loader_all_onehot.drop(columns=args.num_feature_col_list_processed)

    # === Build Dataloader for group 3 ===
    loader_cat_ordinal = loader_cat
    X_df = pd.DataFrame(X, columns=args.full_feature_col_list_processed)
    for scaler_idx, feature_scaler in enumerate(args.feature_scaler_list[::-1]):
        X_df = feature_scaler.inverse_transform(X_df)
        if not isinstance(feature_scaler, CategoryTransform):
            continue
        if feature_scaler._encoder is None:
            # If the feature scaler is not fitted, we cannot use it to transform the data
            # This is mainly for the case when there is no categorical feature in the data
            continue
        # Imputation does not gurantee legal onehot values, so inverse transform of CategoryTransform can generate NaNs.
        # Thus, we need to apply the imputation again after the inverse transform of CategoryTransform
        for i in range(scaler_idx):
            if isinstance(args.feature_scaler_list[i], SimpleImputeTransform):
                X_df = args.feature_scaler_list[i].transform(X_df)
                break
        """ `OrdinalEncoder` can automatically infer the categories when `categories` is provided
        from sklearn.preprocessing import OrdinalEncoder

        enc = OrdinalEncoder(categories=[["Male", "Female"], ["1", "2", "3", "4"]])
        X = [["Male", "2"], ["Female", "3"], ["Female", "3"]]
        print(enc.fit(X))
        print(enc.categories_)
        print(enc.transform([["Female", "1"], ["Male", "4"]]))
        """
        # The data has been reversed to the original data, so we need to drop the original numerical features
        X_df_cat = X_df.drop(args.num_feature_col_list_original, axis=1)
        try:
            data_cat = OrdinalEncoder(
                categories=[list(cat) for cat in feature_scaler.categories_],
            ).fit_transform(X_df_cat)
        except Exception as e:
            # In very rare cases (mainly due to splitting cause some features with incomplete categories),
            # the category scaler fitted on train may generate illegal onehot values (e.g., all zeros).
            # In these cases, we need to impute the inverse transformed data with original categories (e.g., str).
            raise ManualStopError(f"OrdinalEncoder failed with error: {e}")
        data_cat = pd.DataFrame(data_cat, columns=X_df_cat.columns)
        if args.task == "classification":
            data_cat[args.full_target_col_processed] = y
        loader_cat_ordinal = GenericDataLoader(data=data_cat, target_column=None)
        break
    data_all_ordinal_df = pd.concat([loader_num.dataframe(), loader_cat_ordinal.dataframe()], axis=1)
    col_list_sorted = args.full_feature_col_list_original
    col_list_sorted = col_list_sorted + [args.full_target_col_original]
    data_all_ordinal_df = data_all_ordinal_df[col_list_sorted]
    loader_all_cardinal = GenericDataLoader(data=data_all_ordinal_df, target_column=args.full_target_col_processed)

    loader_dict = {
        # Group 1: Default to one-hot encoded features + ordinal labels (if applicable)
        "all": loader_all,
        "cat": loader_cat,  # includes labels for classification
        "num": loader_num,  # includes labels for regression
        # Group 2: one-hot encoded categorical features + labels
        "all_onehot": loader_all_onehot,
        "cat_onehot": loader_cat_onehot,
        # Group 3: ordinal categorical features + labels
        "all_ordinal": loader_all_cardinal,
        "cat_ordinal": loader_cat_ordinal,
    }

    return loader_dict


# ================================================================
# =                                                              =
# =                   Eval dimensions                            =
# =                                                              =
# ================================================================
def compute_density_metrics(args, loader_dict_real: dict, loader_dict_syn: dict) -> dict:
    density_metric_list = [
        # Low-order
        LowOrderMetrics,
        # High-order
        HighOrderMetrics,
    ]
    density_dict = {}

    for density_metric in density_metric_list:
        if density_metric in [LowOrderMetrics]:
            metric_name = density_metric.name()
            res = density_metric().evaluate(
                loader_dict_real["all_original"],
                loader_dict_syn["all_original"],
                metadata={
                    "columns": {
                        col: args.full_feature_col2type_original[col].sdmetrics_dtype
                        for col in loader_dict_real["all_original"].columns
                    },
                },
            )
        elif density_metric in [HighOrderMetrics]:
            metric_name = density_metric.name()
            res = density_metric().evaluate(loader_dict_real["all_onehot"], loader_dict_syn["all_onehot"])
        else:
            raise NotImplementedError(f"Statistical metric {density_metric} is not implemented.")

        if isinstance(res, dict):
            for key, value in res.items():
                density_dict[f"density_{metric_name}_{key}"] = value
        else:
            density_dict[f"density_{metric_name}"] = res

    return density_dict


def compute_privacy_metrics(args, loader_dict_real: dict, loader_dict_syn: dict) -> dict:
    privacy_metric_list = [
        DCR,
    ]
    privacy_dict = {}

    for privacy_metric in privacy_metric_list:
        if privacy_metric in [DCR]:
            metric_name = privacy_metric.name()
            res = privacy_metric().evaluate(
                loader_dict_real["all_original"],
                loader_dict_syn["all_original"],
                metadata={
                    "columns": {
                        col: args.full_feature_col2type_original[col].sdmetrics_dtype
                        for col in loader_dict_real["all_original"].columns
                    },
                },
                fast_mode="dev" in args.tags or "reg_test" in args.tags or args.reg_test,
            )
        else:
            raise NotImplementedError(f"Privacy metric {privacy_metric} is not implemented.")

        if isinstance(res, dict):
            for key, value in res.items():
                privacy_dict[f"privacy_{metric_name}_{key}"] = value
        else:
            privacy_dict[f"privacy_{metric_name}"] = res

    return privacy_dict


def compute_structure_metrics(
    args,
    loader_dict_real: dict,
    loader_dict_syn: dict,
) -> dict:
    proxy_structure_metric_list = [
        UtilityPerFeature,
    ]
    proxy_structure_dict = {}

    for proxy_structure_metric in proxy_structure_metric_list:
        if proxy_structure_metric in [UtilityPerFeature]:
            metric_name = proxy_structure_metric.name()
            column_list = args.full_feature_col_list_original + [args.full_target_col_original]
            res = proxy_structure_metric().evaluate(
                loader_dict_real["all_ordinal"],
                loader_dict_syn["all_ordinal"],
                column_list=column_list,
                time_limit=int(900 // len(column_list)),  # 15 minutes max in total
            )
        else:
            raise NotImplementedError(f"Proxy structure metric {proxy_structure_metric} is not implemented.")

        if isinstance(res, dict):
            for key, value in res.items():
                if isinstance(value, dict):
                    wandb.log({f"structure_{metric_name}_{key}": wandb.Table(dataframe=pd.DataFrame(value))})
                proxy_structure_dict[f"structure_{metric_name}_{key}"] = value
        else:
            proxy_structure_dict[f"structure_{metric_name}"] = res

    return proxy_structure_dict
