import pandas as pd
from tabeval.metrics.eval_density import HighOrderMetrics, LowOrderMetrics
from tabeval.metrics.eval_privacy import DCR
from tabeval.metrics.eval_structure import UtilityPerFeature
from tqdm import tqdm

import wandb


# ================================================================
# =                                                              =
# =                   Data operations                            =
# =                                                              =
# ================================================================
def compute_all_metrics(args, loader_dict_real, loader_dict_syn) -> dict:
    metric_dict = {}

    pbar = tqdm(["train", "valid", "test"])
    for split in pbar:
        pbar.set_description(f"Evaluating on {split} split")
        temp_metric_dict = {}

        # Compute TabEval metrics
        temp_metric_dict |= compute_tabeval_metrics(
            args,
            loader_dict_real=loader_dict_real[split],
            loader_dict_syn=loader_dict_syn,
            split=split,
        )

        # Log metrics
        for metric, value in temp_metric_dict.items():
            wandb.run.summary[f"{split}_metrics/{metric}"] = value

        metric_dict[f"{split}_metrics"] = temp_metric_dict

    return metric_dict


def compute_tabeval_metrics(
    args,
    loader_dict_real: dict,
    loader_dict_syn: dict,
    split: str,
) -> dict:
    # === Initialize the metric dictionary ===
    # All sanity and stat metrics are computed with one-hot encoded features + labetls
    # See prior work: https://github.com/amazon-science/tabsyn/tree/main/eval
    metric_dict = {}

    # === Determine which metrics to compute based on the split and args ===
    eval_density = args.eval_density
    eval_privacy = args.eval_privacy
    eval_structure = args.eval_structure
    if not args.full_split_eval:
        split_eval_overrides = {
            "train": (True, True, False),
            "valid": (False, False, False),
            "test": (False, False, eval_structure),
        }
        eval_density, eval_privacy, eval_structure = split_eval_overrides.get(
            split,
            (eval_density, eval_privacy, eval_structure),
        )

    # === Statistical metrics ===
    if eval_density:
        metric_dict |= compute_density_metrics(args, loader_dict_real, loader_dict_syn)

    # === Privacy metrics ===
    if eval_privacy:
        metric_dict |= compute_privacy_metrics(args, loader_dict_real, loader_dict_syn)

    # === Structure metrics ===
    if eval_structure and split == "test":
        metric_dict |= compute_structure_metrics(args, loader_dict_real, loader_dict_syn)

    return metric_dict


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
                        col: args.full_col2type_eval[col].sdmetrics_dtype
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
                        col: args.full_col2type_eval[col].sdmetrics_dtype
                        for col in loader_dict_real["all_original"].columns
                    },
                },
                fast_mode="dev" in args.tags or "reg_test" in args.tags or args.reg_test,
                max_num_samples=3000,
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
            column_list = args.full_col_list_eval
            res = proxy_structure_metric().evaluate(
                loader_dict_real["all_ordinal"],
                loader_dict_syn["all_ordinal"],
                column_list=column_list,
                time_limit=60,  # 1 minute time limit for each feature's utility evaluation
                custom_hyperparameters={
                    # TODO: restore to default configurations
                    "KNN": {},
                    "RF": {},
                },
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
