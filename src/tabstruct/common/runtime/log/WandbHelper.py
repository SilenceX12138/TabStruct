import os
import time

import pandas as pd
from lightning.pytorch.loggers import WandbLogger

import wandb

from ... import LOG_DIR, WANDB_ENTITY, WANDB_PROJECT


class WandbHelper:
    """Helper class for logging using Wandb."""

    # ================================================================
    # =                                                              =
    # =                     Initialisation                           =
    # =                                                              =
    # ================================================================
    @classmethod
    def setup_wandb(cls, args):
        """Set up the wandb tracking.

        Args:
            args (argparse.Namespace): The parsed arguments.

        """
        # === Intialise wandb logging ===
        wandb_logger = WandbLogger(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            tags=args.tags,
            log_model=args.wandb_log_model,
            settings=wandb.Settings(
                start_method="thread",
                quiet=True,
            ),
            save_dir=LOG_DIR,
        )

        # === Intialise the wandb run (necessary) ===
        # os.environ["WANDB_SILENT"] = "true"

        # For lightning, the wandb_logger will create run inherently.
        # For others, we need to call wandb.init() to create the run. But it may cause warning of duplicate runs for logger.
        # Therefore, we initialise the run here with wandb_logger to avoid the warning.
        # This allows other models to use wandb to log results as well.
        wandb.init(**wandb_logger._wandb_init)

        # Rename the run
        wandb.run.name = cls.get_run_name(args)

        # === Disable wandb when required ===
        if args.disable_wandb:
            # The logger is still inisitalised but not used
            os.environ["WANDB_MODE"] = "disabled"

        # Only add a string attribute when logging args, so that it does not increase the memory usage greatly
        return wandb_logger

    @staticmethod
    def get_run_name(args):
        """Get the run name for wandb logging.

        Args:
            args (argparse.Namespace): The parsed arguments.

        Returns:
            str: The run name. (type_model_dataset_generator_test-id_valid-id_run-id)
        """
        # === Prefix: run type ===
        prefix = "tune" if args.enable_optuna else "debug" if args.debugging else "benchmark"

        # === Suffix: generator + data split +run_id ===
        if args.enable_optuna:
            suffix = f"{args.generator}_{wandb.run.id}"
        else:
            suffix = f"{args.generator}_test-{args.test_id}_valid-{args.valid_id}_{wandb.run.id}"

        # === Run name ===
        run_name = "_".join([prefix, args.model, args.dataset, suffix])

        return run_name

    # ================================================================
    # =                                                              =
    # =            Special operations during runtime                 =
    # =                                                              =
    # ================================================================
    @staticmethod
    def trace_time(summary_key: str):
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()

                wandb.run.summary[f"{summary_key} (second)"] = end_time - start_time

                return result

            return wrapper

        return decorator

    # ================================================================
    # =                                                              =
    # =                   Post-hoc analysis                          =
    # =                                                              =
    # ================================================================
    @staticmethod
    def parse_run_id_from_path(synthetic_data_path: str):
        """Parse the run_id from the synthetic data path.

        Args:
            synthetic_data_path (str): The path to the synthetic data (e.g., .../Euphratic-dev/x1223833/.../synthetic_data_0.csv).

        Returns:
            str: The run_id.
        """
        dir_list = synthetic_data_path.split("/")
        if WANDB_PROJECT not in dir_list:
            return None
        # Reverse the list to find the last occurrence of WANDB_PROJECT
        dir_list = dir_list[::-1]
        pos = dir_list.index(WANDB_PROJECT)

        run_id = dir_list[pos - 1]

        return run_id

    @classmethod
    def retrieve_runs_with_conditions(
        cls,
        entity: str,
        project: str,
        filter_dict: list,
        max_num_runs: int = None,
    ):
        """Retrieve the runs from wandb with the given conditions.

        Args:
            filter_dict (dict): The condition list.
                When retrieving with run_id, please use `{"name": run_id}` as the condition: https://github.com/wandb/wandb/issues/5122

        Returns:
            pandas.Dataframe: The runs.
        """
        # ===== Initialise wandb API =====
        api = wandb.Api(timeout=60)

        # ===== Retrieve runs =====
        # === Retrieve all finished runs with the given conditions ===
        runs = api.runs(
            path=f"{entity}/{project}",
            filters=filter_dict,
            order="-created_at",
        )

        # ===== Filter some runs =====
        runs_filtered = cls.filter_runs(runs, max_num_runs)

        # ===== Transform runs into dataframe =====
        runs_df = cls.transform_runs_into_dataframe(runs_filtered)

        return runs_df

    @staticmethod
    def filter_runs(runs, max_num_runs):
        """
        - runs is a list sorted descending by .created_at
        """
        # === Set the maximum number of runs ===
        if max_num_runs is None:
            max_num_runs = len(runs)
        else:
            max_num_runs = min(max_num_runs, len(runs))

        # === Filter the runs ===
        filtered_runs = []
        for i, run in enumerate(runs):
            if i >= max_num_runs:
                break
            # filter summary runs
            if isinstance(run, str):
                continue

            filtered_runs.append(run)

        return filtered_runs

    @staticmethod
    def transform_runs_into_dataframe(runs):
        """
        Returns
        - Dataframe including
                - all configs
                - performance of best model
        """
        name_list, ids_list, config_list, summary_list = [], [], [], []
        for run in runs:
            # .summary contains the output of the run
            summary_list.append(run.summary._json_dict)

            # .config contains the hyperparameters. We remove special values that start with _.
            config_dict_set = {k: v for k, v in run.config.items() if k != "_runtime"}
            config_dict_runtime = {}
            if "_runtime" in run.config:
                config_dict_runtime = {k: v for k, v in run.config["_runtime"].items()}
            config_list.append(config_dict_set | config_dict_runtime)

            # .name is the human-readable name of the run.
            name_list.append(run.name)
            ids_list.append(run.id)

        return pd.concat(
            [
                pd.DataFrame({"name": name_list}),
                pd.DataFrame({"id": ids_list}),
                pd.DataFrame(summary_list),
                pd.DataFrame(config_list),
            ],
            axis=1,
        )


if __name__ == "__main__":
    run_id = WandbHelper.parse_run_id_from_path("logs/Euphratica-dev/ug3ph2zj/synthetic_samples.csv")
    print("Run ID: ", run_id)

    filter_dict = {
        "$and": [
            {"state": "finished"},
            {"name": run_id},
        ]
    }
    runs_df = WandbHelper.retrieve_runs_with_conditions(
        entity=WANDB_ENTITY, project=WANDB_PROJECT, filter_dict=filter_dict
    )
    print(runs_df.shape, runs_df.columns)
