import argparse
from typing import Union

import numpy as np
import torch

from ... import WANDB_ENTITY, WANDB_PROJECT, generator_list, model_to_do_list, predictior_list, unstable_generator_list
from ...runtime.error.ManualStopError import ManualStopError
from ..log.TerminalIO import TerminalIO
from ..log.WandbHelper import WandbHelper


class AddOnlyNamespace(argparse.Namespace):
    """A class to make the namespace add-only, which only accepts adding new values, but not editing the old ones."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize _runtime directly avoiding __setattr__
        super().__setattr__("_runtime", {})

    def __getattr__(self, name):
        # Avoid infinite recursion from being called by __setattr__
        if name == "_runtime":
            return super().__getattribute__(name)
        # First try _runtime, if not found, let the base Namespace handle it
        if name in self._runtime.keys():
            return self._runtime[name]
        return super().__getattribute__(name)

    def __setattr__(self, name, value):
        # === Set attributes when the object is first created ===
        if not hasattr(self, "_runtime"):
            return super().__setattr__(name, value)

        # === Only allow adding new attributes to `_runtime`===
        if hasattr(self, name):
            raise AttributeError(f"Cannot modify attribute {name}")

        # Directly modify _runtime without using __setattr__
        self._runtime[name] = value

    def __delattr__(self, name):
        raise AttributeError("Cannot delete attribute")


def parse_arguments(args: Union[None, list, AddOnlyNamespace]) -> argparse.Namespace:
    # When the input args is already an AddOnlyNamespace, it will skip the parsing (used while tuning the model)
    if not isinstance(args, AddOnlyNamespace):
        parser = argparse.ArgumentParser()

        # ===== Runtime =====
        parser = add_runtime_setup(parser)
        parser = add_wandb_setup(parser)

        # ===== Data =====
        parser = add_dataset_setup(parser)
        parser = add_curation_setup(parser)

        # ===== Model =====
        # === General setup ===
        parser = add_model_setup(parser)
        # === Lightning models ===
        parser = add_lit_env_setup(parser)
        parser = add_lit_train_setup(parser)
        # === Generative models ===
        parser = add_generation_setup(parser)

        # ===== Evaluation =====
        parser = add_eval_setup(parser)

        # ===== Tuning =====
        parser = add_optuna_setup(parser)

        # ===== Regression test =====
        parser = add_reg_test_setup(parser)

        # ===== Parse the arguments =====
        # When the input args is None, it will parse the command line arguments
        # When the input args is a list, it will parse the input args (list)
        args = parser.parse_args(args)

    # Set up wandb for all runs (even those failed in cross_update and sanity_check)
    wandb_logger = WandbHelper.setup_wandb(args)

    # ===== Post-processing arguments =====
    # === Interactions between arguments ===
    cross_update(args)
    # === Sanity check ===
    sanity_check(args)
    # === Make the arguments read-only ===
    args = AddOnlyNamespace(**vars(args))
    # === Set up the logger string as a runtime argument to support pickle for arguments ===
    args.wandb_logger = wandb_logger

    return args


# ================================================================
# =                                                              =
# =                         Runtime                              =
# =                                                              =
# ================================================================
def add_runtime_setup(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Set up the runtime environment for the script.

    Args:
        parser (argparse.ArgumentParser): The parser to add the arguments.

    Returns:
        argparse.ArgumentParser: The parser with the added arguments.
    """
    # ===== General =====
    parser.add_argument(
        "--pipeline",
        type=str,
        default="prediction",
        choices=["generation", "prediction"],
    )
    parser.add_argument(
        "--task",
        type=str,
        default="classification",
        choices=[
            "regression",
            "classification",
        ],
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="device to run the model on"
    )

    # ===== SEEDS =====
    parser.add_argument("--seed", type=int, default=42)

    return parser


def add_wandb_setup(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Set up the wandb-related arguments.

    Args:
        parser (argparse.ArgumentParser): The parser to add the arguments.

    Returns:
        argparse.ArgumentParser: The parser with the added arguments.
    """
    # === Wandb logging ===
    parser.add_argument("--tags", nargs="+", type=str, default=[], help="Tags for wandb")
    parser.add_argument(
        "--wandb_log_model",
        action="store_true",
        dest="wandb_log_model",
        help="True for storing the model checkpoints in wandb",
    )
    parser.set_defaults(wandb_log_model=False)
    parser.add_argument(
        "--disable_wandb", action="store_true", dest="disable_wandb", help="True if you dont want to crete wandb logs."
    )
    parser.set_defaults(disable_wandb=False)

    return parser


# ================================================================
# =                                                              =
# =                          Data                                =
# =                                                              =
# ================================================================
def add_dataset_setup(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Set up the dataset-related arguments.

    Args:
        parser (argparse.ArgumentParser): The parser to add the arguments.

    Returns:
        argparse.ArgumentParser: The parser with the added arguments.
    """
    # ===== Dataset =====
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--min_sample_per_class", type=int, default=None, help="minimum number of samples per class")
    parser.add_argument("--drop_class_id", type=str, default=None, help="class id (in original dataset) to drop")

    # ===== Data split =====
    parser.add_argument(
        "--split_mode",
        type=str,
        default="stratified",
        choices=["random", "stratified", "fixed"],
    )
    parser.add_argument("--num_repeats", type=int, default=10, help="number of repeats for the cross-validation")
    parser.add_argument("--num_cv_folds", type=int, default=1, help="number of cross-validation folds")
    parser.add_argument("--test_size", type=float, default=0.2, help="Size of the test split")
    parser.add_argument("--test_id", type=int, default=0, help="Index of the test split")
    parser.add_argument("--valid_size", type=float, default=0.1, help="Size of the validation split")
    parser.add_argument("--valid_id", type=int, default=0, help="Index of the validation split")

    # ===== Data processing =====
    parser.add_argument("--categorical_impute", type=str, default="most_frequent", choices=["most_frequent"])
    parser.add_argument("--numerical_impute", type=str, default="mean", choices=["mean", "median"])
    parser.add_argument(
        "--categorical_transform",
        type=str,
        default="onehot",
        choices=["onehot", "ordinal"],
    )
    parser.add_argument(
        "--numerical_transform", type=str, default="standard", choices=["standard", "minmax", "quantile"]
    )
    parser.add_argument(
        "--categorical_as_numerical", action="store_true", help="Treat categorical as numerical during normalisation"
    )
    parser.set_defaults(categorical_as_numerical=False)

    # ===== Data loading =====
    parser.add_argument("--num_workers", type=int, default=0, help="number of workers for loading dataset")
    parser.add_argument("--pin_memory", action="store_true", dest="pin_memory")
    parser.set_defaults(pin_memory=False)

    return parser


def add_curation_setup(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Set up the data curation-related arguments.

    Args:
        parser (argparse.ArgumentParser): The parser to add the arguments.

    Returns:
        argparse.ArgumentParser: The parser with the added arguments.
    """
    # === Data curation ===
    parser.add_argument("--curate_mode", type=str, default=None, choices=["sharing"])
    parser.add_argument("--curate_ratio", type=float, default=1.0, help="#Curate : #Real samples")

    # === Data sources ===
    parser.add_argument("--generator", type=str, default=None, help="name of the generator")
    parser.add_argument("--generator_tags", nargs="+", type=str, default=[], help="tags of the generator")

    return parser


# ================================================================
# =                                                              =
# =                         Model                                =
# =                                                              =
# ================================================================
def add_model_setup(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Set up the model-related arguments.

    Args:
        parser (argparse.ArgumentParser): The parser to add the arguments.

    Returns:
        argparse.ArgumentParser: The parser with the added arguments.
    """
    # ===== Model =====
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=predictior_list + generator_list + model_to_do_list,
    )

    # ===== General settings =====
    parser.add_argument(
        "--use_saved_checkpoint",
        action="store_true",
        help="Use the saved checkpoint for the model",
    )
    parser.set_defaults(use_saved_checkpoint=False)
    parser.add_argument(
        "--saved_checkpoint_path",
        type=str,
        default=None,
        help="name of the wandb artifact name (e.g., model-1dmvja9n:v0)",
    )
    parser.add_argument("--checkpoint_tags", nargs="+", type=str, default=[], help="tags of the checkpoint")
    parser.add_argument(
        "--use_best_hyperparams",
        action="store_true",
        dest="use_best_hyperparams",
        help="True if you don't want to use the best hyperparams for a custom dataset",
    )
    parser.set_defaults(use_best_hyperparams=False)
    parser.add_argument(
        "--save_model",
        action="store_true",
        dest="save_model",
        help="True if you want to save the model",
    )
    parser.set_defaults(save_model=False)

    return parser


def add_lit_env_setup(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Set up the environment-related arguments for the lightning models.

    Args:
        parser (argparse.ArgumentParser): The parser to add the arguments.

    Returns:
        argparse.ArgumentParser: The parser with the added arguments.
    """
    # === Runtime ===
    parser.add_argument("--accelerator", type=str, default="auto", help="type of accelerator for pytorch lightning")
    parser.add_argument("--debugging", action="store_true", dest="debugging")
    parser.set_defaults(debugging=False)
    parser.add_argument("--deterministic", action="store_true", dest="deterministic")
    parser.set_defaults(deterministic=False)

    return parser


def add_lit_train_setup(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # === Training ===
    parser.add_argument(
        "--max_steps_tentative", type=int, default=10000, help="Specify the max number of steps to train."
    )
    parser.add_argument("--batch_size_tentative", type=int, default=512, help="Tentative batch size for training only")
    parser.add_argument("--full_batch_training", action="store_true", dest="full_batch_training")
    parser.set_defaults(full_batch_training=False)

    # === Optimisation ===
    # Optimiser
    parser.add_argument("--optimizer", type=str, choices=["adam", "adamw", "sgd"], default="sgd")
    parser.add_argument("--gradient_clip_val", type=float, default=2.5, help="Upper bound to cut gradients")
    # Scheduler
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        choices=["plateau", "cosine_warm_restart", "linear", "lambda", "none"],
        default="none",
    )
    parser.add_argument("--cosine_warm_restart_eta_min", type=float, default=1e-6)
    parser.add_argument("--cosine_warm_restart_t_0", type=int, default=35)
    parser.add_argument("--cosine_warm_restart_t_mult", type=float, default=1)
    # Model selection
    parser.add_argument(
        "--metric_model_selection",
        type=str,
        default="cross_entropy_loss",
        choices=["cross_entropy_loss", "total_loss", "balanced_accuracy", "mse_loss"],
    )
    parser.add_argument(
        "--patience_early_stopping",
        type=int,
        default=50,
        help="It will train for at least args.check_val_every_n_epoch * args.patience_early_stopping epochs",
    )
    parser.add_argument(
        "--log_every_n_steps_tentative",
        type=int,
        default=500,
        help="number of steps at which to display the Trainer logs (including wandb.log within lightning module)",
    )
    parser.add_argument(
        "--check_val_every_n_epoch_tentative",
        type=int,
        default=1,
        help="number of epochs at which to check the validation",
    )

    return parser


def add_generation_setup(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # === Generation configurations ===
    parser.add_argument(
        "--generation_mode",
        type=str,
        default="stratified",
        choices=["stratified", "uniform"],
        help="class distribution of the generated data",
    )
    parser.add_argument(
        "--generation_num_samples",
        type=int,
        default=None,
        help="number of samples to generate",
    )
    parser.add_argument(
        "--generation_ratio",
        type=float,
        default=1,
        help="#Synthetic : #Real samples to generate",
    )
    parser.add_argument("--synthetic_data_path", type=str, default=None, help="path to load the synthetic data")
    parser.add_argument("--generation_only", action="store_true", help="Only generate synthetic data, do not evaluate")
    parser.set_defaults(generation_only=False)

    return parser


# ================================================================
# =                                                              =
# =                       Evaluation                             =
# =                                                              =
# ================================================================
def add_eval_setup(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Set up the evaluation-related arguments for the lightning models.

    Args:
        parser (argparse.ArgumentParser): The parser to add the arguments.

    Returns:
        argparse.ArgumentParser: The parser with the added arguments.
    """

    # ===== Testing =====
    parser.add_argument(
        "--eval_only", action="store_true", dest="eval_only", help="Load a trained model for evaluation."
    )
    parser.set_defaults(eval_only=False)
    parser.add_argument(
        "--disable_eval_density",
        action="store_false",
        dest="eval_density",
        help="Disable the evaluation statistics.",
    )
    parser.set_defaults(eval_density=True)
    parser.add_argument(
        "--disable_eval_privacy",
        action="store_false",
        dest="eval_privacy",
        help="Disable the evaluation privacy metrics.",
    )
    parser.set_defaults(eval_privacy=True)
    parser.add_argument(
        "--enable_eval_structure",
        action="store_true",
        dest="eval_structure",
        help="Disable the evaluation structure metrics.",
    )
    parser.set_defaults(eval_structure=False)

    return parser


def add_optuna_setup(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Set up the optuna-related arguments.

    Args:
        parser (argparse.ArgumentParser): The parser to add the arguments.

    Returns:
        argparse.ArgumentParser: The parser with the added arguments.
    """
    parser.add_argument("--enable_optuna", action="store_true", dest="enable_optuna")
    parser.set_defaults(enable_optuna=False)
    parser.add_argument(
        "--optuna_trial", type=int, default=20, help="Numer of trials to find the optimal hyperparameters."
    )
    parser.add_argument(
        "--disable_optuna_pruning",
        action="store_false",
        dest="optuna_pruning",
        help="Activate the pruning feature. `MedianPruner` stops unpromising trials at the early stages of training.",
    )
    parser.set_defaults(optuna_pruning=True)
    parser.add_argument(
        "--tune_reduction",
        type=str,
        default="mean",
        choices=["median", "mean", "max", "min"],
        help="The reduction method for aggregating the evaluation results of a single trial.",
    )
    parser.add_argument(
        "--tune_max_workers", type=int, default=5, help="The number of workers for a single tuning trial."
    )

    return parser


def add_reg_test_setup(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Set up the regression test-related arguments for the lightning models.

    Args:
        parser (argparse.ArgumentParser): The parser to add the arguments.

    Returns:
        argparse.ArgumentParser: The parser with the added arguments.
    """
    parser.add_argument(
        "--reg_test",
        action="store_true",
        dest="reg_test",
        help="Run the regression test with a small dataset",
    )
    parser.set_defaults(reg_test=False)

    return parser


# ================================================================
# =                                                              =
# =                     Post-processing                          =
# =                                                              =
# ================================================================
def cross_update(args):
    # === When test_size and valid_size are bigger than 1, they are considered as the number of samples ===
    if args.test_size > 1:
        args.test_size = int(args.test_size)
    if args.valid_size > 1:
        args.valid_size = int(args.valid_size)

    # === Prepare the path to the curate data ===
    if args.curate_mode:
        handle_curate_data_path(args)

    # === Only classification task supports stratified data split ===
    if args.task != "classification" and args.split_mode == "stratified":
        args.split_mode = "random"

    # === For generation, if the model is "real", the synthetic data path will be a dummy path ===
    if args.pipeline == "generation" and args.model == "real":
        args.synthetic_data_path = "dummy_path_for_real_data"

    # === For generation, retrieve the synthetic data path > checkpoint path ===
    if (
        args.pipeline == "generation"
        and args.eval_only
        and not args.synthetic_data_path
        and not args.use_saved_checkpoint
        and not args.saved_checkpoint_path
    ):
        args.generator = args.model
        adjust_curate_data_info(args)
        try:
            args.synthetic_data_path = retrieve_artifact_path(
                args,
                {"tags": args.generator_tags[0], "pipeline": "generation", "model": args.generator},
                "generated_data_path",
                state_strict=False,
            )
        except KeyError:
            # For evaluating the generator, if the synthetic data path is not found, we can raise a manual stop error
            raise ManualStopError(
                f"Cannot find successful runs for generator {args.generator} with tags {args.generator_tags[0]}."
            )
        except Exception as e:
            if args.model in unstable_generator_list:
                # For unstable methods like BN, the crashed sweep runs does not have `valid_id` fields, so we need special handling
                raise ManualStopError(
                    f"Cannot find successful runs for generator {args.generator} with tags {args.generator_tags[0]}."
                )
            raise ValueError(
                f"Error retrieving the synthetic data path for generator {args.generator} with tags {args.generator_tags[0]}: {e}"
            )

    # === Override default settings with best model hyperparameters ===
    if args.use_best_hyperparams:
        pass

    # === Retrieve the model checkpoint when path is not provided ===
    if (
        args.model != "real"
        and len(args.checkpoint_tags) > 0
        and (args.use_saved_checkpoint or args.eval_only)
        and not args.saved_checkpoint_path
    ):
        args.saved_checkpoint_path = retrieve_artifact_path(
            args,
            {
                "tags": args.checkpoint_tags[0],
                "pipeline": args.pipeline,
                "model": args.model,
            },
            "best_model_path",
        )


def sanity_check(args):
    """Sanity check the arguments.

    Args:
        args (argparse.ArgumentParser): The parsed arguments.
    """
    # === Task compatibility checks ===
    if args.task == "regression" and args.split_mode != "random":
        raise ValueError("Regression task only supports random data split.")

    # === Model compatibility checks ===
    if args.model in model_to_do_list:
        raise ManualStopError(f"Model {args.model} is not supported yet.")

    # === Evaluation mode checks ===
    if args.eval_only and not args.saved_checkpoint_path and not args.synthetic_data_path:
        raise ValueError("Either model checkpoint or synthetic data path should be provided for evaluation-only mode.")

    # === Generation pipeline checks ===
    if args.pipeline == "generation":
        if args.synthetic_data_path and args.saved_checkpoint_path:
            raise ValueError("Cannot provide both synthetic data path and generator checkpoint path.")

        if args.categorical_transform not in ["onehot", "ordinal"]:
            raise ValueError("Generation only supports onehot and ordinal encoding for categorical features.")

        if (args.use_saved_checkpoint and not args.saved_checkpoint_path) and len(args.checkpoint_tags) == 0:
            raise ValueError("Checkpoint tags should be provided for the generator checkpoint.")

    # === Synthetic data path checks ===
    if args.synthetic_data_path and args.model != "real":
        validate_synthetic_data_path(args)


def handle_curate_data_path(args):
    """Handle the path to curate data."""
    # Use the synthetic data path as the curate data path
    if args.synthetic_data_path:
        args.curate_data_path = args.synthetic_data_path
    # Retrieve the curate data path from W&B
    else:
        adjust_curate_data_info(args)
        args.curate_data_path = retrieve_artifact_path(
            args,
            {
                "tags": args.generator_tags[0],
                "pipeline": "generation",
                "model": args.generator,
            },
            "generated_data_path",
        )


def retrieve_artifact_path(args, filter_criteria, path_key, state_strict=True):
    """Retrieve artifact path from W&B with given filter criteria.

    Args:
        args: Command line arguments
        filter_criteria: Dictionary with filter criteria
        path_key: Key to extract from the retrieved run data
        state_strict: Whether to filter by `finished` state (default: True)

    Returns:
        str: Path to the artifact
    """
    # === Build filter dictionary ===
    filter_dict = {"$and": [{"state": "finished"}]} if state_strict else {"$and": []}
    # Add mandatory filter criteria
    for key, value in filter_criteria.items():
        if key == "tags":
            filter_dict["$and"].append({"tags": value})
        else:
            filter_dict["$and"].append({f"config.{key}": value})
    # Add dataset filters
    filter_dict["$and"].extend(
        [
            {"config.dataset": args.dataset},
            {"config.split_mode": args.split_mode},
            {"config.test_id": args.test_id},
            {"config.valid_id": args.valid_id},
            {"config.test_size": args.test_size},
            {"config.valid_size": args.valid_size},
        ]
    )

    # Retrieve runs
    runs_df = WandbHelper.retrieve_runs_with_conditions(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        filter_dict=filter_dict,
        max_num_runs=1,
    )

    if runs_df.shape[0] == 0:
        model_name = filter_criteria.get("model", args.model)
        tags = filter_criteria.get("tags", "")
        raise ValueError(f"No run found for {model_name} with tags {tags}.")

    try:
        value = runs_df.iloc[0][path_key]
    except KeyError:
        raise KeyError(f"Key '{path_key}' not found in the retrieved runs DataFrame.")

    return value


def validate_synthetic_data_path(args):
    """Validate synthetic data path and extract generator information."""
    if args.pipeline == "prediction" and not args.curate_mode:
        raise ValueError(
            "Curate mode should also be provided when synthetic data path is provided for prediction task."
        )

    run_id = WandbHelper.parse_run_id_from_path(args.synthetic_data_path)
    run_df = WandbHelper.retrieve_runs_with_conditions(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        filter_dict={"$and": [{"state": "finished"}, {"name": run_id}]},
        max_num_runs=1,
    )

    generator_retrieved = run_df.iloc[0]["model"] if run_df.shape[0] > 0 else None
    generator_retrieved_tags = run_df.iloc[0]["tags"] if run_df.shape[0] > 0 else None

    # Handle different scenarios based on provided and retrieved generator information
    if args.generator and generator_retrieved:
        if args.generator != generator_retrieved:
            raise ValueError("Provided generator and retrieved generator do not match.")
    elif args.generator and not generator_retrieved:
        if len(args.generator_tags) == 0:
            raise ValueError("Generator tags should be provided for customised synthetic data.")
        TerminalIO.print(f"Using customised synthetic data from {args.curate_data_path}.", TerminalIO.OKGREEN)
    elif not args.generator and generator_retrieved:
        args.generator = generator_retrieved
        args.generator_tags = generator_retrieved_tags
    else:
        raise ValueError("Invalid generator and invalid run id for synthetic data.")


def fix_tentative_training_args(args):

    # === batch size ===
    if args.train_num_samples_processed < args.batch_size_tentative:
        args.batch_size = args.train_num_samples_processed
    else:
        args.batch_size = args.batch_size_tentative

    # === max steps ===
    # Training steps need to finish at least 1 epoch
    steps_per_epoch = max(np.ceil(args.train_num_samples_processed / args.batch_size), 1)
    args.max_steps = int(max(steps_per_epoch, args.max_steps_tentative))

    # === log_every_n_steps ===
    # Log at least five times before finish training
    # Log at least once per epoch (one more time from each training epoch end)
    args.log_every_n_steps = int(
        min(min(args.log_every_n_steps_tentative, args.max_steps // 5), max(steps_per_epoch, 1))
    )

    # === check val every n epoch ===
    # Checkpoint at least once before finish training
    max_epochs = max(np.floor(args.max_steps / steps_per_epoch), 1)
    args.check_val_every_n_epoch = int(max(min(args.check_val_every_n_epoch_tentative, max_epochs), 1))


def adjust_curate_data_info(args):
    """Adjust the configuration (e.g., curate_ratio) for curate data."""
    pass
