import logging
import random
import warnings

import lightning as L
import numpy as np
import sklearn
import torch

from src.tabstruct.common.runtime.config.argument import parse_arguments

from ... import LOG_DIR


# ================================================================
# =                                                              =
# =                     Runtime setup                            =
# =                                                              =
# ================================================================
def setup_runtime(args):
    """Set up the runtime environment for the script.

    Args:
        args (List[str], optional): The arguments to parse when calling by test functions. Defaults to None.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    # === Ignore some warnings from repo versions ===
    warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
    # === Set up script logger at DEBUG level ===
    logging.basicConfig(
        filename=f"{LOG_DIR}/logs_exceptions.txt",
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG,
    )
    # === Parse arguments for configurations ===
    args = parse_arguments(args)
    # === Set gloabl random seed ===
    seed_everything(args)

    return args


def seed_everything(args):
    seed = args.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    # set `deterministic` to True only when debugging
    if "dev" in args.tags:
        torch.backends.cudnn.deterministic = True

    L.seed_everything(seed, workers=True)

    return seed
