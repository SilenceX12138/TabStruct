import os

# ================================================================
# =                                                              =
# =                       Path setup                             =
# =                                                              =
# ================================================================
# BASE_DIR = os.getcwd()  # path to the project directory (the path you run `python` command)
BASE_DIR = os.getcwd()  # path to the project directory (the path you set git repo)
while not os.path.exists(os.path.join(BASE_DIR, ".git")):
    BASE_DIR = os.path.dirname(BASE_DIR)
LOG_DIR = f"{BASE_DIR}/logs"

os.makedirs(LOG_DIR, exist_ok=True)

# ================================================================
# =                                                              =
# =                     Runtime setup                            =
# =                                                              =
# ================================================================
TUNE_STUDY_TIMEOUT = 3600 * 2
SINGLE_RUN_TIMEOUT = 3600 * 2

# ================================================================
# =                                                              =
# =                       Wandb setup                            =
# =                                                              =
# ================================================================
# change the name to launch a new W&B project
WANDB_ENTITY = "tabular-foundation-model"
WANDB_PROJECT = "Kittens-dev"

# ================================================================
# =                                                              =
# =                       Env setup                              =
# =                                                              =
# ================================================================
# arm64 architecture may have issues with OpenBLAS using too many threads
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# ================================================================
# =                                                              =
# =                    Metric setup                              =
# =                                                              =
# ================================================================
metric_to_maximize_list = [
    "balanced_accuracy",
    "density_high_order_alpha_precision",
    "r2",
]

# ================================================================
# =                                                              =
# =                     Model setup                              =
# =                                                              =
# ================================================================
predictior_list = [
    # sklearn
    "lr",
    "rf",
    "knn",
    "xgb",
    "tabnet",
    "tabpfn",
    "tabforge",
    "mlp-sklearn",
    # lit
    "mlp",
    "ft-transformer",
]

generator_list = [
    # Real data
    "real",
    "real-test",
    # imblearn
    "smote",
    # tabeval
    "ctgan",
    "tvae",
    "bn",
    "goggle",
    "tabddpm",
    "arf",
    "nflow",
    # custom
    "tabebm",
    "great",
    "nrgboost",
    "tabular-argn",
    "ae",
    "vae",
    "ddpm",
    "tddpm",
    "tabdiff",
    "vesde",
    "vpsde",
    "edm",
    "tabsyn",
    # exploration
    "lae",
    "lfm",
    "lfdm",
    "lfdm-trans",
    "tabforge",
]

unstable_generator_list = [
    "bn",
    "arf",
    "nflow",
    "goggle",
    "great",
]

manual_optimizer_model_list = [
    "tabsyn",
    "lfdm",
    "lfdm-trans",
]

model_to_do_list = []
