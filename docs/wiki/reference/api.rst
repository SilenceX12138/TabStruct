API Reference
=============

This section provides detailed information about TabStruct's core APIs and interfaces.

Core Interfaces
---------------

BaseModel
~~~~~~~~~~

The foundation class for all models in TabStruct.

.. code-block:: python

    from tabstruct.common.model.BaseModel import BaseModel

    class CustomModel(BaseModel):
        def __init__(self, args):
            super().__init__(args)
            # Initialize your model here
            
        def _fit(self, data_module):
            # Implement training logic
            pass

**Key Methods:**

* ``__init__(args)``: Initialize the model with experiment arguments
* ``fit(data_module)``: Public API to train the model
* ``_fit(data_module)``: Abstract method to implement training logic
* ``get_metadata()``: Return model metadata including name and parameters
* ``define_params(reg_test, trial=None, dev=False)``: Define model parameters for different modes

**Parameter Definition Methods:**

* ``_define_default_params()``: Default parameters for production runs
* ``_define_optuna_params(trial)``: Parameters for hyperparameter optimization
* ``_define_single_run_params()``: Parameters for development/debugging
* ``_define_test_params()``: Minimal parameters for testing

Prediction Models
-----------------

BasePredictor
~~~~~~~~~~~~~

Base class for all prediction models.

.. code-block:: python

    from tabstruct.prediction.models.BasePredictor import BasePredictor

**Inheritance Hierarchy:**

* ``BasePredictor`` → ``BaseSklearnPredictor`` → Scikit-learn models (lr, rf, knn, xgb, tabnet, tabpfn, mlp-sklearn)
* ``BasePredictor`` → ``BaseLitPredictor`` → PyTorch Lightning models (mlp, ft-transformer)

**Available Prediction Models:**

**Scikit-learn Models:**

* ``lr``: Logistic Regression / Linear Regression
* ``rf``: Random Forest
* ``knn``: K-Nearest Neighbors
* ``xgb``: XGBoost
* ``tabnet``: TabNet
* ``tabpfn``: TabPFN (Prior-data Fitted Network)
* ``mlp-sklearn``: Multi-layer Perceptron (Scikit-learn)

**Lightning Models:**

* ``mlp``: Multi-layer Perceptron (PyTorch Lightning)
* ``ft-transformer``: Feature Tokenizer + Transformer

Generation Models
-----------------

BaseGenerator
~~~~~~~~~~~~~

Base class for all data generation models.

.. code-block:: python

    from tabstruct.generation.models.BaseGenerator import BaseGenerator

**Inheritance Hierarchy:**

* ``BaseGenerator``
  * ``BaseImblearnGenerator`` → SMOTE
  * ``BaseTabEvalGenerator`` → TabEval-based generators
    * ``BaseTabEvalConditionalGenerator`` → ctgan, tvae, tabddpm
    * ``BaseTabEvalJointGenerator`` → bn, arf, nflow, goggle, great
  * ``BaseMixedGenerator`` → Custom generators (TabSyn, TabDiff, TabEBM)

**Available Generation Models:**

**Real Data:**

* ``real``: Passthrough (no generation)

**Imbalanced-learn:**

* ``smote``: Synthetic Minority Oversampling Technique

**TabEval Generators:**

* ``ctgan``: Conditional Tabular GAN
* ``tvae``: Tabular Variational Autoencoder
* ``bn``: Bayesian Network
* ``goggle``: Gaussian Mixture Models
* ``tabddpm``: Tabular Denoising Diffusion Probabilistic Model
* ``arf``: Autoregressive Flow
* ``nflow``: Normalizing Flow
* ``great``: GReaT (Generation of Realistic Tabular data)

**Custom Generators:**

* ``TabSyn``: Tabular Synthesis with diffusion models
* ``TabDiff``: Tabular Diffusion
* ``TabEBM``: Tabular Energy-Based Model

Data Management
---------------

DataModule
~~~~~~~~~~

Lightning-compatible data module for handling tabular data.

.. code-block:: python

    from tabstruct.common.data.DataModule import DataModule
    
    data_module = DataModule(
        args=args,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test
    )

**Key Attributes:**

* ``X_train``, ``y_train``: Training data (numpy arrays)
* ``X_valid``, ``y_valid``: Validation data (numpy arrays)
* ``X_test``, ``y_test``: Test data (numpy arrays)
* ``train_dataset``, ``valid_dataset``, ``test_dataset``: PyTorch datasets

**Key Methods:**

* ``train_dataloader()``: Returns PyTorch DataLoader for training
* ``val_dataloader()``: Returns PyTorch DataLoader for validation
* ``test_dataloader()``: Returns PyTorch DataLoader for testing

Pipeline Classes
----------------

BasePipeline
~~~~~~~~~~~~

Base class for experiment pipelines.

**Available Pipelines:**

* ``PredictionPipeline``: Handles prediction experiments
* ``GenerationPipeline``: Handles data generation experiments

Experiment Configuration
------------------------

The main configuration is handled through command-line arguments. Key argument categories:

**Core Arguments:**

* ``--pipeline``: prediction | generation
* ``--model``: Model identifier (see Models section)
* ``--task``: classification | regression
* ``--dataset``: Dataset name (tabcamel compatible)
* ``--test_size``, ``--valid_size``: Split sizes
* ``--split_mode``: stratified | random
* ``--seed``: Random seed
* ``--device``: cpu | cuda

**Training Arguments:**

* ``--max_steps_tentative``: Maximum training steps
* ``--batch_size_tentative``: Batch size
* ``--optimizer``: adam | adamw | sgd
* ``--lr_scheduler``: none | plateau | cosine_warm_restart | linear | lambda

**Evaluation Arguments:**

* ``--eval_only``: Skip training, evaluate only
* ``--disable_eval_density``: Disable density evaluation
* ``--disable_eval_privacy``: Disable privacy evaluation
* ``--enable_eval_structure``: Enable structure evaluation

**Hyperparameter Tuning:**

* ``--enable_optuna``: Enable Optuna optimization
* ``--optuna_trial``: Trial number for Optuna
* ``--tune_max_workers``: Maximum workers for tuning

Usage Examples
--------------

**Prediction Pipeline:**

.. code-block:: bash

    python -m src.tabstruct.experiment.run_experiment \
        --pipeline prediction \
        --model xgb \
        --task classification \
        --dataset adult \
        --test_size 0.2 \
        --valid_size 0.2 \
        --seed 42

**Generation Pipeline:**

.. code-block:: bash

    python -m src.tabstruct.experiment.run_experiment \
        --pipeline generation \
        --model ctgan \
        --task classification \
        --dataset adult \
        --test_size 0.2 \
        --valid_size 0.2 \
        --seed 42

**Hyperparameter Tuning:**

.. code-block:: bash

    python -m src.tabstruct.experiment.run_experiment \
        --pipeline prediction \
        --model mlp \
        --task classification \
        --dataset adult \
        --enable_optuna \
        --tune_max_workers 4

Error Handling
--------------

**Common Exceptions:**

* ``ManualStopError``: Raised when model constraints are violated (e.g., TabPFN with >10 classes or >500 features)
* ``ValueError``: Raised for invalid task/model combinations
* ``NotImplementedError``: Raised when abstract methods are not implemented

**Model Constraints:**

* ``TabPFN``: Max 10 classes for classification, max 500 features
* ``TabEBM``: Max 500 features
* Some generators are unstable on large datasets (see ``unstable_generator_list``)

Constants and Configuration
---------------------------

**Key Constants:**

.. code-block:: python

    # Available models
    predictior_list = ["lr", "rf", "knn", "xgb", "tabnet", "tabpfn", "mlp-sklearn", "mlp", "ft-transformer"]
    generator_list = ["real", "smote", "ctgan", "tvae", "bn", "goggle", "tabddpm", "arf", "nflow", "great"]
    
    # Unstable generators (may fail on large datasets)
    unstable_generator_list = ["bn", "arf", "nflow", "goggle", "great"]
    
    # Timeouts
    TUNE_STUDY_TIMEOUT = 3600 * 2  # 2 hours
    SINGLE_RUN_TIMEOUT = 3600 * 2  # 2 hours

**Project Configuration:**

* ``WANDB_ENTITY``: "tabular-foundation-model"
* ``WANDB_PROJECT``: "Euphratica-dev"
* ``LOG_DIR``: "{BASE_DIR}/logs"

Notes
-----

* The framework automatically handles data preprocessing and feature encoding
* Lightning models support distributed training and mixed precision
* All models implement standardized parameter definition methods for reproducibility
* Generation models can handle both conditional and joint generation strategies
* The codebase supports integration with Weights & Biases for experiment tracking
