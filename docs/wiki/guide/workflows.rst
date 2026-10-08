Workflows
=========

Prediction
----------

The prediction pipeline loads and preprocesses data, fits the selected
classifier or regressor, and evaluates training, validation, and test rows.
An illustrative use is comparing a linear baseline with a nonlinear model on
a credit classification dataset.

.. code-block:: bash

   python -m src.tabstruct.experiment.run_experiment \
     --pipeline prediction \
     --task classification \
     --model lr \
     --dataset credit-g \
     --device cpu \
     --save_model \
     --tags tutorial-prediction

For regression, set ``--task regression`` and choose a regression dataset
supported by TabCamel. ``lr`` selects ``LinearRegression`` for regression and
``LogisticRegression`` for classification. Regression metrics are restored to
the original target units. See :doc:`evaluation` for metric keys.

Generation
----------

A generation run fits a model on the training split, generates synthetic rows,
restores original column names and values, and writes ``synthetic_samples.csv``.
An illustrative use is producing a shareable research table, then checking
whether relationships among columns survive generation.

.. code-block:: bash

   python -m src.tabstruct.experiment.run_experiment \
     --pipeline generation \
     --task classification \
     --model smote \
     --dataset credit-g \
     --device cpu \
     --generation_only \
     --generation_ratio 1 \
     --tags tutorial-generation

``--generation_num_samples`` selects an explicit count. Otherwise the count is
``int(processed_training_rows * generation_ratio)``. Classification defaults
to stratified class proportions; ``--generation_mode uniform`` requests equal
class proportions through the shared generation strategy. Individual adapters
may implement class control differently.

Use ``--task unsupervision`` for generation without a designated target, or
``--task regression`` for a continuous target. Check the chosen adapter's task
and dependency constraints in :doc:`../reference/models`.

CSV evaluation
--------------

Run the generation evaluation script with the path produced above:

.. code-block:: bash

   bash docs/tutorial/example_scripts/generation/eval.sh \
     logs/<configured-project>/<run-id>/synthetic_samples.csv

This skips generator fitting and enables the structural evaluator. Density,
privacy, and structure use separate preprocessing views fitted from the real
training data. Evaluation split behavior is described in :doc:`evaluation`.

For online provenance lookup, omit ``--disable_synthetic_data_validation`` and
use a generated CSV whose run exists in the configured W&B project. Alternatively,
``--eval_only --generator_tags tutorial-generation`` can resolve a generated
path from W&B when no CSV or checkpoint path is supplied. Keep the dataset,
split sizes, and split IDs identical. Recorded paths refer to local files;
lookup does not download artifacts from another machine.

Checkpoint lifecycle
--------------------

``--save_model`` writes a pickle wrapper and logs ``best_model_path``. Restore
it with ``--eval_only --saved_checkpoint_path PATH`` using the same task,
dataset, preprocessing, and splits. With ``--use_saved_checkpoint`` instead
of ``--eval_only``, the loaded wrapper enters the fitting workflow; whether
this resumes training is adapter-specific.

The shared helper treats ``knn``, ``smote``, and ``tabebm`` as methods that keep
reference data. Their saved pickle contains a marker rather than a fitted
wrapper, and evaluation reconstructs them from training data. For SMOTE,
prefer saving and evaluating its generated CSV.

``--checkpoint_tags TAG`` with ``--eval_only`` or ``--use_saved_checkpoint``
can resolve ``best_model_path`` from a finished W&B run. A generation run must
choose either a synthetic CSV or a generator checkpoint.

TabFORGE adapters
-----------------

Both pipelines register ``--model tabforge`` as an adapter to the standalone
machine-learning package. Obtain a compatible TabFORGE distribution separately
and install it in the same environment; it is not a declared TabStruct
dependency. Confirm that it exports the required estimators before launching:

.. code-block:: bash

   python -c "from tabforge import TabFORGEClassifier, TabFORGEGenerator"
   python -m src.tabstruct.experiment.run_experiment \
     --pipeline prediction \
     --task classification \
     --model tabforge \
     --dataset credit-g \
     --model_specific_preprocessing \
     --save_model \
     --tags tutorial-tabforge

The `PyPI project named tabforge <https://pypi.org/project/tabforge/>`_ is a
LaTeX template tool and does not supply these machine-learning estimators.

The adapter uses pretrained initialization by default, with feature-encoder
and backbone downloads managed by the standalone package. Choose hardware
appropriate to that model. ``--max_steps_tentative`` and
``--batch_size_tentative`` map into adapter training configuration. Adapter
``model_params`` are nested Python configuration groups; the CLI does not
accept a JSON ``--model_params`` flag.

Hyperparameter tuning
---------------------

Optuna selects the model's search space and evaluates repeated split IDs.
Choose a selection metric produced by the pipeline:

.. code-block:: bash

   python -m src.tabstruct.experiment.run_experiment \
     --pipeline prediction \
     --task classification \
     --model lr \
     --dataset credit-g \
     --device cpu \
     --enable_optuna \
     --optuna_trial 3 \
     --num_repeats 2 \
     --num_cv_folds 1 \
     --tune_max_workers 1 \
     --metric_model_selection balanced_accuracy \
     --tags tutorial-tuning

The tuning objective uses validation metrics, and tuning enables
``full_split_eval``. ``--num_repeats`` and ``--num_cv_folds`` expand tuning
runs; a normal invocation runs only its specified ``test_id`` and ``valid_id``.
The current runner has a two-hour single-run timeout and a two-hour study
budget. Some adapters do not implement an Optuna search space.
