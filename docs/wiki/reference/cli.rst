.. _command-line-reference:

CLI Reference
=============

Run from the repository root:

.. code-block:: bash

   python -m src.tabstruct.experiment.run_experiment --help

The parser in ``common/runtime/config/argument.py`` is the authoritative option
source. The CLI starts a complete experiment; there are no separate ``fit`` or
``predict`` subcommands. See :doc:`../guide/tutorials` for runnable sequences.

.. _core-arguments:

Required arguments
------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Flag
     - Values
   * - ``--task``
     - ``classification``, ``regression``, ``unsupervision``.
   * - ``--model``
     - Identifier from :doc:`models`, chosen for the selected pipeline.
   * - ``--dataset``
     - TabCamel dataset name or a supported dataset source with metadata.

The parser's model choices combine both registries. A parser-accepted identifier
still needs a matching adapter in the chosen pipeline. ``unsupervision`` is
available only in generation.

.. _w-b:

Runtime and logging
-------------------

.. list-table::
   :header-rows: 1
   :widths: 45 20 35

   * - Flag
     - Default
     - Purpose
   * - ``--pipeline``
     - ``prediction``
     - ``prediction`` or ``generation``.
   * - ``--device``
     - CUDA if available, otherwise CPU
     - Adapter execution device.
   * - ``--accelerator``
     - ``auto``
     - Lightning accelerator selection.
   * - ``--seed``
     - ``42``
     - Model/runtime randomness.
   * - ``--tags``
     - Empty
     - One or more W&B tags for organizing runs.
   * - ``--disable_wandb``
     - Disabled
     - Disable logging; tag-based lookup still contacts W&B.
   * - ``--wandb_log_model``
     - Disabled
     - Enable W&B model logging through the Lightning logger.
   * - ``--deterministic`` / ``--debugging``
     - Disabled
     - Adapter/Lightning execution controls.

Entity and project are configured in ``src/tabstruct/common/__init__.py``.
There are no ``--wandb_entity`` or ``--wandb_project`` flags.

Dataset and splits
------------------

.. list-table::
   :header-rows: 1
   :widths: 45 20 35

   * - Flag
     - Default
     - Purpose
   * - ``--split_mode``
     - Task-dependent
     - ``random``, ``stratified``, or ``fixed``.
   * - ``--test_size`` / ``--valid_size``
     - ``0.2`` / ``0.1``
     - Fractions, or counts when greater than one. Validation splits the remaining pool.
   * - ``--test_id`` / ``--valid_id``
     - ``0`` / ``0``
     - Split random states.
   * - ``--min_sample_per_class`` / ``--drop_class_id``
     - Unset
     - Classification filtering.
   * - ``--num_workers`` / ``--pin_memory``
     - ``0`` / Disabled
     - Torch dataloader options.

See :doc:`../guide/data` for the preprocessing flags, schema requirements, and
split formulas. ``--model_specific_preprocessing`` enables adapter overrides;
``--disable_preprocessing_tentative`` requests raw data.

.. _data-curation:

Synthetic training data
-----------------------

``--curate_mode sharing`` replaces training rows using
``--synthetic_data_path PATH`` or W&B lookup via ``--generator`` and
``--generator_tags``. ``--curate_ratio`` defaults to ``1.0``.
``--disable_synthetic_data_validation`` bypasses generator-provenance lookup
for an explicit CSV; it does not disable data parsing.

Model persistence
-----------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Flag
     - Behavior (all disabled or unset by default)
   * - ``--save_model``
     - Save a pickle wrapper under the run directory.
   * - ``--eval_only``
     - Evaluate an existing model or synthetic CSV.
   * - ``--saved_checkpoint_path PATH``
     - Load the selected checkpoint directly.
   * - ``--use_saved_checkpoint``
     - Load a wrapper before the fitting workflow.
   * - ``--checkpoint_tags TAG``
     - Resolve a finished run's ``best_model_path`` when a direct path is absent.

``--use_best_hyperparams`` is parsed, but its current post-processing branch is
a placeholder. It does not retrieve tuned parameters. Use the Optuna workflow
for implemented parameter search. See :doc:`../guide/workflows` for persistence
constraints and methods reconstructed from reference data.

Generation controls
-------------------

.. list-table::
   :header-rows: 1
   :widths: 45 20 35

   * - Flag
     - Default
     - Purpose
   * - ``--generation_mode``
     - ``stratified``
     - ``stratified`` or ``uniform`` class proportions.
   * - ``--generation_num_samples``
     - Unset
     - Explicit sample count.
   * - ``--generation_ratio``
     - ``1``
     - Synthetic count relative to processed training rows.
   * - ``--generation_only``
     - Disabled
     - Generate and save CSV without metrics.
   * - ``--synthetic_data_path PATH``
     - Unset
     - Evaluate an existing original-schema table.

.. _evaluation-toggles:

Evaluation controls
-------------------

``--enable_eval_structure`` enables per-feature utility (default disabled).
``--disable_eval_density`` and ``--disable_eval_privacy`` set the corresponding
flags to false. ``--enable_full_split_eval`` enables density/privacy flag
handling on every split. Structure still runs only on test. Without full split
evaluation, training density/privacy are always evaluated and validation metrics
are empty. See :doc:`../guide/evaluation` before choosing these flags.

.. _lightning-training:

.. _tuning:

Training and tuning
-------------------

.. list-table::
   :header-rows: 1
   :widths: 48 20 32

   * - Flag
     - Default
     - Purpose
   * - ``--max_steps_tentative``
     - ``10000``
     - Requested training budget; at least one epoch is enforced.
   * - ``--batch_size_tentative``
     - ``512``
     - Capped by processed training size.
   * - ``--optimizer``
     - ``sgd``
     - ``adam``, ``adamw``, or ``sgd`` for supporting models.
   * - ``--lr_scheduler``
     - Unset
     - ``plateau``, ``cosine_warm_restart``, ``linear``, or ``lambda``.
   * - ``--split_early_stopping``
     - ``valid``
     - ``train``, ``valid``, or ``test``. Use validation for model selection.
   * - ``--metric_early_stopping``
     - ``total_loss``
     - Metric monitored by supporting Lightning models.
   * - ``--enable_optuna``
     - Disabled
     - Run the adapter's search space.
   * - ``--optuna_trial``
     - ``20``
     - Maximum trial count.
   * - ``--num_repeats`` / ``--num_cv_folds``
     - ``10`` / ``1``
     - Split IDs evaluated within each tuning trial.
   * - ``--tune_max_workers``
     - ``5``
     - Processes per tuning trial; reduced to one under multi-rank launch.
   * - ``--tune_reduction``
     - ``mean``
     - ``mean``, ``median``, ``min``, or ``max`` across split results.
   * - ``--metric_model_selection``
     - ``total_loss``
     - Validation metric used by tuning.
   * - ``--disable_optuna_pruning``
     - Disabled
     - Disable the median pruner.

Training flags affect only adapters that consume them. ``--help`` also lists
scheduler-specific values, gradient clipping, and Lightning logging/validation
cadence. Runtime-derived values are resolved after data preparation.

.. _notes:

Distributed execution
---------------------

The runtime recognizes ``torchrun`` ranks and disables W&B logging on nonzero
ranks. The helper keeps inference collective and assigns serial metrics and
artifact writes to rank zero. Distributed support is adapter-specific; this
behavior does not make every registered generator or predictor support DDP.
