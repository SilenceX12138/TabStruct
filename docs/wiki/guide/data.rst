Data and preprocessing
======================

Dataset selection
-----------------

``--dataset`` is passed to ``TabularDataset(dataset_name=..., task_type=...)``
from TabCamel. Named benchmark datasets such as ``credit-g`` carry their target
and column metadata through that library. The public CLI does not define
``--target_col`` or ``--categorical_columns`` flags.

For a custom dataset, prepare and register its schema using TabCamel's
``TabularDataset`` interface. A supervised table needs a target column and
numerical/categorical column metadata; an ``unsupervision`` table treats all
columns as features. Passing an arbitrary CSV path does not supply missing
supervised metadata. For :doc:`synthetic evaluation <evaluation>`, the real
named dataset supplies the column types.

Splits and reproducibility
--------------------------

The test split is selected first, using ``--test_size`` and ``--test_id``.
The remaining rows are split again using ``--valid_size`` and ``--valid_id``.
Validation size is therefore relative to the remaining training/validation
pool. For fractions, the final proportions are:

.. math::

   p_{test}=t,\qquad p_{valid}=(1-t)v,\qquad p_{train}=(1-t)(1-v).

The defaults ``t=0.2`` and ``v=0.1`` yield approximately 20%, 8%, and 72%.
Values greater than one are converted to absolute sample counts.

Classification defaults to ``stratified`` splitting; regression and
``unsupervision`` default to ``random``. Regression requires random splitting.
``fixed`` is also accepted by the parser and delegated to TabCamel.

``--seed`` controls model randomness. Split random states come from
``test_id`` and ``valid_id``, so changing only ``--seed`` does not select
another train/test split. Keep split IDs and preprocessing identical when
restoring a checkpoint or comparing generators.

Preprocessing
-------------

The shared loader removes constant features. Classification can additionally
filter small classes using ``--min_sample_per_class`` or drop a class using
``--drop_class_id``.

.. list-table:: Shared defaults
   :header-rows: 1
   :widths: 40 20 40

   * - CLI option
     - Default
     - Behavior
   * - ``--categorical_impute_tentative``
     - ``most_frequent``
     - Impute categorical missing values.
   * - ``--numerical_impute_tentative``
     - ``mean``
     - Use ``mean`` or ``median``.
   * - ``--categorical_transform_tentative``
     - ``onehot``
     - Use ``onehot`` or ``ordinal`` encoding.
   * - ``--numerical_transform_tentative``
     - ``standard``
     - Use ``standard``, ``minmax``, or ``quantile``.
   * - ``--categorical_as_numerical_tentative``
     - Disabled
     - Include encoded categorical features in numeric normalization.
   * - ``--model_specific_preprocessing``
     - Disabled
     - Apply the selected adapter's preprocessing configuration.

Imputation and transforms are fitted on training rows and reused on validation
and test rows. ``--disable_preprocessing_tentative`` requests raw data;
model-specific preprocessing can override tentative settings. TabFORGE's
adapters require ``--model_specific_preprocessing`` so their standalone
estimators can own schema handling.

Training on synthetic data
--------------------------

``--curate_mode sharing`` replaces the real training split with synthetic rows.
Validation and test rows remain real. This provides an ML efficacy workflow:
train on synthetic data and evaluate on held-out real data.

The synthetic CSV must contain original feature column names and, for supervised
tasks, the original target column. ``--curate_ratio`` selects the number of
synthetic training rows relative to the original real training split. Ensure
the CSV contains enough rows for the requested ratio.

.. code-block:: bash

   python -m src.tabstruct.experiment.run_experiment \
     --pipeline prediction \
     --task classification \
     --model lr \
     --dataset credit-g \
     --device cpu \
     --curate_mode sharing \
     --curate_ratio 1 \
     --synthetic_data_path logs/<configured-project>/<run-id>/synthetic_samples.csv \
     --disable_synthetic_data_validation \
     --tags tutorial-ml-efficacy

Use a CSV created by the generation tutorial with matching split IDs. The
validation bypass flag skips W&B generator-provenance lookup; it does not
change CSV parsing or the real dataset's schema.
