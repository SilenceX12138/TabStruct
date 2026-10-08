Evaluation
==========

TabStruct evaluates synthetic tables across **four complementary dimensions**.
**TabEval** supplies synthetic-data evaluators, and the prediction pipeline
measures ML efficacy on held-out real rows after training on synthetic data.

.. list-table:: Four evaluation dimensions
   :header-rows: 1
   :widths: 28 72

   * - Dimension
     - Question
   * - **Density estimation**
     - Does the synthetic table preserve the real data distribution?
   * - **Privacy preservation**
     - How much information might synthetic records reveal about training rows?
   * - **ML efficacy**
     - Can a model trained on synthetic rows predict held-out real outcomes?
   * - **Structural fidelity**
     - Does the synthetic table preserve relationships across columns?

These dimensions answer different questions. Global utility adds a
structural perspective that is independent of a single designated target.
The following sections describe the current evaluator configuration and split
policy for reproducible comparisons.

Structural fidelity and global utility
--------------------------------------

The `paper <https://arxiv.org/abs/2509.11950>`_ introduces **global utility**:
use each column as a prediction target in turn and assess what the remaining
columns predict about it. This probes relationships throughout a table without
requiring a ground-truth causal graph.

The generation runner delegates structural evaluation to **TabEval**'s
``UtilityPerFeature`` through ``compute_structure_metrics``. It includes
features and the original supervised target in ``full_col_list_eval`` and uses
an ordinal-encoded view. Enable it with ``--enable_eval_structure``.

The current call specifies ``custom_hyperparameters={"KNN": {}, "RF": {}}``
and ``time_limit=60`` per feature. This is the current configured evaluator;
it does not establish reproduction of the paper's full predictor ensemble,
conditional-independence experiments, or result aggregation. Returned keys are
prefixed ``structure_`` and incorporate the installed TabEval metric name and
its result keys. Inspect the per-feature results as well as any aggregates.

.. _conventional-dimensions:

Evaluator configuration
-----------------------

.. list-table:: Generation evaluators
   :header-rows: 1
   :widths: 24 30 46

   * - Dimension
     - Implementation
     - Input and summary keys
   * - Density
     - ``LowOrderMetrics`` / ``HighOrderMetrics``
     - Original columns plus SDMetrics metadata for low-order metrics;
       one-hot columns for high-order metrics. Keys start with ``density_``.
   * - Privacy
     - ``DCR``
     - Distance to closest record on original columns with metadata;
       at most 3,000 samples. Keys start with ``privacy_``.
   * - Structure
     - ``UtilityPerFeature``
     - Ordinal view of all columns; enabled explicitly.
       Keys start with ``structure_``.
   * - ML efficacy
     - Prediction with ``--curate_mode sharing``
     - Train a predictor on synthetic rows, evaluate on held-out real rows.
       See :doc:`data`.

A DCR measurement describes record proximity; it does not provide a formal
privacy guarantee. Metric schemas come from the installed TabEval version.

Evaluation splits
-----------------

The ordinary generation policy is fixed by split:

.. list-table:: Without ``--enable_full_split_eval``
   :header-rows: 1
   :widths: 20 80

   * - Split
     - Evaluators
   * - ``train``
     - Density and privacy, regardless of the disable flags.
   * - ``valid``
     - No generation metrics (an empty dictionary).
   * - ``test``
     - Structure if ``--enable_eval_structure`` is supplied.

With ``--enable_full_split_eval``, density and privacy follow their enable/
disable flags on all three splits. Structure still runs **only on test**:
``compute_tabeval_metrics`` also checks ``split == "test"``. Tuning enables
full split evaluation automatically so supported validation metrics can be used
for model selection.

For a structure-only evaluation that respects disable flags, use:

.. code-block:: bash

   bash docs/tutorial/example_scripts/generation/eval.sh \
     logs/<configured-project>/<run-id>/synthetic_samples.csv \
     --enable_full_split_eval \
     --disable_eval_density \
     --disable_eval_privacy

``--generation_only`` bypasses the complete metric suite and saves the CSV.
``--model real`` and ``--model real-test`` use real training or test tables as
reference baselines without fitting a generator.

Prediction metrics
------------------

Classification reports ``balanced_accuracy``, ``F1_weighted``,
``precision_weighted``, ``recall_weighted``, ``AUROC_weighted``, ``ECE``, and
``cross_entropy_loss``. Probability columns follow the encoded class order.
Regression reports ``rmse``, ``mse``, and ``r2`` in the restored target space.

Prediction evaluates all three splits. W&B also records weighted class-recall
statistics for classification and feature-selection output for supported
predictors.

Read and compare outputs
------------------------

``run_experiment`` returns a dictionary with ``train_metrics``,
``valid_metrics``, and ``test_metrics`` for evaluated runs. W&B stores scalar
keys as ``<split>_metrics/<metric>``. Generation-only runs return an empty
metric dictionary.

Keep dataset versions, split IDs, row counts, preprocessing, and evaluator
configuration constant when comparing results. Record generator parameters and
installed package versions. Use real-data references to interpret scores; do
not infer structural fidelity from a single downstream classification score.
