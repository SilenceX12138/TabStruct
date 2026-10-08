Tutorials
=========

These scripts run from the repository root after :doc:`quickstart` setup.
They use the ``credit-g`` classification dataset, CPU execution, and split IDs
zero. W&B logging is online by default. Append ``--disable_wandb`` to run a
local check with explicit artifact paths.

.. raw:: html

   <div class="task-grid tutorial-grid" aria-label="Runnable tutorials">
     <section class="task-card">
       <div class="task-card-body">
         <h3>Predict and restore</h3>
         <p>Train a Credit classifier, save its fitted wrapper, and evaluate the checkpoint.</p>
       </div>
       <a class="task-art" href="#predict-and-restore" aria-label="Jump to the prediction tutorial"><img src="../_static/prediction.svg" alt="Tabular features mapped to predictions" width="240" height="120"></a>
       <a class="task-examples" href="#predict-and-restore">Run the tutorial <span aria-hidden="true">↓</span></a>
     </section>
     <section class="task-card">
       <div class="task-card-body">
         <h3>Generate and evaluate</h3>
         <p>Create a synthetic Credit table and evaluate it through TabEval.</p>
       </div>
       <a class="task-art" href="#generate-and-evaluate-a-table" aria-label="Jump to the generation tutorial"><img src="../_static/generation.svg" alt="A real table transformed into synthetic rows" width="240" height="120"></a>
       <a class="task-examples" href="#generate-and-evaluate-a-table">Run the tutorial <span aria-hidden="true">↓</span></a>
     </section>
   </div>

Predict and restore
-------------------

Train a linear classifier and save its wrapper:

.. code-block:: bash

   bash docs/tutorial/example_scripts/prediction/train.sh

.. literalinclude:: ../../tutorial/example_scripts/prediction/train.sh
   :language: bash

**Expected output:** metrics for all splits and
``logs/<configured-project>/<run-id>/lr.pkl``. Read ``best_model_path`` from
the W&B summary, or locate the file under that directory for a local run:

.. code-block:: bash

   find logs -name lr.pkl

Restore the path from the training run:

.. code-block:: bash

   bash docs/tutorial/example_scripts/prediction/eval.sh \
     logs/<configured-project>/<run-id>/lr.pkl

.. literalinclude:: ../../tutorial/example_scripts/prediction/eval.sh
   :language: bash

**Expected output:** evaluation of the saved classifier on the same splits.
This workflow requires the same dependencies and preprocessing configuration
as training. See :doc:`workflows` for methods that reconstruct from reference
training rows instead of saving a fitted wrapper.

Generate and evaluate a table
-----------------------------

Generate synthetic rows before evaluating them:

.. code-block:: bash

   bash docs/tutorial/example_scripts/generation/train.sh

.. literalinclude:: ../../tutorial/example_scripts/generation/train.sh
   :language: bash

**Expected output:** ``synthetic_samples.csv`` under the run directory, with
original feature names and the original target column. The sample count is
one times the processed training count. SMOTE needs enough training examples
in every class for its neighbor configuration.

Read ``generated_data_path`` from the W&B summary, or locate the CSV for a
local run:

.. code-block:: bash

   find logs -name synthetic_samples.csv

Pass the path from the generation run to the evaluator:

.. code-block:: bash

   bash docs/tutorial/example_scripts/generation/eval.sh \
     logs/<configured-project>/<run-id>/synthetic_samples.csv

.. literalinclude:: ../../tutorial/example_scripts/generation/eval.sh
   :language: bash

**Expected output:** density and DCR metrics on training rows and per-feature
structural metrics against real test rows. Structural evaluation trains
predictors for every column and takes longer than CSV generation.

The evaluation script deliberately bypasses W&B generator-provenance lookup
for its explicit CSV path. CSV loading, column types, and metric preprocessing
still use the real named dataset. Keep the generation and evaluation split
settings identical. See :doc:`evaluation` for the split policy and
:doc:`../reference/cli` for additional flags.

Python entry point
------------------

The same workflow can return its metric dictionary to Python:

.. code-block:: python

   from src.tabstruct.experiment.run_experiment import run_experiment

   metrics = run_experiment([
       "--pipeline", "prediction",
       "--task", "classification",
       "--model", "lr",
       "--dataset", "credit-g",
       "--device", "cpu",
       "--tags", "tutorial-python",
   ])
   print(metrics["test_metrics"]["balanced_accuracy"])

This launches a complete experiment, including logging and runtime setup.
For lower-level interfaces and fitted-state requirements, see
:doc:`../reference/api`.
