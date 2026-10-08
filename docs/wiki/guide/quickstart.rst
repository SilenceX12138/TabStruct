Quickstart
==========

Requirements
------------

Use Python 3.10 or later and a Git checkout of
`the public repository <https://github.com/SilenceX12138/TabStruct>`_.
Run all commands from the repository root: the runtime discovers ``logs/``
by walking upwards to a ``.git`` directory.

Dataset downloads and online W&B logging require network access. Classical
baselines can run on CPU. Deep generators and pretrained models may require
GPU resources, model downloads, and model-specific dependencies.

.. _install:

Installation
------------

.. code-block:: bash

   git clone https://github.com/SilenceX12138/TabStruct.git
   cd TabStruct
   conda create -n tabstruct python=3.10.18
   conda activate tabstruct
   bash scripts/utils/install.sh

The script installs the package, Mostly AI, DGL and PyG extensions, and
PyTorch 2.2.2 with CUDA 12.1 wheels. Use it in a compatible Linux environment.
For a prepared environment, install the checkout directly:

.. code-block:: bash

   python -m pip install -e .

This installs declared dependencies; it does not run the additional wheel
installation steps in ``install.sh``. A successful CPU baseline does not
establish that every GPU generator dependency is installed.

Configure logging
-----------------

Set ``WANDB_ENTITY`` and ``WANDB_PROJECT`` in
``src/tabstruct/common/__init__.py`` to a workspace you can write to:

.. code-block:: python

   WANDB_ENTITY = "your-wandb-entity"
   WANDB_PROJECT = "tabstruct"

.. code-block:: bash

   wandb login

The checked-in values select a development workspace. Configure your own
workspace before running examples. Logging is online by default, with local
W&B files under ``logs/wandb``. Use ``--disable_wandb`` for a local check;
W&B run lookup by tags still requires access to the configured project.

.. _train-a-predictor:

Run a classifier
----------------

.. code-block:: bash

   python -m src.tabstruct.experiment.run_experiment \
     --pipeline prediction \
     --task classification \
     --model lr \
     --dataset credit-g \
     --device cpu \
     --save_model \
     --tags tutorial-prediction

``--task``, ``--model``, and ``--dataset`` are required. TabCamel loads the
named dataset and may download it on first use. You do not need to create a
CSV for this example.

Expected outputs
----------------

The terminal shows data preparation, training, and evaluation stages. W&B
summaries contain ``train_metrics/``, ``valid_metrics/``, and ``test_metrics/``
keys, including ``balanced_accuracy``, ``F1_weighted``, and ``AUROC_weighted``.
Exact scores depend on the dataset and installed model versions.

With ``--save_model``, the runner saves
``logs/<configured-project>/<run-id>/lr.pkl`` and records ``best_model_path``.
A Python call to ``run_experiment`` also returns the metric dictionary.

.. _evaluate-a-saved-checkpoint:

Restore the model
-----------------

Use the saved path from the preceding run with the same dataset, task, model,
split IDs, and preprocessing configuration:

.. code-block:: bash

   bash docs/tutorial/example_scripts/prediction/eval.sh \
     logs/<configured-project>/<run-id>/lr.pkl

The script evaluates the restored model on all three splits. It does not
expose a separate prediction-file export command.

.. _generate-and-evaluate-synthetic-data:

Next steps
----------

Follow :doc:`tutorials` for CSV generation and evaluation, :doc:`data` for split
semantics, and :doc:`../reference/cli` for the complete public option groups.
