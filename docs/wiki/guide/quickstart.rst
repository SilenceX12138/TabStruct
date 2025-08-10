Quickstart
==========

Install
-------

.. code-block:: bash

   pip install -e .

Train a predictor
-----------------

.. code-block:: bash

   python -m src.tabstruct.experiment.run_experiment \
     --model mlp \
     --dataset adult \
     --test_size 0.2 \
     --valid_size 0.1 \
     --tags dev

Evaluate a saved checkpoint
---------------------------

.. code-block:: bash

   python -m src.tabstruct.experiment.run_experiment \
     --model mlp \
     --eval_only \
     --saved_checkpoint_path /path/to/checkpoint \
     --dataset adult \
     --test_size 0.2 \
     --valid_size 0.1 \
     --tags dev

Generate and evaluate synthetic data
-----------------------------------

.. code-block:: bash

   # Generate
   python -m src.tabstruct.experiment.run_experiment \
     --pipeline generation \
     --generation_only \
     --generation_ratio 10 \
     --model smote \
     --dataset mfeat-fourier \
     --test_size 0.2 \
     --valid_size 0.1 \
     --tags dev

   # Evaluate past generation
   python -m src.tabstruct.experiment.run_experiment \
     --pipeline generation \
     --model smote \
     --eval_only \
     --dataset mfeat-fourier \
     --test_size 0.2 \
     --valid_size 0.1 \
     --generator_tags dev \
     --curate_ratio 10 \
     --tags dev
