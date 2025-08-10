Command line reference
======================

Entry point: ``python -m src.tabstruct.experiment.run_experiment``

Core arguments
--------------

- --pipeline: prediction | generation
- --model: see Models section
- --task: classification | regression (prediction); generation infers from dataset
- --dataset: dataset name (as supported by tabcamel)
- --test_size, --valid_size: float in (0,1] or integer counts
- --split_mode: stratified | random (regression -> random only)
- --seed: int
- --device: cpu | cuda

Data curation
-------------
- --curate_mode: sharing
- --curate_ratio: float, number of curated per real sample
- --generator, --generator_tags: reference past generation in W&B
- --synthetic_data_path: explicit path

Lightning training
------------------
- --max_steps_tentative, --batch_size_tentative, --full_batch_training
- --optimizer [adam|adamw|sgd], --gradient_clip_val
- --lr_scheduler [none|plateau|cosine_warm_restart|linear|lambda]
- --metric_model_selection, --patience_early_stopping
- --log_every_n_steps_tentative, --check_val_every_n_epoch_tentative

Evaluation toggles
------------------
- --eval_only, --disable_eval_density, --disable_eval_privacy, --enable_eval_structure

Tuning
------
- --enable_optuna, --optuna_trial, --disable_optuna_pruning, --tune_reduction, --tune_max_workers

W&B
---
- --tags, --wandb_log_model, --disable_wandb, --checkpoint_tags

Notes
-----
- For generation eval-only, either provide ``--synthetic_data_path`` or ensure matching ``--generator_tags`` are retrievable.
- Regression tasks require ``--split_mode random``.
