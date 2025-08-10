Overview
========

TabStruct is a modular framework for tabular machine learning with two pipelines:

- Prediction: supervised learning on tabular data
- Generation: synthesise tabular data and evaluate it

Key components
--------------

- Experiment runner (experiment/run_experiment.py): CLI entrypoint
- Pipelines (experiment/pipeline): orchestrates model training/eval
- Data layer (DataHelper, DataModule): split, curate, preprocess, and load data
- Model layer (prediction/models, generation/models): sklearn baselines, Lightning models and other tabular models
- Tuning (experiment/tune): Optuna sweeps

Metrics
-------
- Classification: balanced_accuracy, F1_weighted, precision, recall, AUROC_weighted, ECE, cross_entropy_loss
- Regression: rmse, mse, r2

W&B
---
- Project and entity are defined in `src/tabstruct/common/__init__.py`
- All runs log summaries and artifacts under `logs/` and W&B
