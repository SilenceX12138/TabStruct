Models
======

Prediction
----------
- sklearn: lr, rf, knn, xgb, tabnet, tabpfn, mlp-sklearn
- lightning: mlp, ft-transformer

Generation
----------
- real (passthrough)
- imblearn: smote
- tabeval: ctgan, tvae, bn, goggle, tabddpm, arf, nflow, great

Selecting parameters
--------------------
- Each model class defines default, single-run, test, and Optuna search spaces via ``define_params``.
- Lightning models further have ``architecture`` and ``optimization`` subsections.

Caveats
-------
- Some generators are unstable on large datasets (see ``unstable_generator_list`` in ``src/tabstruct/common/__init__.py``).
- Regression requires random splits; stratified is classification-only.
