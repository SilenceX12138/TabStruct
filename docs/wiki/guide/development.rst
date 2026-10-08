Developer guide
===============

Read the implemented workflow
-----------------------------

Start with ``run_experiment`` and follow its calls to ``setup_runtime`` and
``PipelineHelper.run_pipeline``. The selected helper runs
``benchmark_model`` in this order:

1. Resolve the model identifier to an adapter class.
2. Prepare the data and resolve tentative preprocessing/training arguments.
3. Create or restore the wrapper and fit it when appropriate.
4. Run inference, save generation output, and compute split metrics.

The :doc:`overview` maps public modules to these responsibilities.

Add a predictor
---------------

Use ``BaseSklearnPredictor`` for an estimator with ``fit``, ``predict``, and
``predict_proba`` interfaces, or ``BaseLitPredictor`` for Lightning training.
A concrete adapter supplies its estimator, parameter definitions, and any
special preprocessing. Regression probability prediction returns ``None``.

Use an existing wrapper such as
``src/tabstruct/prediction/models/sklearn/Linear.py`` as the pattern. Then
register its identifier in ``predictior_list`` (the implementation spelling)
in ``src/tabstruct/common/__init__.py`` and add a matching branch to
``PredictorHelper._model_handler``.

Callers use public ``fit``, ``predict``, and ``predict_proba`` methods.
Underscored hooks implement adapter behavior. Validate probability shape
``(n_rows, n_classes)`` and encoded class ordering before metric computation.

Add a generator
---------------

Choose an existing base strategy in ``BaseGenerator.py``: joint, conditional,
class-focused, TabEval-backed, or mixed/Lightning generation. Implement its
required fitting and generation hooks, plus parameter definitions. Use the
concrete bases to reuse class-allocation and conditional sampling behavior.

Register the identifier in ``generator_list`` and add the corresponding
``GeneratorHelper._model_handler`` branch. ``generate()`` reads counts and
class distribution from runtime arguments; CSV restoration and evaluation
remain in the helper. Check that exported columns match the original dataset
and that repeated generation calls use the intended seed behavior.

Parameter and preprocessing contracts
-------------------------------------

Define ordinary model parameters in ``_define_default_params`` and an optional
Optuna search space in ``_define_optuna_params``.

``get_model_specific_scaler_config()`` returns ``context``,
``feature_scaler``, and ``target_scaler`` dictionaries. The helper applies
these when ``--model_specific_preprocessing`` is enabled.

``AddOnlyNamespace`` retains parsed values and adds derived runtime fields.
Avoid mutating an existing argument through this interface. Pass new adapter
parameters explicitly and preserve feature/target transform state for restore.

Proportionate validation
------------------------

For a new adapter, check loading, fitting, output schema, inference, and
save/restore where supported on a small realistic dataset. Compare metrics
using the same split IDs and preprocessing. Run the public style checker from
the repository root:

.. code-block:: bash

   bash scripts/utils/style_check.sh

Build the documentation
-----------------------

The site uses Sphinx with the Book Theme. Install the documentation dependencies
in your working environment if necessary:

.. code-block:: bash

   python -m pip install sphinx myst-parser sphinx-book-theme
   python -m sphinx \
     -b html \
     -a \
     -W \
     --keep-going \
     docs/wiki \
     docs/wiki/_build/html
   python -m http.server \
     8081 \
     --bind 127.0.0.1 \
     --directory docs/wiki/_build/html

Open ``http://127.0.0.1:8081/``. The public documentation deployment also uses
``make html`` inside ``docs/wiki``. Keep existing page routes and links working,
include new pages in the toctree, and verify mobile navigation, search, theme
switching, code copy controls, and figure/table readability.
