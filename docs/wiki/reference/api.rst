API Reference
=============

TabStruct is organized around experiment wrappers. Examples below use imports
from ``src.tabstruct`` and assume execution from a prepared repository checkout.
The model adapters consume resolved runtime arguments and prepared data; they
do not expose a top-level ``tabstruct.fit(X, y)`` estimator API.

.. _usage-examples:

Experiment entry point
----------------------

.. py:function:: run_experiment(args=None)

   Import from ``src.tabstruct.experiment.run_experiment``. ``args`` can be a
   list of CLI tokens or ``None`` to parse the process command line. Runtime
   setup initializes logging, parses required options, and fixes seeds.

   Returns split metric dictionaries for prediction and evaluated generation:
   ``{"train_metrics": {...}, "valid_metrics": {...}, "test_metrics": {...}}``.
   Generation-only runs return ``{}``. Handled manual-stop/timeout conditions can
   also yield an empty dictionary; inspect terminal output and run state.

.. code-block:: python

   from src.tabstruct.experiment.run_experiment import run_experiment

   metrics = run_experiment([
       "--pipeline", "prediction",
       "--task", "classification",
       "--model", "lr",
       "--dataset", "credit-g",
       "--device", "cpu",
       "--tags", "tutorial-api",
   ])
   print(metrics["test_metrics"])

Core interfaces
---------------

BaseModel
~~~~~~~~~

Import ``BaseModel`` from ``src.tabstruct.common.model.BaseModel``.

.. py:class:: BaseModel(args)

   Retains resolved runtime arguments and ``args.model_params``. Concrete
   wrappers create ``self.model``. Instantiate through a model helper when
   preprocessing and runtime-derived fields are needed.

   .. py:method:: fit(data_module)

      Calls the adapter's ``_fit`` hook. Requires a prepared ``DataModule``.

   .. py:method:: eval()

      Sets a wrapped Torch module to evaluation mode when applicable.

   .. py:method:: get_metadata()

      Returns ``{"name": class_name, "params": model_params}``.

   .. raw:: html

      <span id="BaseModel.define_params"></span>

   .. py:classmethod:: get_model_specific_scaler_config()

      Returns ``context``, ``feature_scaler``, and ``target_scaler`` groups.

Prediction Models
-----------------

BasePredictor
~~~~~~~~~~~~~

Import from ``src.tabstruct.prediction.models.BasePredictor``. Concrete
``BaseSklearnPredictor`` and ``BaseLitPredictor`` bases implement estimator or
Lightning training behavior.

.. py:class:: BasePredictor(args)

   Inherits ``BaseModel``. Call ``fit`` before inference or restore a fitted
   wrapper using the helper's persistence API.

   .. py:method:: predict(X)

      Returns class indices or regression predictions, normally ``(n_rows,)``.
      The shared pipeline uses processed array inputs; adapters owning raw
      schemas can accept their DataFrame views.

   .. py:method:: predict_proba(X)

      Classification returns ``(n_rows, n_classes)`` aligned to encoded class
      order. Shared regression adapters return ``None``.

   .. py:method:: feature_selection(X=None)

      Adapter-specific feature output. Only call for a supporting model.

Generation Models
-----------------

BaseGenerator
~~~~~~~~~~~~~

Import from ``src.tabstruct.generation.models.BaseGenerator``. This abstract
base provides count and class-distribution logic; concrete strategy bases
supply joint, conditional, or class-focused sampling.

.. py:class:: BaseGenerator(args)

   Inherits ``BaseModel``. Shared fitting combines processed features and the
   target (for supervised tasks), prepares conditions, and calls the model hook.
   Some adapters, including TabFORGE, override fitting to own raw schema handling.

   .. py:method:: generate()

      Generates rows using ``generation_num_samples`` or ``generation_ratio``
      and the configured class proportions. Returns the adapter's tabular
      output; the helper normalizes supported DataFrame or ``X_syn``/``y_syn``
      dictionary formats before restoring original columns and exporting CSV.
      This wrapper method has no ``n_samples`` positional argument.

   .. py:method:: compute_class2synthetic_samples()

      Returns a dictionary of class identifiers and requested sample counts.

Data Management
---------------

DataHelper
~~~~~~~~~~

Import from ``src.tabstruct.common.data.DataHelper``.
``create_data_module(args)`` loads, splits, curates, and preprocesses a TabCamel
dataset, adds runtime metadata, and returns a ``DataModule``.
``split_full_dataset(args, full_set)`` returns train/validation/test datasets
and their index arrays. ``recover_original_data(args, X, y)`` reverses fitted
transforms for CSV export and returns ``X_original`` and ``y_original``.

DataModule
~~~~~~~~~~

.. py:class:: DataModule(args, train_set, valid_set, test_set)

   Import from ``src.tabstruct.common.data.DataModule``. The split arguments
   are ``TabularDataset`` objects, rather than separate ``X_train``/``y_train``
   constructor keywords.

   ``X_train_df``, ``X_valid_df``, and ``X_test_df`` retain feature DataFrames.
   Supervised ``y_*_df`` views retain the target column. ``X_*`` and ``y_*``
   expose NumPy arrays when the dataset is tensor-compatible and otherwise
   preserve the underlying frame; targets are ``None`` for ``unsupervision``.

   ``train_dataloader()``, ``val_dataloader()``, and ``test_dataloader()``
   return Lightning-compatible loaders. Batches contain ``(X, y, indices)``
   with ``y=None`` for unsupervised data.

.. _basepipeline:

Pipeline Classes
----------------

``PipelineHelper.pipeline_handler(pipeline)`` returns ``PredictionPipeline``
or ``GenerationPipeline``. ``run_pipeline(args)`` delegates to that pipeline's
``run`` method. The pipeline chooses ``PredictorHelper`` or ``GeneratorHelper``;
both inherit from ``BaseModelHelper``.

.. raw:: html

   <span id="notes"></span>

Model helpers and persistence
-----------------------------

.. list-table:: Class methods
   :header-rows: 1
   :widths: 40 60

   * - Method
     - Contract
   * - ``model_handler(model)``
     - Resolve the string identifier to its adapter class.
   * - ``benchmark_model(args)``
     - Prepare, fit/restore, infer, and evaluate; return metrics.
   * - ``prepare_data(args)``
     - Resolve preprocessing and training settings; return ``DataModule``.
   * - ``fit_model(args, data_module)``
     - Return a fitted/restored wrapper, or ``None`` for CSV-only generation evaluation.
   * - ``inference(data_module, model)``
     - Return split prediction dictionaries or generator tabular output.
   * - ``save_model(model)``
     - Save a pickle under the active run directory; return its path.
   * - ``load_model(checkpoint_path)``
     - Read the saved pickle. Dataset/preprocessing preparation remains the caller's responsibility.

Prediction inference maps each split to ``{"y_pred": ..., "y_hat": ...}``.
Generation CSVs are saved with original columns and supervised target values.
Saved wrappers retain model state and arguments, including preprocessing
information; the CLI prepares the dataset again and assigns current arguments
when loading. Keep split and preprocessing settings consistent.

The helper saves a marker for ``knn``, ``smote``, and ``tabebm`` and refits them
from reference training rows. See :doc:`../guide/workflows` for restore commands.

Hyperparameter Tuning
---------------------

``TunerHelper.tune_model(args)`` constructs an ``OptunaTuner``, runs its study,
logs trial metrics, and returns the best trial's metric dictionary.
``--metric_model_selection`` names a validation metric; repeat/fold options
control the nested experiment runs. Consult :doc:`cli` for defaults.

Experiment Configuration
------------------------

``parse_arguments(args)`` in ``common/runtime/config/argument.py`` builds an
``AddOnlyNamespace`` after initializing W&B and resolving interacting options.
``setup_runtime(args)`` additionally sets logging and seeds. Parsing a list of
CLI tokens therefore has logging and provenance-lookup side effects; it is
not a pure configuration reader.

``AddOnlyNamespace`` allows adding derived runtime fields while preventing
replacement/deletion of existing fields. The model helper resolves preprocessing
and training fields after loading the data. Use ``run_experiment`` for the
complete setup sequence.

Constants and Configuration
---------------------------

``BASE_DIR``, ``LOG_DIR``, ``WANDB_ENTITY``, ``WANDB_PROJECT``,
``SINGLE_RUN_TIMEOUT``, and ``TUNE_STUDY_TIMEOUT`` are defined in
``src/tabstruct/common/__init__.py``. The same module owns the model registries
and the list of metrics whose tuning objective is maximized.

Error Handling
--------------

``ManualStopError`` marks an expected stop such as an unsupported adapter
configuration; the runner handles it alongside timeouts. Other exceptions are
raised after its cleanup/logging step. Inspect run state and terminal output
when the returned metric dictionary is empty.
