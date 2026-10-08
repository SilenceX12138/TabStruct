.. _models:

Models Reference
================

Model IDs are registered in ``src/tabstruct/common/__init__.py`` and resolved
by ``PredictorHelper`` or ``GeneratorHelper``. Use the exact lowercase CLI IDs
below. Registrations describe available adapters; task support and dependencies
are enforced by each wrapper.

.. _prediction:

Prediction Models
-----------------

.. list-table:: ``--pipeline prediction``
   :header-rows: 1
   :widths: 22 34 44

   * - ID
     - Adapter
     - Backend / notes
   * - ``lr``
     - ``LinearModel``
     - Logistic regression for classification, linear regression for regression.
   * - ``rf``
     - ``RandomForest``
     - Scikit-learn classifier or regressor.
   * - ``knn``
     - ``KNN``
     - Scikit-learn; reconstructed from training data for evaluation.
   * - ``xgb``
     - ``XGBoost``
     - XGBoost classifier or regressor.
   * - ``mlp-sklearn``
     - ``MLPSklearn``
     - Scikit-learn multilayer perceptron.
   * - ``tabnet``
     - ``TabNet``
     - TabNet dependency required.
   * - ``tabpfn``
     - ``TabPFN``
     - TabPFN estimator; weights and backend requirements apply.
   * - ``tabforge``
     - ``TabFORGE``
     - Standalone TabFORGE classifier/regressor; requires model-specific preprocessing.
   * - ``mlp``
     - ``LitMLP``
     - Lightning feed-forward network.
   * - ``ft-transformer``
     - ``LitFTTransformer``
     - Lightning feature-tokenizer transformer.

.. _generation:

Generation Models
-----------------

.. list-table:: ``--pipeline generation``
   :header-rows: 1
   :widths: 22 34 44

   * - ID
     - Adapter
     - Backend / notes
   * - ``real`` / ``real-test``
     - ``Real``
     - Real training/test tables as evaluation references.
   * - ``smote``
     - ``SMOTE``
     - Imbalanced-learn interpolation; reference-data based.
   * - ``ctgan``
     - ``CTGAN``
     - TabEval conditional GAN plugin.
   * - ``tvae``
     - ``TVAE``
     - TabEval variational autoencoder plugin.
   * - ``bn``
     - ``BN``
     - TabEval Bayesian network plugin.
   * - ``goggle``
     - ``GOGGLE``
     - TabEval graph-based generator plugin.
   * - ``tabddpm``
     - ``TabDDPM``
     - TabEval TabDDPM plugin.
   * - ``arf``
     - ``ARF``
     - Adversarial random forests via TabEval.
   * - ``nflow``
     - ``NFLOW``
     - TabEval normalizing-flow plugin.
   * - ``great``
     - ``GReaT``
     - Language-model generator; backend and weight requirements apply.
   * - ``tabebm``
     - ``TabEBM``
     - Energy-based generator; reference-data based.
   * - ``nrgboost``
     - ``NRGBoost``
     - NRGBoost package adapter.
   * - ``tabular-argn``
     - ``TabularARGN``
     - Mostly AI autoregressive generator.
   * - ``ae`` / ``vae``
     - ``LitAE`` / ``LitVAE``
     - Lightning autoencoder / variational autoencoder.
   * - ``ddpm`` / ``tddpm``
     - ``LitDDPM`` / ``LitTDDPM``
     - Custom Lightning diffusion adapters.
   * - ``edm`` / ``vesde`` / ``vpsde``
     - ``LitEDM`` / ``LitVESDE`` / ``LitVPSDE``
     - Continuous diffusion adapters.
   * - ``tabsyn`` / ``tabdiff``
     - ``LitTabSyn`` / ``LitTabDiff``
     - TabSyn latent diffusion / TabDiff mixed-type diffusion.
   * - ``tabforge``
     - ``TabFORGE``
     - Standalone TabFORGE generator; requires model-specific preprocessing.

.. _selecting-parameters:

.. _caveats:

Choose and configure an adapter
-------------------------------

Begin with a CPU baseline such as ``lr`` or ``smote``. Add a deep or pretrained
model after confirming its dependencies, task support, and hardware needs.
The installation script supplies additional wheels for several backends;
registered IDs do not imply that every optional backend is installed.

The current TabPFN wrapper stops classification runs with more than 10 classes
and datasets with more than 500 processed features. It samples at most 10,000
training rows before fitting. These are wrapper constraints even when the
installed TabPFN backend supports larger inputs.

Ordinary parameters come from ``_define_default_params``; tuning calls
``_define_optuna_params``. The CLI has shared training/preprocessing flags
rather than an arbitrary parameter
JSON option. For lower-level Python integration, helpers accept resolved
``args.model_params`` before model creation.

TabFORGE is registered separately in each pipeline. Both adapters default to
pretrained initialization and require ``--model_specific_preprocessing``.
See :doc:`../guide/workflows` for installation and a launch example.

Paper and current registry
--------------------------

The published experimental roster and the current registry are different
scopes. New adapters and experimental latent variants are implementation
extensions. Use the paper's generator, dataset, and evaluation configurations
when reproducing its results; use this page to select a current CLI adapter.
