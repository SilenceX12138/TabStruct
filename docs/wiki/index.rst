.. _tabstruct-tabular-structural-fidelity:

.. _overview:

TabStruct Documentation
=======================

.. raw:: html

   <section class="home-banner" aria-labelledby="home-title">
     <div class="banner-title">
       <h1 id="home-title"><span class="brand-tab">Tab</span><span class="brand-struct">Struct</span></h1>
       <p class="hero-subtitle">Structural Fidelity of Tabular Data</p>
       <h2>A shared benchmark for tabular generation and evaluation</h2>
       <p class="hero-description">Measure synthetic tables across <strong>density estimation</strong>, <strong>privacy preservation</strong>, <strong>ML efficacy</strong>, and <strong>structural fidelity</strong>.</p>
       <nav class="home-actions" aria-label="Documentation shortcuts">
         <a class="doc-button primary" href="guide/quickstart.html">Get started <span aria-hidden="true">→</span></a>
         <a class="doc-button secondary" href="https://arxiv.org/abs/2509.11950">ICLR 2026 Oral</a>
         <a class="doc-button secondary" href="https://github.com/SilenceX12138/TabStruct">Codebase</a>
       </nav>
     </div>
     <figure class="workflow-figure" id="id1">
       <a class="workflow-viewport" href="_static/workflow.svg" aria-label="Open the TabStruct workflow at full size">
         <img src="_static/workflow.svg" alt="Prepare data with TabCamel, then generate or predict; evaluate synthetic data with TabEval and measure ML efficacy through synthetic-training prediction." width="1100" height="410">
       </a>
       <figcaption><a href="https://github.com/SilenceX12138/TabCamel">TabCamel</a> prepares the data; <a href="https://github.com/SilenceX12138/TabEval">TabEval</a> supplies synthetic-data evaluators.</figcaption>
     </figure>
   </section>

.. _key-features:

.. _data-generation:

.. _evaluation-dimensions:

.. _predictive-tasks:

.. _example-workflows:

.. _generate-synthetic-data:

.. _evaluate-synthetic-data:

.. _predict-on-tabular-data:

Choose a workflow
-----------------

.. raw:: html

   <div class="task-grid" aria-label="TabStruct workflows">
     <section class="task-card">
       <div class="task-card-body">
         <h3>Generate a table</h3>
         <p>Create synthetic tables with the original feature and target schema.</p>
         <p class="task-scenario">Compare generators for a research data-sharing workflow.</p>
         <a class="task-api" href="reference/api.html#basegenerator">Generator API <span aria-hidden="true">↗</span></a>
       </div>
       <a class="task-art" href="guide/tutorials.html#generate-and-evaluate-a-table" aria-label="Open the generation tutorial">
         <img src="_static/generation.svg" alt="An input table becomes a synthetic table" width="240" height="120">
       </a>
       <a class="task-examples" href="guide/tutorials.html#generate-and-evaluate-a-table">Generation tutorial <span aria-hidden="true">→</span></a>
     </section>
     <section class="task-card">
       <div class="task-card-body">
         <h3>Evaluate synthetic data</h3>
         <p>Assess density, privacy, ML efficacy, and structural fidelity.</p>
         <p class="task-scenario">Check whether a synthetic table retains relationships across columns.</p>
         <a class="task-api" href="guide/evaluation.html">TabEval integration <span aria-hidden="true">↗</span></a>
       </div>
       <a class="task-art" href="guide/evaluation.html" aria-label="Open the four evaluation dimensions">
         <img src="_static/evaluation.svg" alt="Four complementary evaluation dimensions" width="240" height="120">
       </a>
       <a class="task-examples" href="guide/evaluation.html">Evaluation guide <span aria-hidden="true">→</span></a>
     </section>
     <section class="task-card">
       <div class="task-card-body">
         <h3>Benchmark a predictor</h3>
         <p>Train classifiers or regressors and compare held-out performance.</p>
         <p class="task-scenario">Establish a credit classification baseline and restore its checkpoint.</p>
         <a class="task-api" href="reference/api.html#basepredictor">Predictor API <span aria-hidden="true">↗</span></a>
       </div>
       <a class="task-art" href="guide/tutorials.html#predict-and-restore" aria-label="Open the prediction tutorial">
         <img src="_static/prediction.svg" alt="Tabular features map to target predictions" width="240" height="120">
       </a>
       <a class="task-examples" href="guide/tutorials.html#predict-and-restore">Prediction tutorial <span aria-hidden="true">→</span></a>
     </section>
   </div>

.. _installation:

.. _logging-with-w-b:

.. _quick-sanity-check:

First run
---------

After :doc:`installation and W&B setup <guide/quickstart>`, run from the
repository root:

.. code-block:: bash

   python -m src.tabstruct.experiment.run_experiment \
     --pipeline prediction \
     --task classification \
     --model lr \
     --dataset credit-g \
     --device cpu \
     --save_model \
     --tags tutorial-prediction

The runner returns split-specific metrics and records ``best_model_path``
when saving a model. Continue with the :doc:`tutorials <guide/tutorials>` or
browse the :doc:`model registry <reference/models>`.

.. _contents:

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Learn

   guide/quickstart
   guide/overview
   guide/data
   guide/workflows
   guide/evaluation
   guide/tutorials

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Reference

   reference/cli
   reference/api
   reference/models

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Development

   guide/development

.. _citation:

.. _citation-and-license:

Citations
---------

.. code-block:: bibtex

   @inproceedings{jiang2026tabstruct,
     title={TabStruct: Measuring Structural Fidelity of Tabular Data},
     author={Jiang, Xiangjian and Simidjievski, Nikola and Jamnik, Mateja},
     booktitle={The Fourteenth International Conference on Learning Representations},
     year={2026}
   }

   @inproceedings{jiang2025well,
     title={How Well Does Your Tabular Generator Learn the Structure of Tabular Data?},
     author={Jiang, Xiangjian and Simidjievski, Nikola and Jamnik, Mateja},
     booktitle={ICLR 2025 Workshop on Deep Generative Models in Machine Learning: Theory, Principle and Efficacy},
     year={2025}
   }
