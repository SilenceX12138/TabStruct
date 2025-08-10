TabStruct: Measuring Structural Fidelity of Tabular Data
========================================================

.. |arxiv| image:: https://img.shields.io/badge/Arxiv-Paper-olivegreen
   :target: https://arxiv.org/abs/2503.09453

.. |license| image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0

.. |python| image:: https://img.shields.io/badge/python-3.10+-blue.svg
   :target: https://www.python.org/downloads/

|arxiv| |license| |python|

Overview
--------

**TabStruct** is an end-to-end benchmark for **tabular data generation, prediction, and evaluation**. It ships with ready-to-use pipelines for:

* generating high-quality synthetic tables
* training predictive models
* analysing results with a rich suite of metrics - especially those that quantify **structural fidelity**

All components are designed to plug-and-play, so you can mix, match, and extend them to suit your own workflow.

Key Features
------------

**Data generation**

* Out-of-the-box support for popular tabular generators: **SMOTE, TVAE, CTGAN, NFlow, TabDDPM, ARF**, and more.

**Evaluation dimensions**

* **Density estimation** – How well does the synthetic data approximate the real distribution?
* **Privacy preservation** – Does the generator leak sensitive records?
* **ML efficacy** – How do models trained on synthetic data perform compared to real data?
* **Structural fidelity** – Does the generator respect the causal structures of real data?

**Predictive tasks**

* Classification & regression pipelines built on **scikit-learn**, with optional neural-network backbones.

Installation
------------

We recommend managing dependencies with **conda** + **mamba**.

.. code-block:: bash

   # 1️⃣ Upgrade conda and activate the base env
   conda update -n base -c conda-forge conda
   conda activate base

   # 2️⃣ Install the high-performance dependency resolver
   conda install conda-libmamba-solver --yes
   conda config --set solver libmamba
   conda install -c conda-forge mamba --yes

   # 3️⃣ Create a new conda env
   conda create --name tabstruct python=3.10.18 --no-default-packages
   conda activate tabstruct

   # 4️⃣ Set up the env
   bash scripts/utils/install.sh

.. note::
   Search the codebase for absolute paths and replace them with paths on your machine.

Citation
--------

For attribution in academic contexts, please cite this work as:

.. code-block:: bibtex

   @inproceedings{jiang2025well,
     title={How Well Does Your Tabular Generator Learn the Structure of Tabular Data?},
     author={Jiang, Xiangjian and Simidjievski, Nikola and Jamnik, Mateja},
     booktitle={ICLR 2025 Workshop on Deep Generative Model in Machine Learning: Theory, Principle and Efficacy}
   }

Contents
--------

.. toctree::
	:maxdepth: 2
	:caption: Guide

	guide/overview
	guide/quickstart

.. toctree::
	:maxdepth: 2
	:caption: Reference

	reference/api
	reference/cli
	reference/models

