# [ICLR 2026 Oral] TabStruct – Tabular Structural Fidelity

<div align="center">

<img src="docs/wiki/_media/repo_logo_landscape.png" width="60%">

[![Arxiv-Paper](https://img.shields.io/badge/Arxiv-Paper-olivegreen?style=for-the-badge)](https://arxiv.org/abs/2509.11950)
[![Docs](https://img.shields.io/badge/Docs-Wiki-blue?style=for-the-badge)](https://silencex12138.github.io/TabStruct/)
[![CI Status](https://img.shields.io/github/actions/workflow/status/SilenceX12138/TabStruct/style_check.yaml?branch=master&style=for-the-badge)](https://github.com/SilenceX12138/TabStruct/actions/workflows/style_check.yaml?branch=master)
[![PyPI version](https://img.shields.io/pypi/v/tabstruct?style=for-the-badge)](https://badge.fury.io/py/tabstruct)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)

</div>

> [!IMPORTANT]
> Official code for the paper ["TabStruct: Measuring Structural Fidelity of Tabular Data"](https://arxiv.org/abs/2509.11950), published in The Fourteenth International Conference on Learning Representations (ICLR 2026 Oral).
>
> TabStruct provides the full experimental pipeline used in the paper, including generation, predictive modelling, and evaluation protocols for structural fidelity.
>
> Authored by [Xiangjian Jiang](https://silencex12138.github.io/), [Nikola Simidjievski](https://simidjievskin.github.io/), [Mateja Jamnik](https://www.cl.cam.ac.uk/~mj201/), University of Cambridge, UK

## 📌 Overview

![TabStruct banner](https://s2.loli.net/2025/05/16/TZ1clpvNBDhi8AE.png)

**TabStruct** is an end‑to‑end benchmark for **tabular data generation, prediction, and evaluation**. It ships with ready‑to‑use pipelines for

- **generating** high‑quality synthetic tables,
- **predicting** with machine learning models, and
- **analysing** results with a rich suite of metrics – especially those that quantify **structural fidelity**.

The benchmark is designed for both research and applied workflows: you can run standard baselines out of the box, or plug in custom generators/predictors and fairly evaluate them under the same protocol.
All components are designed to plug‑and‑play, so you can mix, match, and extend them to suit your own workflow.

## 📚 Key Features

### Data generation

- Out‑of‑the‑box support for popular tabular generators: **SMOTE, TVAE, CTGAN, NFlow, TabDDPM, ARF**, and more.
- Supports customised setups (classical oversampling, deep generative models, and probabilistic approaches) so different modelling assumptions can be compared under one interface.

### Evaluation dimensions

- **Density estimation** – How well does the synthetic data approximate the real distribution?
- **Privacy preservation** – Does the generator leak sensitive records?
- **ML efficacy** – How do models trained on synthetic data perform compared to real data?
- **Structural fidelity** – Does the generator respect the causal structures of real data?

### Predictive tasks

- Classification & regression pipelines built on **scikit‑learn**, with optional neural‑network backbones.
- Unified training/evaluation entry points make it straightforward to benchmark models across datasets with consistent splits, logging, and reproducibility settings.

## 🚀 Installation

We recommend managing dependencies with **conda** + **mamba**.

```bash
# 1️⃣ Upgrade conda and activate the base env
conda update -n base -c conda-forge conda
conda activate base

# 2️⃣ Install the high‑performance dependency resolver
conda install conda-libmamba-solver --yes
conda config --set solver libmamba
conda install -c conda-forge mamba --yes

# 3️⃣ Create a new conda env
conda create --name tabstruct python=3.10.18 --no-default-packages
conda activate tabstruct

# 4️⃣ Set up the env
bash scripts/utils/install.sh
```

## 📊 Logging with W\&B

TabStruct logs every experiment to **Weights & Biases** (W\&B). Use the default project or set your own credentials in `src/tabstruct/common/__init__.py`:

```python
WANDB_ENTITY  = "tabular-data-generation"
WANDB_PROJECT = "TabStruct"
```

## ✅ Quick sanity check

<details>

<summary>Run a toy classification job (K‑NN on the <b>Adult</b> dataset):</summary>

```bash
python -m src.tabstruct.experiment.run_experiment \
  --model knn \
  --save_model \
  --dataset adult \
  --test_size 0.2 \
  --valid_size 0.1 \
  --tags ENV-TEST
```

A successful run prints a series of **green** log lines like:

```
[YYYY‑MM‑DD] Codebase: >>>>>>>>>> Launching create_data_module() <<<<<<<<<<<
…
```

If you see those, congratulations – your environment is ready! 🎉

</details>

## 💥 Example Workflows

### 1. Generate synthetic data

<details>
<summary>Template script: <em>docs/tutorial/example_scripts/generation/train.sh</em></summary>

```bash
python -m src.tabstruct.experiment.run_experiment \
    --pipeline "generation" \
    --generation_only \
    --model "smote" \
    --dataset "mfeat-fourier" \
    --test_size 0.2 \
    --valid_size 0.1 \
    --tags "dev"
```

</details>

### 2. Evaluate synthetic data

<details>
<summary>Template script: <em>docs/tutorial/example_scripts/generation/eval.sh</em></summary>

```bash
python -m src.tabstruct.experiment.run_experiment \
	--pipeline "generation" \
	--model "smote" \
	--eval_only \
	--dataset "mfeat-fourier" \
	--test_size 0.2 \
	--valid_size 0.1 \
	--generator_tags "dev" \
	--tags "dev"
```

</details>

### 3. Predict on tabular data

<details>
<summary>Template script: <em>docs/tutorial/example_scripts/prediction/train.sh</em></summary>

```bash
python -m src.tabstruct.experiment.run_experiment \
	--model 'mlp' \
	--save_model \
	--max_steps_tentative 1500 \
	--dataset 'adult' \
	--test_size 0.2 \
	--valid_size 0.1 \
	--tags 'dev'
```

</details>

## 📖 Citation

For attribution in academic contexts, please cite this work as:

```bibtex
@inproceedings{jiang2026tabstruct,
  title={TabStruct: Measuring Structural Fidelity of Tabular Data},
  author={Jiang, Xiangjian and Simidjievski, Nikola and Jamnik, Mateja},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}


@inproceedings{jiang2025well,
  title={How Well Does Your Tabular Generator Learn the Structure of Tabular Data?},
  author={Jiang, Xiangjian and Simidjievski, Nikola and Jamnik, Mateja},
  booktitle={ICLR 2025 Workshop on Deep Generative Model in Machine Learning: Theory, Principle and Efficacy}
}
```

<!--  -->
