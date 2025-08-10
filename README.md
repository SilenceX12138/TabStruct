# TabStruct: Measuring Structural Fidelity of Tabular Data

<div align="center">

<img src="docs/wiki/_media/repo_logo.png" height="150px">

[![Arxiv-Paper](https://img.shields.io/badge/Arxiv-Paper-olivegreen)](https://arxiv.org/abs/2503.09453)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Downloads](https://static.pepy.tech/badge/tabstruct)](https://pypi.org/project/tabstruct/)

</div>

> [!IMPORTANT]
> Official code for the paper ["How Well Does Your Tabular Generator Learn the Structure of Tabular Data?"](https://arxiv.org/abs/2503.09453), published in Deep Generative Model in Machine Learning: Theory, Principle and Efficacy (ICLR 2025 Workshop).
>
> Authored by [Xiangjian Jiang](https://silencex12138.github.io/), [Nikola Simidjievski](https://simidjievskin.github.io/), [Mateja Jamnik](https://www.cl.cam.ac.uk/~mj201/), University of Cambridge, UK

## 📌 Overview

![TabStruct banner](https://s2.loli.net/2025/05/16/TZ1clpvNBDhi8AE.png)

**TabStruct** is an end‑to‑end benchmark for **tabular data generation, prediction, and evaluation**. It ships with ready‑to‑use pipelines for

- generating high‑quality synthetic tables,
- training predictive models, and
- analysing results with a rich suite of metrics – especially those that quantify **structural fidelity**.

All components are designed to plug‑and‑play, so you can mix, match, and extend them to suit your own workflow.

## 📖 Citation

For attribution in academic contexts, please cite this work as:

```
@inproceedings{jiang2025well,
  title={How Well Does Your Tabular Generator Learn the Structure of Tabular Data?},
  author={Jiang, Xiangjian and Simidjievski, Nikola and Jamnik, Mateja},
  booktitle={ICLR 2025 Workshop on Deep Generative Model in Machine Learning: Theory, Principle and Efficacy}
}
```

## 📚 Key Features

### Data generation

- Out‑of‑the‑box support for popular tabular generators: **SMOTE, TVAE, CTGAN, NFlow, TabDDPM, ARF**, and more.

### Evaluation dimensions

- **Density estimation** – How well does the synthetic data approximate the real distribution?
- **Privacy preservation** – Does the generator leak sensitive records?
- **ML efficacy** – How do models trained on synthetic data perform compared to real data?
- **Structural fidelity** – Does the generator respect the causal structures of real data?

### Predictive tasks

- Classification & regression pipelines built on **scikit‑learn**, with optional neural‑network backbones.

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

> **Heads‑up:** Search the codebase for absolute paths and replace them with paths on your machine.

## 📊 Logging with W\&B

TabStruct logs every experiment to **Weights & Biases** (W\&B). Use the default project or set your own credentials in `src/tabstruct/common/__init__.py`:

```python
WANDB_ENTITY  = "tabular-data-generation"
WANDB_PROJECT = "TabStruct"
```

## ✅ Quick sanity check

Run a toy classification job (K‑NN on the **Adult** dataset):

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

```text
[YYYY‑MM‑DD] Codebase: >>>>>>>>>> Launching create_data_module() <<<<<<<<<<<
…
```

If you see those, congratulations – your environment is ready! 🎉

## 💥 Example Workflows

### 1. Generate synthetic data

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

Template script: `docs/tutorial/example_scripts/generation/train.sh`.

### 2. Evaluate synthetic data

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

Template script: `docs/tutorial/example_scripts/generation/eval.sh`.

### 3. Predict on tabular data

```shell
python -m src.tabstruct.experiment.run_experiment \
	--model 'mlp' \
	--save_model \
	--max_steps_tentative 1500 \
	--dataset 'adult' \
	--test_size 0.2 \
	--valid_size 0.1 \
	--tags 'dev'
```

Template script: `docs/tutorial/example_scripts/prediction/train.sh`

<!--  -->
