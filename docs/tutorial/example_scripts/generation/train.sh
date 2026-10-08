#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root. Save a CSV before invoking generation/eval.sh.
python -m src.tabstruct.experiment.run_experiment \
  --pipeline generation \
  --task classification \
  --model smote \
  --dataset credit-g \
  --test_size 0.2 \
  --valid_size 0.1 \
  --test_id 0 \
  --valid_id 0 \
  --device cpu \
  --generation_only \
  --generation_ratio 1 \
  --tags tutorial-generation \
  "$@"
