#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root. Extra CLI options can be appended.
python -m src.tabstruct.experiment.run_experiment \
  --pipeline prediction \
  --task classification \
  --model lr \
  --dataset credit-g \
  --test_size 0.2 \
  --valid_size 0.1 \
  --test_id 0 \
  --valid_id 0 \
  --device cpu \
  --save_model \
  --tags tutorial-prediction \
  "$@"
