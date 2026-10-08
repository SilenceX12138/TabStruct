#!/usr/bin/env bash
set -euo pipefail

# First run generation/train.sh, then pass its synthetic_samples.csv path.
if [ "$#" -eq 0 ]; then
  echo "Usage: bash docs/tutorial/example_scripts/generation/eval.sh SYNTHETIC_CSV [CLI options]" >&2
  exit 2
fi
synthetic_path="$1"
shift

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
  --eval_only \
  --synthetic_data_path "$synthetic_path" \
  --disable_synthetic_data_validation \
  --enable_eval_structure \
  --tags tutorial-generation-eval \
  "$@"
