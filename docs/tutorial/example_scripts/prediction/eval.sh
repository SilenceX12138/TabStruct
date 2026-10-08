#!/usr/bin/env bash
set -euo pipefail

# First run prediction/train.sh, then pass its saved lr.pkl path.
if [ "$#" -eq 0 ]; then
  echo "Usage: bash docs/tutorial/example_scripts/prediction/eval.sh CHECKPOINT [CLI options]" >&2
  exit 2
fi
checkpoint_path="$1"
shift

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
  --eval_only \
  --saved_checkpoint_path "$checkpoint_path" \
  --tags tutorial-prediction-eval \
  "$@"
