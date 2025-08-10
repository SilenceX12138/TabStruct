python -m src.tabstruct.experiment.run_experiment \
    --pipeline "generation" \
    --generation_only \
    --generation_ratio 10 \
    --model "smote" \
    --dataset "mfeat-fourier" \
    --test_size 0.2 \
    --valid_size 0.1 \
    --tags "dev"
