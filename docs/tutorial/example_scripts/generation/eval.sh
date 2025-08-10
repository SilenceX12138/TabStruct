python -m src.tabstruct.experiment.run_experiment \
	--pipeline "generation" \
	--model "smote" \
	--eval_only \
	--dataset "mfeat-fourier" \
	--test_size 0.2 \
	--valid_size 0.1 \
	--generator_tags "dev" \
	--curate_ratio 10 \
	--tags "dev"
