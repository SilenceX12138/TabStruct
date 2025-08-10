python -m src.tabstruct.experiment.run_experiment \
	--model 'mlp' \
	--eval_only \
	--saved_checkpoint_path "/path/to/checkpoint" \
	--dataset 'adult' \
	--test_size 0.2 \
	--valid_size 0.1 \
	--tags 'dev'
