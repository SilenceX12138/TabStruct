python -m src.tabstruct.experiment.run_experiment \
	--model 'mlp' \
	--save_model \
	--max_steps_tentative 1500 \
	--dataset 'adult' \
	--test_size 0.2 \
	--valid_size 0.1 \
	--tags 'dev'
