import multiprocessing as mp
from abc import abstractmethod
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor as Pool
from multiprocessing import Manager

import numpy as np
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
from optuna.trial import TrialState

from src.tabstruct.common import TUNE_STUDY_TIMEOUT, WANDB_ENTITY, WANDB_PROJECT
from src.tabstruct.common.runtime.config.argument import AddOnlyNamespace
from src.tabstruct.common.runtime.log.TerminalIO import TerminalIO
from src.tabstruct.experiment.pipeline.PipelineHelper import PipelineHelper


class BaseTuner:
    def __init__(self, args) -> None:
        self.args = args

        # === Set the model to optimise ===
        self.model_class = PipelineHelper.pipeline_handler(self.args.pipeline).model_helper.model_handler(
            self.args.model
        )

    @abstractmethod
    def tune(self):
        raise NotImplementedError("This method has to be implemented by the sub class")


class OptunaTuner(BaseTuner):

    # ================================================================
    # =                                                              =
    # =                      Intialisation                           =
    # =                                                              =
    # ================================================================
    def __init__(self, args):
        super().__init__(args)

        self.optimization_direction = self.get_optimization_direction()
        self.pruner = optuna.pruners.MedianPruner() if self.args.optuna_pruning else optuna.pruners.NopPruner()

        # Log the used model_params in each trial
        self.param_list = []
        # Log the metric_dict in each trial
        self.metric_dict_list = []
        # Log the failed runs (this will be modified within each subprocess, and thus should be protected)
        manager = Manager()
        self.failed_run_list = manager.list()

    # ================================================================
    # =                                                              =
    # =                         Tuning                               =
    # =                                                              =
    # ================================================================
    def tune(self):
        # === Optuna study ===
        # Initialize the study
        sampler = optuna.samplers.TPESampler(seed=self.args.seed)
        study = optuna.create_study(direction=self.optimization_direction, sampler=sampler, pruner=self.pruner)
        # Start with the default hyperparameters
        study.enqueue_trial(params=self.model_class._define_default_params())
        # Optimize the study
        study.optimize(
            self.objective,
            n_trials=self.args.optuna_trial,
            timeout=TUNE_STUDY_TIMEOUT,
            # The current official callback is not compatible with multi-process optuna trial.
            # callbacks=self.get_callbacks(),
        )

        metric_dict_best = self.parse_metric_dict_best(study)
        return metric_dict_best

    def objective(self, trial):
        # === Set the trial parameters ===
        args_list = self.get_args_for_single_trial(trial)

        # === Perform the single trial ===
        metric_dict = self.single_trial(args_list)
        score = metric_dict["valid_metrics"][self.args.metric_model_selection]

        # === Log the metric values for each trial ===
        self.metric_dict_list.append(metric_dict)

        return score

    # ================================================================
    # =                                                              =
    # =                     Trials and Runs                          =
    # =                                                              =
    # ================================================================
    def get_args_for_single_trial(self, trial):
        # === Define model parameters ===
        model_params = self.model_class.define_params(reg_test=False, trial=trial, dev="dev" in self.args.tags)
        # Prune the run if the hyperparameters are already used
        if trial.params in self.param_list:
            self.prune_used_params(trial)
        else:
            self.param_list.append(trial.params)

        # === Repeated cross-validation ===
        new_args_list = []
        for test_id in range(self.args.num_repeats):
            for valid_id in range(self.args.num_cv_folds):
                temp_args = Namespace(**vars(self.args))
                # === Remove the arguments for tuning ===
                temp_args.enable_optuna = False
                delattr(temp_args, "_runtime")

                # === Set the new test_id and valid_id ===
                temp_args.test_id = test_id
                temp_args.valid_id = valid_id

                # === Add runtime args ===
                temp_args = AddOnlyNamespace(**vars(temp_args))
                temp_args.model_params = model_params

                new_args_list.append(temp_args)

        return new_args_list

    def single_trial(self, args_list):
        with Pool(mp_context=mp.get_context("spawn"), max_workers=self.args.tune_max_workers) as pool:
            # 1. The metric value order matches the input order (https://discuss.python.org/t/is-pool-map-return-the-same-order-as-the-input-order/18897)
            # 2. When once subprocess crashes, the other subprocesses will crash, and thus exceptions should be handled.
            metric_dict_list = list(
                pool.map(OptunaTuner.single_run, args_list, [self.failed_run_list] * len(args_list))
            )

        metric_dict_agg = self.reduce_metric_dict_list(metric_dict_list)

        return metric_dict_agg

    # multi-processing only allows "global" functions to be executed in Pool, which means the function should be valid outside the range of Pool -> not tied to an instance
    @classmethod
    def single_run(cls, args, failed_run_list):
        metric_dict = {}
        try:
            # Lazy import to avoid circular import between run_experiment and tuning
            from ..run_experiment import run_experiment

            metric_dict = run_experiment(args)
        except Exception as e:
            failed_run_list.append(
                {
                    "model": args.model,
                    "dataset": args.dataset,
                    "test_id": args.test_id,
                    "valid_id": args.valid_id,
                }
            )
            TerminalIO.print(f"Failed run: {args.model}-{args.dataset}-{args.test_id}-{args.valid_id}", TerminalIO.FAIL)
            TerminalIO.print(f"Error: {e}", TerminalIO.FAIL)

        return metric_dict

    # ================================================================
    # =                                                              =
    # =                         Utils                                =
    # =                                                              =
    # ================================================================
    def reduce_metric_dict_list(self, metric_dict_list):
        metric_dict_agg = {}
        for metric_dict in metric_dict_list:
            for split, split_dict in metric_dict.items():
                metric_dict_agg[split] = metric_dict_agg.get(split, {})
                for metric, value in split_dict.items():
                    if metric not in metric_dict_agg[split]:
                        metric_dict_agg[split][metric] = []
                    metric_dict_agg[split][metric].append(value)

        for split, split_dict in metric_dict_agg.items():
            for metric, metric_list in split_dict.items():
                metric_dict_agg[split][metric] = self.reduce_metric_list(metric_list)

        return metric_dict_agg

    def reduce_metric_list(self, metric_list):
        if self.args.tune_reduction == "mean":
            metric = np.mean(metric_list)
        elif self.args.tune_reduction == "median":
            metric = np.median(metric_list)
        elif self.args.tune_reduction == "max":
            if self.optimization_direction != "maximize":
                raise ValueError("The metric is not maximized, and thus you cannot use max reduction.")
            metric = np.max(metric_list)
        elif self.args.tune_reduction == "min":
            if self.optimization_direction != "minimize":
                raise ValueError("The metric is not minimized, and thus you cannot use min reduction.")
            metric = np.min(metric_list)
        else:
            raise ValueError(f"Reduction method {self.args.tune_reduction} is not supported.")

        return metric

    def get_optimization_direction(self):
        if self.args.metric_model_selection in ["balanced_accuracy"]:
            return "maximize"
        else:
            return "minimize"

    def get_callbacks(self):
        callbacks = []

        # Log the hyperparameters and metric for model_selection to W&B
        callbacks.append(
            WeightsAndBiasesCallback(
                metric_name=self.args.metric_model_selection,
                wandb_kwargs={
                    "entity": WANDB_ENTITY,
                    "project": WANDB_PROJECT,
                },
            )
        )

        return callbacks

    def prune_used_params(self, trial):
        # Q: Why does optuna sample the same hyperparameters again?
        # A: https://stackoverflow.com/questions/64836142/optuna-suggests-the-same-parameter-values-in-a-lot-of-trials-duplicate-trials-t
        used_trial_number = self.param_list.index(trial.params)

        self.param_list.append(trial.params)
        self.metric_dict_list.append(self.metric_dict_list[used_trial_number])

        trial.report(self.metric_dict_list[-1]["valid_metrics"][self.args.metric_model_selection], step=trial.number)
        raise optuna.exceptions.TrialPruned()

    def parse_metric_dict_best(self, study):
        best_trial = study.best_trial
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        metric_dict = {
            "dicrection": str(study.direction),
            "metric_model_selection": self.args.metric_model_selection,
            "num_total_trials": len(study.trials),
            "num_pruned_trials": len(pruned_trials),
            "num_finished_trials": len(complete_trials),
            "best_trial": {
                "number": best_trial.number,
                "model_params": best_trial.params,
                "metric_dict": self.metric_dict_list[best_trial.number],
            },
        }

        return metric_dict
