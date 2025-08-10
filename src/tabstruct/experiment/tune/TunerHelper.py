import pandas as pd
import wandb

from src.tabstruct.common.runtime.log.TerminalIO import TerminalIO

from .BaseTuner import OptunaTuner


class TunerHelper:

    @classmethod
    @TerminalIO.trace_func
    def tune_model(cls, args):
        # === Initialize the tuner ===
        tuner = cls.create_tuner(args)

        # === Tune the model ===
        metric_dict_best = tuner.tune()

        # === Log the tuning results ===
        cls.log_tuning_results(tuner, metric_dict_best)

        return metric_dict_best

    @classmethod
    def create_tuner(cls, args):
        tuner = OptunaTuner(args)

        return tuner

    @classmethod
    def log_tuning_results(cls, tuner, metric_dict_best):
        # === Log the best trial ===
        wandb.run.summary.update(metric_dict_best)

        # === Log the tuning trace ===
        for i in range(len(tuner.param_list)):
            log_dict_step = {}

            # Log the params for each trial
            param_dict = tuner.param_list[i]
            log_dict_step = log_dict_step | {
                f"param/{param_name}": param_value for param_name, param_value in param_dict.items()
            }

            # Log metric values for all trials
            metric_dict = tuner.metric_dict_list[i]
            for split_name in metric_dict.keys():
                log_dict_step = log_dict_step | {
                    f"{split_name}/{metric_name}": metric_value
                    for metric_name, metric_value in metric_dict[split_name].items()
                }
            # The last step of wandb.log goes to summary automatically
            # the step of log is determined by the number of times wandb.log is called
            wandb.log(log_dict_step)

        # === Log the failed trials ===
        failed_run_dict = {}
        for failed_run in tuner.failed_run_list:
            for key, value in failed_run.items():
                if key not in failed_run_dict:
                    failed_run_dict[key] = []
                failed_run_dict[key].append(value)
        wandb.log({"failed_runs": wandb.Table(dataframe=pd.DataFrame(failed_run_dict))})
