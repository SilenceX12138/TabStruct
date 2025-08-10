import wandb

from src.tabstruct.common.runtime.config.env import setup_runtime
from src.tabstruct.common.runtime.error.ManualStopError import ManualStopError
from src.tabstruct.common.runtime.log.TerminalIO import TerminalIO

# Because PipelineHelper includes the repos for almost all runs, if we use lazy import, it could lead to change in results, and affecting the reproducibility.
from .pipeline.PipelineHelper import PipelineHelper


@TerminalIO.trace_func
def run_experiment(args=None):
    metric_dict = {}
    caught_exception = None
    try:
        # === Set up the env for experiments ===
        args = setup_runtime(args)

        # === Run the experiment ===
        if args.enable_optuna:
            # Use lazy import to avoid circular import between run_experiment and tuning
            from .tune.TunerHelper import TunerHelper

            metric_dict = TunerHelper.tune_model(args)
        else:
            metric_dict = PipelineHelper.run_pipeline(args)
        args.manual_stop = False
    # Handle manual stop error
    except (ManualStopError, TimeoutError) as e:
        TerminalIO.print(f"Manual stop error: {e}", color=TerminalIO.FAIL)
        if args is not None:
            args.manual_stop = True
    # Handle other exceptions
    except Exception as e:
        # Store the exception to re-raise after cleanup
        caught_exception = e
    # Always log args (even there is an exception beyond ManualStopError)
    finally:
        if args is not None:
            # only log for the main process when using distributed training
            # args.wandb_logger.experiment.config.update(args)
            args.wandb_logger.log_hyperparams(args)

        if caught_exception:
            # Re-raise the exception after cleanup
            raise caught_exception

        # When running in multi-process mode, wandb will not stop automatically until all processes are done -> all runs will share the same wandb run.
        # So, we need to finish the wandb logging manually for each run.
        wandb.finish()

    return metric_dict


if __name__ == "__main__":
    run_experiment()
