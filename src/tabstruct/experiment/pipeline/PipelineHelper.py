from wrapt_timeout_decorator import timeout

from src.tabstruct.common import SINGLE_RUN_TIMEOUT
from src.tabstruct.common.runtime.log.TerminalIO import TerminalIO

from .utils.GenerationPipeline import GenerationPipeline
from .utils.PredictionPipeline import PredictionPipeline


class PipelineHelper:

    @classmethod
    @timeout(SINGLE_RUN_TIMEOUT)
    @TerminalIO.trace_func
    def run_pipeline(cls, args):
        # === Select the pipeline ===
        pipeline_class = cls.pipeline_handler(args.pipeline)

        # === Run the experiment ===
        metric_dict = pipeline_class.run(args)

        return metric_dict

    @classmethod
    def pipeline_handler(cls, pipeline: str):
        match pipeline:
            case "generation":
                return GenerationPipeline
            case "prediction":
                return PredictionPipeline
            case _:
                raise ValueError(f"Invalid pipeline: {pipeline}")
