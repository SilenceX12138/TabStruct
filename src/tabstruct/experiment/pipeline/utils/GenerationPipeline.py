from src.tabstruct.generation.models.GeneratorHelper import GeneratorHelper

from ..BasePipeline import BasePipeline


class GenerationPipeline(BasePipeline):
    """Experiment class for prediction tasks."""

    model_helper = GeneratorHelper

    @classmethod
    def _run(cls, args):
        metric_dict = cls.model_helper.benchmark_model(args)

        return metric_dict
