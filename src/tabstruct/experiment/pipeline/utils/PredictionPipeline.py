from src.tabstruct.prediction.models.PredictorHelper import PredictorHelper

from ..BasePipeline import BasePipeline


class PredictionPipeline(BasePipeline):
    """Experiment class for prediction tasks."""

    model_helper = PredictorHelper

    @classmethod
    def _run(cls, args):
        metric_dict = cls.model_helper.benchmark_model(args)

        return metric_dict
