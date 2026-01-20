# src/my_a4s_plugin/plugin.py
from a4s_plugin_interface.models.measure import Measure
from a4s_plugin_interface import metric
from a4s_plugin_interface.base_evaluation_plugin import BaseEvaluationPlugin
from pydantic import BaseModel, Field


METRIC_MAE = "Mean Absolute Error"
METRIC_MSE = "Mean Squared Error"
METRIC_R2 = "R2 Score"

class ConfigForSchema(BaseModel):
    pass

class RegressionPerformancePlugin(BaseEvaluationPlugin[ConfigForSchema]):

    def evaluate(self, dataset_pid) -> list[Measure]:
        # TODO: to be implemented
        return 
    

    @metric(METRIC_MAE)
    def metric_mae(self, result: dict) -> list[Measure]:
        # TODO: to be implemented
        return []