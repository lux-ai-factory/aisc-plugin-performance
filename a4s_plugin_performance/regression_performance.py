# src/my_a4s_plugin/plugin.py
from a4s_plugin_interface.models.measure import Measure
from a4s_plugin_interface import metric
from a4s_plugin_interface.base_evaluation_plugin import BaseEvaluationPlugin
from pydantic import BaseModel, Field


class ConfigForSchema(BaseModel):
    frequency: str = Field(
        default="1D",
        description="Frequency threshold (days only, e.g. '1D', '7D')",
        pattern=r"^[1-9]\d*D$"
    )
    window_size: str = Field(
        default="7 days",
        description="Window size for sliding window (days only, e.g. '7 days', '30 days')",
        pattern=r"^[1-9]\d* days$"
    )

class RegressionPerformancePlugin(BaseEvaluationPlugin[ConfigForSchema]):

    def evaluate(self, dataset_pid) -> list[Measure]:
        
        return {
            "accuracy": 0.95,
            "F1": 0.92
        }

    @metric("accuracy")
    def metric_accuracy(self, values: dict) -> list[Measure]:
        accuracy = values.get("accuracy", 0.0)
        return [Measure(name="accuracy", score=accuracy)]