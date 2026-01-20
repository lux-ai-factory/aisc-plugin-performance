# src/my_a4s_plugin/__init__.py
from .classif_performance import ClassificationPerformancePlugin
from .regression_performance import RegressionPerformancePlugin

__all__ = [
    "ClassificationPerformancePlugin",
    "RegressionPerformancePlugin"
]