"""Integration tests for regression plugin using real datasets."""

import json
from pathlib import Path

import pytest

from predictive_insights import RegressionPerformancePlugin


DATASETS_DIR = Path(__file__).parent.parent / "datasets" / "regression"

FEATURE_COLUMNS = ["feat1", "feat2", "feat3", "feat4"]


@pytest.fixture
def regression_plugin():
    """Create a regression plugin instance."""
    return RegressionPerformancePlugin()


@pytest.fixture
def regression_data():
    """Load regression test dataset."""
    test_data = (DATASETS_DIR / "testing_data.csv").read_bytes()
    model_data = (DATASETS_DIR / "rf_reg_model.onnx").read_bytes()
    return test_data, model_data


@pytest.fixture
def regression_inputs(regression_plugin, regression_data, datashape_payload):
    """Plugin with test-dataset, model and datashape inputs attached."""
    test_data, model_data = regression_data
    regression_plugin.set_input_content("test-dataset", test_data)
    regression_plugin.set_input_content("model", model_data)
    payload = datashape_payload([*FEATURE_COLUMNS, "target"], target="target")
    regression_plugin.set_input_content("datashape", json.dumps(payload).encode())
    return regression_plugin


class TestRegressionPerformancePlugin:
    """Integration tests for RegressionPerformancePlugin."""

    def test_plugin_name(self, regression_plugin):
        assert regression_plugin.plugin_name == "Regression Performance"

    def test_display_icon(self, regression_plugin):
        assert regression_plugin.display_icon == "trending_up"

    def test_description(self, regression_plugin):
        description = regression_plugin.description
        assert description.startswith("Evaluates a trained ONNX regression model")
        assert "**Configuration:**" in description
        assert "- **Frequency**" in description
        assert "- **Window Size**" in description

    def test_metric_names(self, regression_plugin):
        metric_names = regression_plugin.metric_names()
        expected = [
            "Mean Absolute Error",
            "Mean Squared Error",
            "Root Mean Squared Error",
            "Explained Variance",
            "R2",
        ]
        assert metric_names == expected

    def test_set_input_providers(self, regression_plugin, regression_data):
        import pandas as pd
        from onnxruntime import InferenceSession

        test_data, model_data = regression_data

        regression_plugin.set_input_content("test-dataset", test_data)
        regression_plugin.set_input_content("model", model_data)

        assert regression_plugin._input_provider_instances.get("test-dataset") is not None
        assert regression_plugin._input_provider_instances.get("model") is not None
        assert isinstance(regression_plugin.get_input_data("test-dataset"), pd.DataFrame)
        assert isinstance(regression_plugin.get_input_data("model"), InferenceSession)

    def test_evaluate_missing_datashape_raises(self, regression_plugin, regression_data):
        test_data, model_data = regression_data
        regression_plugin.set_input_content("test-dataset", test_data)
        regression_plugin.set_input_content("model", model_data)

        with pytest.raises(ValueError, match="Data Shape input is required"):
            regression_plugin.evaluate({"frequency": "", "window_size": ""})

    def test_evaluate_without_windowing(self, regression_inputs):
        results = regression_inputs.evaluate({"frequency": "", "window_size": ""})

        assert isinstance(results, dict)
        # Check all expected metrics are present
        for metric in regression_inputs.metric_names():
            assert metric in results
            assert "score" in results[metric][0]
            assert len(results[metric]) == 1  # Single window

    def test_evaluate_metrics_valid_values(self, regression_inputs):
        results = regression_inputs.evaluate({"frequency": "", "window_size": ""})

        # MAE, MSE, RMSE should be non-negative
        for metric in [
            "Mean Absolute Error",
            "Mean Squared Error",
            "Root Mean Squared Error",
        ]:
            score = results[metric][0]["score"]
            assert score >= 0.0, f"{metric} is negative: {score}"

        # Explained Variance and R2 can be negative for bad models, but typically <= 1
        for metric in ["Explained Variance", "R2"]:
            score = results[metric][0]["score"]
            assert score <= 1.0, f"{metric} > 1: {score}"
            assert isinstance(score, float)

    def test_get_metric_visualizations(self, regression_inputs):
        visualizations = regression_inputs.get_metric_visualizations(
            {"frequency": "", "window_size": ""}
        )

        assert len(visualizations) == 2
        # First should be a table
        assert visualizations[0].chart_type.value == "table"
