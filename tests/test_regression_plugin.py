"""Integration tests for regression plugin using real datasets."""

from pathlib import Path

import pytest

from a4s_plugin_performance.regression.plugin import RegressionPerformancePlugin
from a4s_plugin_performance.utils import Feature, FeatureType


DATASETS_DIR = Path(__file__).parent.parent / "datasets" / "regression"


@pytest.fixture
def regression_plugin():
    """Create a regression plugin instance with mocked logger."""
    plugin = RegressionPerformancePlugin()
    return plugin


@pytest.fixture
def regression_data():
    """Load regression test dataset."""
    test_data = (DATASETS_DIR / "testing_data.csv").read_bytes()
    model_data = (DATASETS_DIR / "rf_reg_model.onnx").read_bytes()
    return test_data, model_data


class TestRegressionPerformancePlugin:
    """Integration tests for RegressionPerformancePlugin."""

    def test_plugin_name(self, regression_plugin):
        assert regression_plugin.plugin_name == "Regression Performance"

    def test_display_icon(self, regression_plugin):
        assert regression_plugin.display_icon == "trending_up"

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

        assert (
            regression_plugin._input_provider_instances.get("test-dataset") is not None
        )
        assert regression_plugin._input_provider_instances.get("model") is not None
        assert isinstance(
            regression_plugin.get_input_data("test-dataset"), pd.DataFrame
        )
        assert isinstance(regression_plugin.get_input_data("model"), InferenceSession)

    def test_parse_config_from_dataset(self, regression_plugin, regression_data):
        test_data, _ = regression_data
        config = regression_plugin.parse_config_from_dataset(test_data)

        assert config is not None
        assert "features" in config
        assert len(config["features"]) > 0
        # Check that features have expected structure
        for feature in config["features"]:
            assert "name" in feature
            assert "type" in feature
            assert "min" in feature
            assert "max" in feature

    def test_evaluate_without_windowing(self, regression_plugin, regression_data):
        test_data, model_data = regression_data

        regression_plugin.set_input_content("test-dataset", test_data)
        regression_plugin.set_input_content("model", model_data)

        # Create config matching the dataset structure
        features = [
            Feature(name="feat1", min=0.0, max=100.0, type=FeatureType.FLOAT),
            Feature(name="feat2", min=0.0, max=10.0, type=FeatureType.FLOAT),
            Feature(name="feat3", min=0.0, max=100.0, type=FeatureType.FLOAT),
            Feature(name="feat4", min=0.0, max=1.0, type=FeatureType.INTEGER),
            Feature(name="target", min=0.0, max=100.0, type=FeatureType.FLOAT),
        ]

        config_data = {
            "frequency": "",
            "window_size": "",
            "features": [f.model_dump() for f in features],
            "date_feature": None,
            "target_feature": "target",
        }

        results = regression_plugin.evaluate(config_data)

        assert isinstance(results, dict)
        # Check all expected metrics are present
        for metric in regression_plugin.metric_names():
            assert metric in results
            assert "score" in results[metric][0]
            assert len(results[metric]) == 1  # Single window

    def test_evaluate_metrics_valid_values(self, regression_plugin, regression_data):
        test_data, model_data = regression_data

        regression_plugin.set_input_content("test-dataset", test_data)
        regression_plugin.set_input_content("model", model_data)

        features = [
            Feature(name="feat1", min=0.0, max=100.0, type=FeatureType.FLOAT),
            Feature(name="feat2", min=0.0, max=10.0, type=FeatureType.FLOAT),
            Feature(name="feat3", min=0.0, max=100.0, type=FeatureType.FLOAT),
            Feature(name="feat4", min=0.0, max=1.0, type=FeatureType.INTEGER),
            Feature(name="target", min=0.0, max=100.0, type=FeatureType.FLOAT),
        ]

        config_data = {
            "frequency": "",
            "window_size": "",
            "features": [f.model_dump() for f in features],
            "date_feature": None,
            "target_feature": "target",
        }

        results = regression_plugin.evaluate(config_data)

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

    def test_get_metric_visualizations(self, regression_plugin, regression_data):
        test_data, model_data = regression_data

        regression_plugin.set_input_content("test-dataset", test_data)
        regression_plugin.set_input_content("model", model_data)

        # Create config matching the dataset structure (target must be in features)
        features = [
            Feature(name="feat1", min=0.0, max=100.0, type=FeatureType.FLOAT),
            Feature(name="feat2", min=0.0, max=10.0, type=FeatureType.FLOAT),
            Feature(name="feat3", min=0.0, max=100.0, type=FeatureType.FLOAT),
            Feature(name="feat4", min=0.0, max=1.0, type=FeatureType.INTEGER),
            Feature(name="target", min=0.0, max=100.0, type=FeatureType.FLOAT),
        ]

        config_data = {
            "frequency": "",
            "window_size": "",
            "features": [f.model_dump() for f in features],
            "date_feature": None,
            "target_feature": "target",
        }

        visualizations = regression_plugin.get_metric_visualizations(config_data)

        assert len(visualizations) == 2
        # First should be a table
        assert visualizations[0].chart_type.value == "table"
