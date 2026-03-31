"""Integration tests for classification plugin using real datasets."""

from pathlib import Path

import pytest

from a4s_plugin_performance import ClassificationPerformancePlugin
from a4s_plugin_performance.utils import Feature, FeatureType


DATASETS_DIR = Path(__file__).parent.parent / "datasets" / "classification"


@pytest.fixture
def classification_plugin():
    """Create a classification plugin instance."""
    plugin = ClassificationPerformancePlugin()
    return plugin


@pytest.fixture
def classification_data():
    """Load classification test dataset."""
    test_data = (DATASETS_DIR / "testing_data.csv").read_bytes()
    model_data = (DATASETS_DIR / "rf_cls_model.onnx").read_bytes()
    return test_data, model_data


class TestClassificationPerformancePlugin:
    """Integration tests for ClassificationPerformancePlugin."""

    def test_plugin_name(self, classification_plugin):
        assert classification_plugin.plugin_name == "Classification Performance"

    def test_display_icon(self, classification_plugin):
        assert classification_plugin.display_icon == "category"

    def test_metric_names(self, classification_plugin):
        metric_names = classification_plugin.metric_names()
        expected = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "Confusion-Matrix",
            "MCC",
            "SCE",
            "ECE",
            "MCE",
        ]
        assert metric_names == expected

    def test_set_input_providers(self, classification_plugin, classification_data):
        import pandas as pd
        from onnxruntime import InferenceSession

        test_data, model_data = classification_data

        classification_plugin.set_input_content("test-dataset", test_data)
        classification_plugin.set_input_content("model", model_data)

        assert (
            classification_plugin._input_provider_instances.get("test-dataset")
            is not None
        )
        assert classification_plugin._input_provider_instances.get("model") is not None
        assert isinstance(
            classification_plugin.get_input_data("test-dataset"), pd.DataFrame
        )
        assert isinstance(
            classification_plugin.get_input_data("model"), InferenceSession
        )

    def test_parse_config_from_dataset(
        self, classification_plugin, classification_data
    ):
        test_data, _ = classification_data
        config = classification_plugin.parse_config_from_dataset(test_data)

        assert config is not None
        assert "features" in config
        assert len(config["features"]) > 0
        # Check that features have expected structure
        for feature in config["features"]:
            assert "name" in feature
            assert "type" in feature
            assert "min" in feature
            assert "max" in feature

    def test_evaluate_without_windowing(
        self, classification_plugin, classification_data
    ):
        test_data, model_data = classification_data

        classification_plugin.set_input_content("test-dataset", test_data)
        classification_plugin.set_input_content("model", model_data)

        # Create config matching the dataset structure
        features = [
            Feature(name="feat1", min=0.0, max=100.0, type=FeatureType.FLOAT),
            Feature(name="feat2", min=0.0, max=10.0, type=FeatureType.FLOAT),
            Feature(name="feat3", min=0.0, max=100.0, type=FeatureType.FLOAT),
            Feature(name="feat4", min=0.0, max=1.0, type=FeatureType.INTEGER),
            Feature(name="target", min=0.0, max=1.0, type=FeatureType.INTEGER),
        ]

        config_data = {
            "frequency": "",
            "window_size": "",
            "features": [f.model_dump() for f in features],
            "date_feature": None,
            "target_feature": "target",
        }

        results = classification_plugin.evaluate(config_data)

        assert isinstance(results, dict)
        # Check all expected metrics are present
        for metric in ["Accuracy", "Precision", "Recall", "F1-Score", "MCC"]:
            assert metric in results
            assert "score" in results[metric][0]
            assert len(results[metric]) == 1  # Single window

        # Check calibration metrics
        for metric in ["SCE", "ECE", "MCE"]:
            assert metric in results
            assert "score" in results[metric][0]

    def test_evaluate_metrics_valid_ranges(
        self, classification_plugin, classification_data
    ):
        test_data, model_data = classification_data

        classification_plugin.set_input_content("test-dataset", test_data)
        classification_plugin.set_input_content("model", model_data)

        features = [
            Feature(name="feat1", min=0.0, max=100.0, type=FeatureType.FLOAT),
            Feature(name="feat2", min=0.0, max=10.0, type=FeatureType.FLOAT),
            Feature(name="feat3", min=0.0, max=100.0, type=FeatureType.FLOAT),
            Feature(name="feat4", min=0.0, max=1.0, type=FeatureType.INTEGER),
            Feature(name="target", min=0.0, max=1.0, type=FeatureType.INTEGER),
        ]

        config_data = {
            "frequency": "",
            "window_size": "",
            "features": [f.model_dump() for f in features],
            "date_feature": None,
            "target_feature": "target",
        }

        results = classification_plugin.evaluate(config_data)

        # Accuracy, Precision, Recall, F1 should be in [0, 1]
        for metric in ["Accuracy", "Precision", "Recall", "F1-Score"]:
            score = results[metric][0]["score"]
            assert 0.0 <= score <= 1.0, f"{metric} out of range: {score}"

        # MCC is normalized to [0, 1] in the plugin
        mcc = results["MCC"][0]["score"]
        assert 0.0 <= mcc <= 1.0, f"MCC out of range: {mcc}"

        # Calibration metrics should be non-negative
        for metric in ["SCE", "ECE", "MCE"]:
            score = results[metric][0]["score"]
            assert score >= 0.0, f"{metric} is negative: {score}"

    def test_get_metric_visualizations(
        self, classification_plugin, classification_data
    ):
        test_data, model_data = classification_data

        classification_plugin.set_input_content("test-dataset", test_data)
        classification_plugin.set_input_content("model", model_data)

        # Create config matching the dataset structure (target must be in features)
        features = [
            Feature(name="feat1", min=0.0, max=100.0, type=FeatureType.FLOAT),
            Feature(name="feat2", min=0.0, max=10.0, type=FeatureType.FLOAT),
            Feature(name="feat3", min=0.0, max=100.0, type=FeatureType.FLOAT),
            Feature(name="feat4", min=0.0, max=1.0, type=FeatureType.INTEGER),
            Feature(name="target", min=0.0, max=1.0, type=FeatureType.INTEGER),
        ]

        config_data = {
            "frequency": "",
            "window_size": "",
            "features": [f.model_dump() for f in features],
            "date_feature": None,
            "target_feature": "target",
        }

        visualizations = classification_plugin.get_metric_visualizations(config_data)

        assert len(visualizations) > 0
        # First should be a table
        assert visualizations[0].chart_type.value == "table"
