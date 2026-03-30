"""Tests for utility functions."""

from a4s_plugin_performance.utils import (
    Feature,
    FeatureType,
    group_metrics,
)


class TestFeatureType:
    """Tests for FeatureType enum."""

    def test_feature_type_values(self):
        assert FeatureType.INTEGER.value == "Integer"
        assert FeatureType.FLOAT.value == "Float"
        assert FeatureType.CATEGORICAL.value == "Categorical"
        assert FeatureType.DATE.value == "Date"

    def test_feature_type_is_string_enum(self):
        assert isinstance(FeatureType.INTEGER, str)
        assert FeatureType.INTEGER == "Integer"


class TestFeature:
    """Tests for Feature model."""

    def test_feature_creation(self):
        feature = Feature(name="test_col", min=0.0, max=100.0, type=FeatureType.FLOAT)
        assert feature.name == "test_col"
        assert feature.min == 0.0
        assert feature.max == 100.0
        assert feature.type == FeatureType.FLOAT

    def test_feature_model_dump(self):
        feature = Feature(name="age", min=18.0, max=65.0, type=FeatureType.INTEGER)
        dump = feature.model_dump()
        assert dump["name"] == "age"
        assert dump["min"] == 18.0
        assert dump["max"] == 65.0
        assert dump["type"] == FeatureType.INTEGER
        assert "pid" in dump


class TestGroupMetrics:
    """Tests for group_metrics function."""

    def test_group_single_dict(self):
        dicts = [{"a": {"x": 1, "y": 2}}]
        result = group_metrics(dicts)
        assert result == {"a": [{"x": 1, "y": 2}]}

    def test_group_multiple_dicts(self):
        dicts = [{"a": {"x": 1}}, {"a": {"x": 2, "y": 3}, "b": {"z": 4}}]
        result = group_metrics(dicts)
        assert result == {"a": [{"x": 1}, {"x": 2, "y": 3}], "b": [{"z": 4}]}

    def test_group_empty_list(self):
        result = group_metrics([])
        assert result == {}

    def test_group_with_different_keys(self):
        dicts = [
            {"metric1": {"score": 0.9, "time": "2024-01-01"}},
            {"metric1": {"score": 0.85, "time": "2024-01-02"}},
            {"metric2": {"score": 0.7, "time": "2024-01-01"}},
        ]
        result = group_metrics(dicts)
        assert isinstance(result["metric1"], list)
        assert len(result["metric1"]) == 2
        assert result["metric1"][0].get("score") == 0.9
        assert result["metric1"][1].get("score") == 0.85

        assert isinstance(result["metric2"], list)
        assert len(result["metric2"]) == 1
        assert result["metric2"][0].get("score") == 0.7
