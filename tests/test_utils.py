"""Tests for utility functions."""

from predictive_insights.utils import group_metrics


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
