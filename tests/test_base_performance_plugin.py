"""Tests for BasePerformanceEvaluationPlugin (inputs + datashape resolution)."""

import json

import pytest

from predictive_insights import ClassificationPerformancePlugin
from predictive_insights.config_form import ConfigForm


@pytest.fixture
def plugin():
    return ClassificationPerformancePlugin()


class TestEvaluationInputs:
    """Tests for the evaluation input slots and config type."""

    def test_input_definitions(self, plugin):
        names = {definition.name for definition in plugin.input_definitions}
        assert names == {"test-dataset", "model", "datashape"}

    def test_input_types(self, plugin):
        from aisc_plugin_interface import InputType

        by_name = {d.name: d for d in plugin.input_definitions}
        assert by_name["test-dataset"].input_type == InputType.DATASET
        assert by_name["model"].input_type == InputType.MODEL
        assert by_name["datashape"].input_type == InputType.DATASHAPE

    def test_datashape_uses_json_provider(self, plugin):
        """JSON-backed inputs default to JsonInputProvider (no provider needed)."""
        from aisc_plugin_interface import JsonInputProvider

        assert plugin._input_provider_types["datashape"] is JsonInputProvider

    def test_config_type(self, plugin):
        assert plugin.config_type is ConfigForm

    def test_config_schema_has_no_data_shape_fields(self, plugin):
        props = plugin.get_config_form_schema().get("properties", {})
        for removed in ("features", "target_feature", "date_feature"):
            assert removed not in props


class TestReadDatashape:
    """Tests for the shared _read_datashape resolution helper."""

    def test_missing_datashape_raises(self, plugin):
        with pytest.raises(ValueError, match="Data Shape input is required"):
            plugin._read_datashape()

    def test_missing_target_raises(self, plugin, datashape_payload):
        payload = datashape_payload(["a", "b"], target=None)
        plugin.set_input_content("datashape", json.dumps(payload).encode())
        with pytest.raises(ValueError, match="must designate a target"):
            plugin._read_datashape()

    def test_roles_resolved(self, plugin, datashape_payload):
        payload = datashape_payload(
            ["feat1", "feat2", "date", "target"], target="target", date="date"
        )
        plugin.set_input_content("datashape", json.dumps(payload).encode())
        target, date, features = plugin._read_datashape()
        assert target == "target"
        assert date == "date"
        assert features == ["feat1", "feat2"]

    def test_no_date_returns_none(self, plugin, datashape_payload):
        payload = datashape_payload(["feat1", "target"], target="target")
        plugin.set_input_content("datashape", json.dumps(payload).encode())
        target, date, features = plugin._read_datashape()
        assert target == "target"
        assert date is None
        assert features == ["feat1"]
