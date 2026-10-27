"""Tests for the plugin ConfigForm."""

from predictive_insights.config_form import ConfigForm


class TestConfigForm:
    """Tests for the ConfigForm Pydantic model."""

    def test_default_values(self):
        config = ConfigForm()
        assert config.frequency == ""
        assert config.window_size == ""

    def test_model_dump(self):
        config = ConfigForm(frequency="1D", window_size="7D")
        dump = config.model_dump()
        assert dump["frequency"] == "1D"
        assert dump["window_size"] == "7D"

    def test_no_data_shape_fields(self):
        """Data-shape fields must not live in the plugin config (migration)."""
        props = ConfigForm.model_json_schema().get("properties", {})
        for removed in ("features", "target_feature", "date_feature"):
            assert removed not in props

    def test_schema_uses_string_text_inputs(self):
        schema = ConfigForm.model_json_schema()
        assert schema["properties"]["frequency"]["type"] == "string"
        assert schema["properties"]["window_size"]["type"] == "string"
