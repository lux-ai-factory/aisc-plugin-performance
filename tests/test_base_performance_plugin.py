"""Tests for BasePerformanceEvaluationPlugin (on_config_change)."""

import pytest

from predictive_insights import ClassificationPerformancePlugin
from predictive_insights.utils import Feature, FeatureType
from predictive_insights.config_form import ConfigForm


@pytest.fixture
def plugin():
    return ClassificationPerformancePlugin()


@pytest.fixture
def base_form():
    return ConfigForm(
        target_feature=None,
        date_feature=None,
        frequency="",
        window_size="",
        features=[],
    )


def make_features(feature_list: list[tuple[str, FeatureType]]) -> list[Feature]:
    return [
        Feature(name=name, min=0.0, max=1.0, type=ftype) for name, ftype in feature_list
    ]


class TestOnConfigChange:
    """Tests for on_config_change schema and form data handling."""

    def test_returns_schema_and_ui_schema(self, plugin, base_form):
        """Should return a tuple of (form_data, schema, ui_schema)."""
        result = plugin.on_config_change(base_form)
        assert len(result) == 3
        form_data, schema, ui_schema = result
        assert isinstance(schema, dict)
        assert isinstance(ui_schema, dict)

    def test_none_form_data_hides_all_fields(self, plugin):
        """When form_data is None, all fields should be hidden."""
        _, schema, ui_schema = plugin.on_config_change(None)
        assert ui_schema.get("date_feature") == {"ui:widget": "hidden"}
        assert ui_schema.get("target_feature") == {"ui:widget": "hidden"}
        assert ui_schema.get("frequency") == {"ui:widget": "hidden"}
        assert ui_schema.get("window_size") == {"ui:widget": "hidden"}

    def test_no_features_hides_date_and_target(self, plugin, base_form):
        """With no features, date/target fields should be hidden."""
        _, schema, ui_schema = plugin.on_config_change(base_form)
        assert ui_schema.get("date_feature") == {"ui:widget": "hidden"}
        assert ui_schema.get("target_feature") == {"ui:widget": "hidden"}

    def test_no_date_feature_candidates_hides_date_field(self, plugin):
        """When no features qualify as date features, date field should be hidden."""
        form = ConfigForm(
            target_feature="target",
            date_feature=None,
            features=make_features(
                [
                    ("target", FeatureType.INTEGER),
                    ("feat1", FeatureType.FLOAT),
                ]
            ),
        )
        _, schema, ui_schema = plugin.on_config_change(form)
        assert ui_schema.get("date_feature") == {"ui:widget": "hidden"}
        assert ui_schema.get("frequency") == {"ui:widget": "hidden"}
        assert ui_schema.get("window_size") == {"ui:widget": "hidden"}

    def test_date_feature_schema_is_plain_string_with_enum(self, plugin):
        """Date feature schema should be type string with enum, not anyOf."""
        form = ConfigForm(
            target_feature="target",
            date_feature=None,
            features=make_features(
                [
                    ("target", FeatureType.INTEGER),
                    ("issue_d", FeatureType.DATE),
                ]
            ),
        )
        _, schema, _ = plugin.on_config_change(form)
        date_schema = schema["properties"]["date_feature"]
        assert "anyOf" not in date_schema
        assert date_schema["type"] == "string"
        assert date_schema["enum"] == ["", "issue_d"]
        assert date_schema["default"] == ""

    def test_date_feature_default_is_empty_string(self, plugin):
        """Default should be '' (no selection)."""
        form = ConfigForm(
            target_feature="target",
            date_feature=None,
            features=make_features(
                [
                    ("target", FeatureType.INTEGER),
                    ("timestamp", FeatureType.DATE),
                    ("created_at", FeatureType.DATE),
                ]
            ),
        )
        _, schema, _ = plugin.on_config_change(form)
        date_schema = schema["properties"]["date_feature"]
        assert date_schema["default"] == ""

    def test_target_feature_schema_is_plain_string_with_enum(self, plugin):
        """Target feature schema should be type string with enum, not anyOf."""
        form = ConfigForm(
            target_feature=None,
            date_feature=None,
            features=make_features(
                [
                    ("feat1", FeatureType.FLOAT),
                    ("target", FeatureType.INTEGER),
                ]
            ),
        )
        _, schema, _ = plugin.on_config_change(form)
        target_schema = schema["properties"]["target_feature"]
        assert "anyOf" not in target_schema
        assert target_schema["type"] == "string"
        assert "" in target_schema["enum"]
        assert "target" in target_schema["enum"]

    def test_target_feature_default_is_last_candidate(self, plugin):
        """Default for target should be the last candidate."""
        form = ConfigForm(
            target_feature=None,
            date_feature=None,
            features=make_features(
                [
                    ("feat1", FeatureType.FLOAT),
                    ("target", FeatureType.INTEGER),
                    ("label", FeatureType.INTEGER),
                ]
            ),
        )
        _, schema, _ = plugin.on_config_change(form)
        target_schema = schema["properties"]["target_feature"]
        assert target_schema["default"] == "label"

    def test_date_feature_enum_includes_categorical_features(self, plugin):
        """Categorical features with date-like data should appear in enum."""
        form = ConfigForm(
            target_feature="target",
            date_feature=None,
            features=make_features(
                [
                    ("target", FeatureType.INTEGER),
                    ("issue_d", FeatureType.DATE),
                    ("str_date", FeatureType.CATEGORICAL),
                ]
            ),
        )
        _, schema, _ = plugin.on_config_change(form)
        date_schema = schema["properties"]["date_feature"]
        assert "issue_d" in date_schema["enum"]
        assert "str_date" in date_schema["enum"]

    def test_form_data_date_feature_none_converted_to_empty_string(self, plugin):
        """When date_feature is None and options exist, form_data gets ''."""
        form = ConfigForm(
            target_feature=None,
            date_feature=None,
            features=make_features(
                [
                    ("target", FeatureType.INTEGER),
                    ("issue_d", FeatureType.DATE),
                ]
            ),
        )
        form_data, _, _ = plugin.on_config_change(form)
        assert form_data is not form  # new instance
        assert form_data.date_feature == ""

    def test_form_data_date_feature_empty_string_preserved(self, plugin):
        """When date_feature is '' (converted to None by validator), it gets '' back."""
        form = ConfigForm(
            target_feature=None,
            date_feature="",
            features=make_features(
                [
                    ("target", FeatureType.INTEGER),
                    ("issue_d", FeatureType.DATE),
                ]
            ),
        )
        # model_validator converts "" -> None, so form.date_feature is None
        assert form.date_feature is None
        form_data, _, _ = plugin.on_config_change(form)
        assert form_data.date_feature == ""

    def test_form_data_date_feature_has_value_preserved(self, plugin):
        """When date_feature is set to a valid feature, form_data unchanged."""
        form = ConfigForm(
            target_feature="target",
            date_feature="issue_d",
            features=make_features(
                [
                    ("target", FeatureType.INTEGER),
                    ("issue_d", FeatureType.DATE),
                ]
            ),
        )
        form_data, _, _ = plugin.on_config_change(form)
        assert form_data is form
        assert form_data.date_feature == "issue_d"

    def test_form_data_target_feature_none_converted_to_default(self, plugin):
        """When target_feature is None and options exist, form_data gets last candidate."""
        form = ConfigForm(
            target_feature=None,
            date_feature=None,
            features=make_features(
                [
                    ("feat1", FeatureType.FLOAT),
                    ("target", FeatureType.INTEGER),
                    ("label", FeatureType.INTEGER),
                ]
            ),
        )
        form_data, _, _ = plugin.on_config_change(form)
        assert form_data is not form
        assert form_data.target_feature == "label"

    def test_form_data_target_feature_has_value_preserved(self, plugin):
        """When target_feature is set to a valid value, form_data unchanged."""
        form = ConfigForm(
            target_feature="target",
            date_feature=None,
            features=make_features(
                [
                    ("feat1", FeatureType.FLOAT),
                    ("target", FeatureType.INTEGER),
                    ("label", FeatureType.INTEGER),
                ]
            ),
        )
        form_data, _, _ = plugin.on_config_change(form)
        assert form_data is form
        assert form_data.target_feature == "target"

    def test_handles_dict_form_data(self, plugin):
        """Should handle dict form_data (not just ConfigForm)."""
        form_dict = {
            "target_feature": None,
            "date_feature": None,
            "frequency": "",
            "window_size": "",
            "features": [
                {"name": "feat1", "min": 0.0, "max": 1.0, "type": FeatureType.FLOAT},
                {"name": "issue_d", "min": 0.0, "max": 0.0, "type": FeatureType.DATE},
                {"name": "target", "min": 0.0, "max": 1.0, "type": FeatureType.INTEGER},
            ],
        }
        form_data, _, _ = plugin.on_config_change(form_dict)
        assert isinstance(form_data, dict)
        assert form_data["date_feature"] == ""
        assert form_data["target_feature"] == "target"

    def test_schema_not_mutated_between_calls(self, plugin):
        """Each call to on_config_change should produce a fresh schema."""
        form1 = ConfigForm(
            target_feature=None,
            date_feature=None,
            features=make_features(
                [
                    ("target", FeatureType.INTEGER),
                ]
            ),
        )
        form2 = ConfigForm(
            target_feature=None,
            date_feature=None,
            features=make_features(
                [
                    ("target", FeatureType.INTEGER),
                    ("issue_d", FeatureType.DATE),
                ]
            ),
        )
        _, schema_no_date, _ = plugin.on_config_change(form1)
        _, schema_with_date, _ = plugin.on_config_change(form2)
        date_schema = schema_with_date["properties"].get("date_feature")
        assert date_schema is not None
        assert "enum" in date_schema
        empty_date_schema = schema_no_date["properties"]["date_feature"]
        assert "type" in empty_date_schema or "anyOf" in empty_date_schema

    def test_ui_schema_independent_across_calls(self, plugin):
        """UI schema should be a fresh deep copy each time."""
        form = ConfigForm(
            target_feature=None,
            date_feature=None,
            features=make_features(
                [
                    ("target", FeatureType.INTEGER),
                    ("issue_d", FeatureType.DATE),
                ]
            ),
        )
        result1 = plugin.on_config_change(form)
        result2 = plugin.on_config_change(form)
        assert result1[2] is not result2[2]
