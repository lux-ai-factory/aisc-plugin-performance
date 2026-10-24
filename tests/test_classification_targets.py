"""The classification plugin assesses a local model file or an AI card target (a scoring service
reached through its connection) and measures both the same way."""

import json
from pathlib import Path

import pytest

from predictive_insights import ClassificationPerformancePlugin

from .scoring_fixtures import config, measures, scoring_dataset_csv, scoring_model_bytes

GOLDEN = Path(__file__).parent / "fixtures" / "classification_measures.json"


def plugin_with(model: bytes | None = None) -> ClassificationPerformancePlugin:
    plugin = ClassificationPerformancePlugin()
    plugin._set_artifact_callback(lambda name, content: None)
    plugin.set_input_content("test-dataset", scoring_dataset_csv())
    if model is not None:
        plugin.set_input_content("model", model)
    return plugin


def test_a_local_model_file_is_measured_as_before():
    results = plugin_with(model=scoring_model_bytes()).evaluate(config())
    assert measures(results) == json.loads(GOLDEN.read_text())


def test_it_assesses_an_ai_card_target_or_a_local_model_file():
    from aisc_plugin_interface import card_needs_connection, target_kinds_of

    assert target_kinds_of(ClassificationPerformancePlugin) == ("card", "local_model")
    assert card_needs_connection(ClassificationPerformancePlugin)
