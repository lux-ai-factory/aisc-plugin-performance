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


PID = "0f7c1e2a-aaaa-bbbb-cccc-1234567890ab"
KEY = "component:0b9c7a1e-0000-4000-8000-000000000002"


def scorer_answer(answer_of):
    """A scoring service's answer to a row: ``answer_of(p)`` with p the probability of class 1 that
    the fixed model gives the row."""
    import math

    from .scoring_fixtures import BIAS, FEATURES, WEIGHTS

    def answer(request):
        row = json.loads(request["body"])
        z = sum(w * row[f] for w, f in zip(WEIGHTS, FEATURES)) + BIAS
        return answer_of(1 / (1 + math.exp(-z)))

    return answer


@pytest.fixture
def card_target(scoring_service, monkeypatch):
    """An AI card target whose connection reaches the scoring service at POST /score."""
    connection = {
        "name": "scoring", "label": "Scoring service", "kind": "rest", "method": "POST", "path": "/score",
        "headers": {}, "secret_header": None, "body_template": "{{input}}", "response_path": "$",
        "refusal": None, "model": None, "timeout_s": 5, "secret": None, "updated_at": "2026-10-04T10:00:00Z",
        "target": {"key": KEY, "kind": "component", "label": "Scoring service"},
        "base_url": scoring_service.base, "allowed_hosts": [scoring_service.host], "denied_addresses": [],
    }
    scoring_service.route(f"/internal/projects/{PID}/targets/{KEY}/connection", 200, connection)
    monkeypatch.setenv("PLATFORM_URL", scoring_service.base)
    monkeypatch.setenv("PLATFORM_CONNECTIONS_TOKEN", "service-token")
    return scoring_service


def plugin_on_card_target():
    plugin = plugin_with()
    plugin.set_input_content("target", json.dumps({"value": f"target:{PID}/{KEY}"}).encode())
    return plugin


def test_an_ai_card_target_is_measured_like_the_local_model(card_target):
    card_target.route("/score", 200, scorer_answer(lambda p: {"label": int(p > 0.5), "probability": p}))
    results = plugin_on_card_target().evaluate(config())

    golden = json.loads(GOLDEN.read_text())
    got = measures(results)
    assert {name: len(values) for name, values in got.items()} == {name: len(values) for name, values in golden.items()}
    for name, values in golden.items():
        assert [m["score"] for m in got[name]] == pytest.approx([m["score"] for m in values], abs=1e-6), name
    assert len([r for r in card_target.seen if r["path"] == "/score"]) == 24       # each row sent once


def test_the_config_says_where_a_scoring_service_answers_the_probability(card_target):
    card_target.route("/score", 200, scorer_answer(
        lambda p: {"decision": "decline" if p > 0.5 else "approve", "p_default": p}))
    results = plugin_on_card_target().evaluate(dict(config(), target_probability_field="p_default"))

    golden = json.loads(GOLDEN.read_text())
    assert measures(results).keys() == golden.keys()
    assert results["Accuracy"][0]["score"] == pytest.approx(golden["Accuracy"][0]["score"])


def test_an_answer_without_the_probability_fails_naming_its_path(card_target):
    from aisc_plugin_interface.connections import EndpointBadResponse

    card_target.route("/score", 200, {"decision": "approve"})
    with pytest.raises(EndpointBadResponse, match=r"row 1.*'p_default'"):
        plugin_on_card_target().evaluate(dict(config(), target_probability_field="p_default"))
