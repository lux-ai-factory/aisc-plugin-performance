"""Tests for the generated per-dataset data shapes (tests/files/<case>/).

The datashapes are derived deterministically from each TRAIN dataset by
``tests/generate_files.py`` in the real platform format — a features-only
payload (``{"features": [...]}``) — and these tests assert that structure.
"""

import json
from typing import Any, cast

import pytest

from tests import generate_files

CASES = sorted(generate_files.CASES)

ALLOWED_SEMANTIC = ("numeric", "categorical", "datetime", "text", "boolean")
ALLOWED_ROLES = ("feature", "target", "date", "ignore")


def load_features(case: str) -> list[dict[str, Any]]:
    """Load the features list of a generated datashape payload."""
    payload = json.loads((generate_files.FILES / case / "datashape.json").read_text())
    features = payload.get("features")
    assert isinstance(features, list) and features
    return [f for f in features if isinstance(f, dict)]


def test_datashape_is_features_only() -> None:
    """The platform emits a features-only payload (no metadata wrapper)."""
    payload = json.loads((generate_files.FILES / "classification" / "datashape.json").read_text())
    assert set(payload.keys()) == {"features"}


@pytest.mark.parametrize("case", CASES)
def test_features_non_empty(case: str) -> None:
    assert len(load_features(case)) >= 1


@pytest.mark.parametrize("case", CASES)
def test_exactly_one_target_feature(case: str) -> None:
    features = load_features(case)
    targets = [f["name"] for f in features if f.get("role") == "target"]
    assert targets == [generate_files.CASES[case]["target"]]


@pytest.mark.parametrize("case", CASES)
def test_feature_metadata_is_complete(case: str) -> None:
    for feature in load_features(case):
        assert feature.get("name")
        assert feature.get("dtype")
        assert feature.get("semantic_type") in ALLOWED_SEMANTIC
        assert feature.get("role") in ALLOWED_ROLES
        assert feature.get("null_count", -1) >= 0
        assert feature.get("unique_count", -1) >= 0


@pytest.mark.parametrize("case", CASES)
def test_numeric_features_have_stats(case: str) -> None:
    for feature in load_features(case):
        if feature.get("semantic_type") == "numeric" and feature.get("null_count") == 0:
            assert feature.get("min") is not None
            assert feature.get("max") is not None
            assert feature.get("mean") is not None
            assert feature.get("std") is not None


def test_time_series_default_format() -> None:
    """Mirrors the default datashape: target designated, date NOT auto-assigned."""
    features = load_features("time_series")
    assert [f["name"] for f in features if f.get("role") == "target"] == ["charged_off"]

    issue_d = next(f for f in features if f["name"] == "issue_d")
    assert issue_d["semantic_type"] == "datetime"
    assert issue_d["role"] == "feature"


def test_datashape_covers_every_train_column() -> None:
    """The datashape must describe every column of its train dataset."""
    import pandas as pd

    for case in ("classification", "regression"):
        df = pd.read_csv(f"datasets/{case}/training_data.csv")
        features = load_features(case)
        assert {f["name"] for f in features} == set(df.columns)


def test_config_json_is_schema_valid() -> None:
    """Every case's config.json must validate against the real ConfigForm."""
    from predictive_insights.config_form import ConfigForm

    for case in CASES:
        config: dict[str, Any] = json.loads(
            (generate_files.FILES / case / "config.json").read_text()
        )
        model = ConfigForm.model_validate(config)
        expected = cast("dict[str, Any]", generate_files.CASES[case]["config"])
        assert model.frequency == expected["frequency"]
        assert model.window_size == expected["window_size"]
