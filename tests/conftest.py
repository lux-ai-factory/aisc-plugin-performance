"""Shared fixtures and auto-regeneration of generated test fixtures.

``tests/generate_files.py`` deterministically derives each case's inputs
(dataset/model/datashape/config) into ``tests/files/<case>/``. If the fixtures
are missing (fresh clone), regenerate them at import time so no test is skipped.
"""

import json
from typing import Any

import pytest

from tests import generate_files


def _ensure_fixtures() -> None:
    """Regenerate tests/files/<case>/ fixtures when they are missing."""
    if not generate_files.FILES.exists() or not any(generate_files.FILES.iterdir()):
        generate_files.main()


_ensure_fixtures()


def load_case(case: str) -> dict[str, Any]:
    """Load a generated case folder (config.json, datashape.json, inputs)."""
    folder = generate_files.FILES / case
    return {
        "config": json.loads((folder / "config.json").read_text()),
        "datashape": json.loads((folder / "datashape.json").read_text()),
        "dataset": (folder / "dataset.csv").read_bytes(),
        "model": (folder / "model.onnx").read_bytes(),
    }


def make_datashape_payload(
    columns: list[str],
    target: str | None,
    date: str | None = None,
) -> dict[str, Any]:
    """Build a features-only datashape payload for the given columns."""
    features = []
    for name in columns:
        role = "feature"
        if name == target:
            role = "target"
        elif name == date:
            role = "date"
        features.append(
            {
                "name": name,
                "dtype": "datetime64[ns]" if role == "date" else "float32",
                "semantic_type": "datetime" if role == "date" else "numeric",
                "role": role,
                "null_count": 0,
                "unique_count": 2 if role == "target" else 100,
                "min": 0.0,
                "max": 100.0,
            }
        )
    return {"features": features}


@pytest.fixture
def datashape_payload() -> Any:
    """Fixture exposing the make_datashape_payload builder to tests."""
    return make_datashape_payload
