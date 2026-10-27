"""Deterministic test-fixture generator for aisc-plugin-performance.

Derives a DataShape payload from each **train** dataset in ``datasets/`` and
writes self-contained case folders into ``tests/files/<case>/`` (gitignored):

    tests/files/
    ├── classification/   dataset.csv + model.onnx + datashape.json + config.json
    ├── regression/       dataset.csv + model.onnx + datashape.json + config.json
    └── time_series/      datashape.json + config.json   (inputs too large to copy)

Regenerate with:

    uv run python tests/generate_files.py

A tests/conftest.py hook regenerates missing fixtures automatically at
collection time, so a fresh clone only needs ``git clone && uv sync && pytest``.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from predictive_insights.config_form import ConfigForm  # noqa: E402

FILES = Path(__file__).resolve().parent / "files"

# Categorical thresholds (20/50/0.05) mirror the platform's feature derivation.
CATEGORY_SMALL = 20
CATEGORY_ABS_CAP = 50
CATEGORY_REL = 0.05

CASES: dict[str, dict[str, object]] = {
    "classification": {
        "train_csv": "classification/training_data.csv",
        "test_csv": "classification/testing_data.csv",
        "model_onnx": "classification/rf_cls_model.onnx",
        "target": "target",
        "date": None,
        "config": {"frequency": "", "window_size": ""},
    },
    "regression": {
        "train_csv": "regression/training_data.csv",
        "test_csv": "regression/testing_data.csv",
        "model_onnx": "regression/rf_reg_model.onnx",
        "target": "target",
        "date": None,
        "config": {"frequency": "", "window_size": ""},
    },
    "time_series": {
        "train_csv": "time_series/training_data.csv",
        "test_csv": "time_series/testing_data.csv",
        "model_onnx": "time_series/rf_TS_model.onnx",
        "target": "charged_off",
        "date": None,
        "config": {"frequency": "1M", "window_size": "3M"},
    },
}


def build_feature(name: str, series: pd.Series, role: str) -> dict[str, Any]:
    """Derive a real-format feature dict for one training column."""
    unique = int(series.nunique(dropna=True))
    null = int(series.isna().sum())
    n_rows = int(len(series))
    dtype = str(series.dtype)

    datetime_sem = bool(pd.api.types.is_datetime64_any_dtype(series.dtype))
    if not datetime_sem and (
        pd.api.types.is_object_dtype(series.dtype) or isinstance(series.dtype, pd.StringDtype)
    ):
        parsed = pd.to_datetime(series, errors="coerce")
        if bool(parsed.notna().all()):
            datetime_sem = True

    if datetime_sem:
        feature: dict[str, Any] = {
            "name": name,
            "dtype": dtype,
            "semantic_type": "datetime",
            "role": role,
            "null_count": null,
            "unique_count": unique,
        }
        if unique <= CATEGORY_ABS_CAP:
            values = [_json_value(v) for v in series.dropna().unique().tolist()]
            feature["categories"] = values
            feature["category_mapping"] = {str(v): str(v) for v in values}
        return feature

    categorical = unique <= CATEGORY_SMALL or (
        unique <= CATEGORY_ABS_CAP and n_rows > 0 and unique / n_rows < CATEGORY_REL
    )

    if pd.api.types.is_bool_dtype(series.dtype):
        return {
            "name": name,
            "dtype": dtype,
            "semantic_type": "boolean",
            "role": role,
            "null_count": null,
            "unique_count": unique,
            "categories": [False, True],
        }

    if pd.api.types.is_numeric_dtype(series.dtype):
        low, high = series.min(), series.max()
        feature = {
            "name": name,
            "dtype": dtype,
            "semantic_type": "categorical" if categorical else "numeric",
            "role": role,
            "null_count": null,
            "unique_count": unique,
            "min": _json_value(low) if pd.notna(low) else None,
            "max": _json_value(high) if pd.notna(high) else None,
        }
        if categorical:
            values = [_json_value(v) for v in series.dropna().unique().tolist()]
            feature["categories"] = values
            feature["category_mapping"] = {str(v): str(v) for v in values}
        else:
            feature["mean"] = float(series.mean()) if pd.notna(series.mean()) else None
            feature["std"] = float(series.std()) if pd.notna(series.std()) else None
        return feature

    feature: dict[str, Any] = {
        "name": name,
        "dtype": dtype,
        "semantic_type": "categorical" if categorical else "text",
        "role": role,
        "null_count": null,
        "unique_count": unique,
    }
    if unique <= CATEGORY_ABS_CAP:
        values = [_json_value(v) for v in series.dropna().unique().tolist()]
        feature["categories"] = values
        if categorical:
            feature["category_mapping"] = {str(v): str(v) for v in values}
    return feature


def _json_value(value: object) -> object:
    """Convert numpy scalars to plain Python (JSON-safe) values."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_datashape(case: dict[str, object]) -> dict:
    """Derive a features-only DataShape payload for a case from its TRAIN dataset."""
    import pandas as pd

    train = ROOT / "datasets" / str(case["train_csv"])
    df = pd.read_csv(train)

    target = case["target"]
    date = case["date"]
    features = [
        build_feature(
            col,
            df[col],
            "target" if col == target else ("date" if col == date else "feature"),
        )
        for col in df.columns
    ]
    return {"features": features}


def main() -> None:
    """Generate the tests/files/<case>/ fixtures for every case."""
    FILES.mkdir(parents=True, exist_ok=True)

    for name, case in CASES.items():
        out = FILES / name
        out.mkdir(parents=True, exist_ok=True)

        (out / "datashape.json").write_text(json.dumps(build_datashape(case), indent=2) + "\n")
        (out / "config.json").write_text(
            json.dumps(ConfigForm(**cast("dict[str, Any]", case["config"])).model_dump(), indent=2)
            + "\n"
        )

        # Copy eval inputs so small case folders are self-contained. The
        # time_series dataset (65 MB) and ONNX model (408 MB) stay in datasets/.
        if name != "time_series":
            shutil.copyfile(ROOT / "datasets" / str(case["test_csv"]), out / "dataset.csv")
            shutil.copyfile(ROOT / "datasets" / str(case["model_onnx"]), out / "model.onnx")

    print(f"Generated test fixtures in {FILES}")


if __name__ == "__main__":
    main()
