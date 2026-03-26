# a4s-plugin-performance

[![CI](https://github.com/lux-ai-factory/a4s-plugin-performance/actions/workflows/ci.yml/badge.svg)](https://github.com/lux-ai-factory/a4s-plugin-performance/actions/workflows/ci.yml)

A Python plugin for ML model performance evaluation, implementing classification and regression metrics. Built on the [a4s-plugin-interface](https://github.com/lux-ai-factory/a4s-plugin-interface) framework.

## Features

- **Classification metrics**: Accuracy, Precision, Recall, F1-Score, MCC, Confusion Matrix
- **Calibration metrics**: ECE (Expected Calibration Error), MCE (Maximum Calibration Error), SCE (Static Calibration Error)
- **Regression metrics**: MAE, MSE, RMSE, R2, Explained Variance
- **Time-windowed evaluation**: Compute metrics over sliding windows with configurable frequency
- **Multiple input formats**: CSV and Parquet datasets, ONNX models

## Installation

```bash
uv sync
```

## Quick Start

Both plugins take a test dataset (`*.csv` or `*.parquet`) and an ONNX model file (`*.onnx`).

```python
from a4s_plugin_performance import ClassificationPerformancePlugin

plugin = ClassificationPerformancePlugin()
plugin.set_dataset_input_provider(test_data_bytes)
plugin.set_model_input_provider(model_bytes)

results = plugin.evaluate({
    "features": [...],
    "target_feature": "label",
    "date_feature": "timestamp",  # optional
    "frequency": "7D",            # optional
    "window_size": "30D",         # optional
})
```

## Plugins

### Classification Performance Plugin

Computes classification metrics using predicted classes (`y_pred`), predicted probabilities (`y_pred_proba`), and ground truth (`y_true`).

**Metrics:**
| Metric | Description |
|--------|-------------|
| Accuracy | Proportion of correct predictions |
| Precision | True positives / (True positives + False positives) |
| Recall | True positives / (True positives + False negatives) |
| F1-Score | Harmonic mean of precision and recall |
| MCC | Matthews Correlation Coefficient (normalized to [0,1]) |
| Confusion Matrix | Full confusion matrix |
| ECE | Expected Calibration Error |
| MCE | Maximum Calibration Error |
| SCE | Static Calibration Error |

### Regression Performance Plugin

Computes regression metrics using model outputs (`y_pred`) and ground truth (`y_true`).

**Metrics:**
| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error |
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| R2 | Coefficient of determination |
| Explained Variance | Variance explained by the model |

## Time-Windowed Evaluation

Metrics can be computed over the entire test set or over temporal windows:

```python
config = {
    "date_feature": "timestamp",  # Column containing dates
    "frequency": "7D",            # Window hop size (e.g., weekly)
    "window_size": "30D",         # Window duration (e.g., 30 days)
}
```

This produces metrics for each time window, useful for monitoring model performance over time.

## Development

### Setup

```bash
# Install dependencies
uv sync

# Install pre-commit hooks
uv run pre-commit install
```

### Commands

```bash
# Run tests
uv run pytest

# Run a single test
uv run pytest tests/test_utils.py::TestMergeDicts::test_merge_single_dict

# Linting
uv run ruff check src/

# Linting with auto-fix
uv run ruff check --fix src/

# Type checking
uv run ty check src/

# Format code
uv run ruff format src/

# Run all pre-commit hooks manually
uv run pre-commit run --all-files
```

### Pre-commit Hooks

This project uses [pre-commit](https://pre-commit.com/) to run checks before each commit:

- **Ruff** - Linting and formatting
- **ty** - Type checking

Hooks are installed automatically when you run `uv run pre-commit install`. To skip hooks temporarily:

```bash
git commit --no-verify -m "message"
```

### Project Structure

```
src/a4s_plugin_performance/
├── __init__.py                    # Public exports
├── base_performance_plugin.py     # Abstract base class for plugins
├── config_form.py                 # Pydantic config + UI schema
├── data_input_provider.py         # CSV/Parquet data reader
├── model_input_provider.py        # ONNX model wrapper
├── iterators.py                   # Date windowing utilities
├── utils.py                       # Shared utilities and decorators
├── classification/
│   ├── plugin.py                  # ClassificationPerformancePlugin
│   └── calibration_metrics.py     # ECE, MCE, SCE metrics
└── regression/
    └── plugin.py                  # RegressionPerformancePlugin
```

### Test Datasets

Sample datasets are provided in `datasets/` for testing:

```
datasets/
├── classification/          # Binary classification task
├── regression/              # Regression task
└── time_series/             # Time-series with date column
```

## Tech Stack

- **Python** 3.12+
- **Pydantic** - Configuration and validation
- **NumPy/pandas** - Data processing
- **scikit-learn** - Metric implementations
- **ONNX Runtime** - Model inference
- **uv** - Package management
- **Ruff** - Linting and formatting
- **ty** - Type checking

## Plugin Metadata

| Field | Value |
|-------|-------|
| Name | A4S Performance Plugin |
| Description | A Python plugin for ML model performance evaluation, implementing classification metrics (Accuracy, Precision, Recall, F1-Score, MCC, Confusion Matrix), calibration metrics (ECE, MCE, SCE), and regression metrics (MAE, MSE, RMSE, R2, Explained Variance). Supports time-windowed evaluation and ONNX model inference. |
| License | [MIT] |
| Verification type | Technical test |
| Project | [a4s-plugin-performance](https://github.com/lux-ai-factory/a4s-plugin-performance) |
| Branch | main |
| Version | 0.1.1 |
| Project maturity | Deployed |
| Scientific reference | — |
| Verification targets | [Model Performance] [Classification Metrics] [Regression Metrics] [Calibration Analysis] |
| Sector | [AI/ML] [Data Science] [MLOps] |
