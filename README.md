# Performance Plugins

This repo implements two plugins to evaluate the performance of an ML model, with one for classifiers and the other for regressors.
These plugins rely on the [a4s-plugin-interface](https://github.com/lux-ai-factory/a4s-plugin-interface).

Both plugins take a test dataset (`*.csv` or `*.parquet` file), and an `*.onnx` file for the model.

The reported metrics can either be on the entire test set, or on batches given a window size and a frequency (hop size).

## Classification Performance Plugin

> Refer to the [plugin file](a4s_plugin_performance/classif_performance.py) for the full implementation.

Once we compute the predictions on the test set, we use the vectors $y_{pred}$ (the predicted classes), $y_{pred\_proba}$ (the associated probabilites for each class) and $y_{pred}$ (the ground truth) to measure several classification metrics such as the accuracy, f1 score, precision and recall, and evaluate the calibration of our model with measures such as the ECE (expected calibration error).

## Regression Performance Plugin

> Refer to the [plugin file](a4s_plugin_performance/regression_performance.py) for the full implementation.

The regressions measures are computed using the vectors $y_{pred}$ (the outputs of the model) and $y_{pred}$ (the ground truth).
We report for instance the MSE (mean squared error), RMSE (root mean squared error), MAE (mean absolute error) and R2.

# Repo Structure

```bash
.
├── datasets
│   ├── classification
│   │   ├── rf_cls_model.onnx
│   │   ├── testing_data.csv
│   │   └── training_data.csv
│   ├── regression
│   │   ├── rf_reg_model.onnx
│   │   ├── testing_data.csv
│   │   └── training_data.csv
│   └── time_series
│       ├── rf_TS_model.onnx
│       ├── testing_data.csv
│       └── training_data.csv
├── pyproject.toml
├── README.md
├── src
│   └── a4s_plugin_performance
│       ├── __init__.py
│       ├── base_performance_plugin.py
│       ├── classification
│       │   ├── __init__.py
│       │   ├── calibration_metrics.py
│       │   └── plugin.py
│       ├── config_form.py
│       ├── data_input_provider.py
│       ├── iterators.py
│       ├── model_input_provider.py
│       ├── regression
│       │   ├── __init__.py
│       │   └── plugin.py
│       └── utils.py
└── uv.lock
```
