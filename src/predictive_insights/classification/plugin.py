from datetime import datetime
from functools import partial
from typing import TYPE_CHECKING, Any

from aisc_plugin_interface import MetricVisualization, ChartType, MetricDirection

from ..utils import add_metrics, group_metrics
from ..base_performance_plugin import BasePerformanceEvaluationPlugin
from ..data_input_provider import dataframe_iter
from ..model_input_provider import OnnxModelSession

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


@add_metrics
class ClassificationPerformancePlugin(BasePerformanceEvaluationPlugin):
    plugin_name = "Classification Performance"

    description = (
        "Evaluates a trained ONNX classification model against a test dataset, reporting "
        "accuracy, precision, recall, F1-score, MCC, the full confusion matrix, and the "
        "calibration metrics SCE, ECE and MCE.\n\n"
        "**Configuration:**\n\n"
        "- **Frequency** — data frequency of batch starts for time-series analysis "
        "(e.g. `30D`, `1M`). Leave empty to evaluate the whole test set at once.\n"
        "- **Window Size** — duration of each analysis window (e.g. `90 days`, `3 months`); "
        "together with Frequency this splits the evaluation into sliding time windows.\n\n"
        "**Try it:** upload a test dataset and an ONNX model, and attach the project's "
        "**Data Shape** so the target and date columns are resolved automatically."
    )

    ui_icon = "category"

    performance_metric_names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "Confusion-Matrix",
        "MCC",
    ]

    calibration_metric_names = [
        "SCE",
        "ECE",
        "MCE",
    ]

    @classmethod
    def metric_names(cls) -> list[str]:
        return cls.performance_metric_names + cls.calibration_metric_names

    def _calculate_metrics(
        self,
        y_true: "npt.NDArray[np.integer[Any]]",
        y_pred_proba: "npt.NDArray[np.floating[Any]]",
        y_pred: "npt.NDArray[np.integer[Any]]",
        date: datetime | None = None,
    ) -> list[dict[str, dict[str, Any]]]:
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix,
            matthews_corrcoef,
        )
        from .calibration_metrics import classification_calibration_score_metrics

        if date is None:
            date = datetime.now()

        # Simple performance metrics
        performance_metric_functions = [
            accuracy_score,
            partial(precision_score, zero_division=0, average="weighted"),
            partial(recall_score, zero_division=0, average="weighted"),
            partial(f1_score, zero_division=0, average="weighted"),
        ]
        performance_metric_directions = [
            MetricDirection.HIGHER_IS_BETTER,
            MetricDirection.HIGHER_IS_BETTER,
            MetricDirection.HIGHER_IS_BETTER,
            MetricDirection.HIGHER_IS_BETTER,
        ]

        metrics = [
            {name: {"score": fct(y_true, y_pred), "time": date, "direction": direction}}
            for name, fct, direction in zip(
                self.performance_metric_names,
                performance_metric_functions,
                performance_metric_directions,
            )
        ]

        # Confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        max_i, max_j = conf_matrix.shape

        metrics.extend(
            {
                self.performance_metric_names[-2]: {
                    "score": float(conf_matrix[i][j]),
                    "time": date,
                    "description": f"({i + 1},{j + 1})/({max_i},{max_j})",
                }
            }
            for i in range(max_i)
            for j in range(max_j)
        )

        # MCC
        metrics.append(
            {
                self.performance_metric_names[-1]: {
                    "score": (matthews_corrcoef(y_true, y_pred) + 1) / 2,
                    "time": date,
                }
            }
        )

        # Calibration
        calibration_dicts = classification_calibration_score_metrics(y_true, y_pred_proba, y_pred)
        for d in calibration_dicts:
            metrics.append({d["name"]: {"score": d["score"], "time": date}})

        return metrics

    def evaluate(self, config_data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        import numpy as np
        import pandas as pd
        from onnxruntime import InferenceSession

        self.logger.info("Starting classification evaluation")
        config = self.validate_config_form_data(config_data)

        target_col, date_feature, columns_features = self._read_datashape()
        frequency = config.frequency.strip()
        window_size = config.window_size.strip()
        # TODO: add `date_round`

        if not columns_features:
            self.logger.warning("No input features found with role 'feature' in the data shape")

        self.logger.debug(
            "Config: target=%s, date_feature=%s, frequency=%s, window=%s, features=%d",
            target_col,
            date_feature,
            frequency,
            window_size,
            len(columns_features),
        )

        try:
            df_test = self.get_input_data("test-dataset")
        except Exception:
            self.logger.exception("Failed to load test dataset")
            raise

        assert isinstance(df_test, pd.DataFrame)
        x_test_np = df_test[columns_features].to_numpy()
        y_true = df_test[target_col].to_numpy()
        self.logger.debug("Test data shape: %s, target shape: %s", x_test_np.shape, y_true.shape)

        session = self.get_input_data("model")
        assert isinstance(session, InferenceSession)
        model_session = OnnxModelSession(session)
        self.logger.debug("Running model predictions")

        try:
            y_pred_proba = model_session.predict(x_test_np, probabilities=True)
        except Exception:
            self.logger.exception("Model prediction failed for input shape %s", x_test_np.shape)
            raise

        y_pred = np.argmax(y_pred_proba, axis=1)
        self.logger.debug(
            "Predictions shape: %s, probabilities shape: %s",
            y_pred.shape,
            y_pred_proba.shape,
        )

        df = self.get_input_data("test-dataset")
        assert isinstance(df, pd.DataFrame)
        dates_masks = list(dataframe_iter(df, date_feature, frequency, window_size))
        iterations = len(dates_masks)
        self.logger.info("Processing %d time windows", iterations)

        # save prediction artifact
        df_pred = pd.DataFrame(y_pred_proba, index=df.index)
        if date_feature is not None and date_feature in df.columns:
            df_pred.insert(0, date_feature, df[date_feature])
        self.upload_artifact("predictions.csv", df_pred.to_csv(index=True).encode("utf-8"))

        results = []
        pbar = self.progress_bar(
            dates_masks,
            total=iterations,
            start=1,
            with_index=True,
            desc="Classification metrics",
        )
        for i, (date, mask) in pbar:
            if mask.sum() == 0:
                self.logger.warning(
                    "Window %d/%d (date=%s) has no samples, skipping",
                    i,
                    iterations,
                    date,
                )
                continue

            self.logger.debug(
                "Processing window %d/%d (date=%s, samples=%d)",
                i,
                iterations,
                date,
                mask.sum(),
            )

            try:
                results.extend(
                    self._calculate_metrics(
                        y_true[mask], y_pred_proba[mask], y_pred[mask], date=date
                    )
                )
            except Exception:
                self.logger.exception(
                    "Metric calculation failed for window %d/%d (date=%s)",
                    i,
                    iterations,
                    date,
                )
                raise

        self.logger.info("Classification evaluation completed")
        return group_metrics(results)

    def get_metric_visualizations(self, config_data: dict) -> list[MetricVisualization]:
        config = self.validate_config_form_data(config_data)

        performance_chart_metrics = [
            metric_name
            for metric_name in self.performance_metric_names
            if "Matrix" not in metric_name
        ]

        # df_date_iterator = self.dataset_input_provider.iter(
        #     config.date_feature, config.frequency, config.window_size
        # )
        # is_multivalued = len(list(islice(df_date_iterator, 2))) > 1

        is_multivalued = bool(config.frequency.strip() and config.window_size.strip())
        per_chart_type = ChartType.LINE if is_multivalued else ChartType.RADAR
        # cal_chart_type = ChartType.LINE if is_multivalued else ChartType.BARS

        charts = [
            # MetricVisualization(
            #     chart_type=ChartType.TABLE,
            #     title="Classification Evaluation Metrics",
            #     description="All the metrics for the classification model evaluation.",
            #     metrics=self.get_metrics(),
            # ),
            MetricVisualization(
                chart_type=per_chart_type,
                title="Classification Performance Metrics",
                description="The performance metrics for the classification model.",
                metrics=performance_chart_metrics,
            ),
            # MetricVisualization(
            #     chart_type=cal_chart_type,
            #     title="Confusion Matrix",
            #     description="The confusion matrix for the classification model.",
            #     metrics=["Confusion-Matrix"],
            # ),
            # MetricVisualization(
            #     chart_type=cal_chart_type,
            #     title="Calibration Metrics",
            #     description="The calibration metrics for the classification model.",
            #     metrics=self.calibration_metric_names,
            # ),
        ]

        return charts
