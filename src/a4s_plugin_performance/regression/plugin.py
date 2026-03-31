from datetime import datetime
from typing import TYPE_CHECKING, Any

from a4s_plugin_interface import TaskProgress
from a4s_plugin_interface.models.measure import MetricVisualization, ChartType

from ..utils import add_metrics, group_metrics
from ..base_performance_plugin import BasePerformanceEvaluationPlugin
from ..data_input_provider import dataframe_iter
from ..model_input_provider import OnnxModelSession

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


@add_metrics
class RegressionPerformancePlugin(BasePerformanceEvaluationPlugin):
    plugin_name = "Regression Performance"

    performance_metric_names = [
        "Mean Absolute Error",
        "Mean Squared Error",
        "Root Mean Squared Error",
        "Explained Variance",
        "R2",
    ]

    @classmethod
    def metric_names(cls) -> list[str]:
        return cls.performance_metric_names

    @property
    def display_icon(self) -> str:
        return "trending_up"

    def _calculate_metrics(
        self,
        y_true: "npt.NDArray[np.floating[Any]]",
        y_pred: "npt.NDArray[np.floating[Any]]",
        date: datetime | None = None,
    ) -> dict[str, dict[str, Any]]:
        from sklearn.metrics import (
            mean_absolute_error,
            root_mean_squared_error,
            mean_squared_error,
            explained_variance_score,
            r2_score,
        )

        if date is None:
            date = datetime.now()

        performance_metric_functions = [
            mean_absolute_error,
            root_mean_squared_error,
            mean_squared_error,
            explained_variance_score,
            r2_score,
        ]

        return {
            name: {"score": fct(y_true, y_pred), "time": date}
            for name, fct in zip(
                self.performance_metric_names, performance_metric_functions
            )
        }

    def evaluate(self, config_data: dict[str, Any]) -> dict[str, dict[str, list[Any]]]:
        import pandas as pd
        from onnxruntime import InferenceSession

        self.logger.info("Starting regression evaluation")
        config = self.validate_config_form_data(config_data)

        target_col = config.target_feature
        date_feature = config.date_feature
        frequency = config.frequency
        window_size = config.window_size
        # TODO: add `date_round`

        columns_features = [
            f.name for f in config.features if f.name not in (target_col, date_feature)
        ]

        if not columns_features:
            self.logger.warning(
                "No input features found after excluding target and date"
            )

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
        self.logger.debug(
            "Test data shape: %s, target shape: %s", x_test_np.shape, y_true.shape
        )

        session = self.get_input_data("model")
        assert isinstance(session, InferenceSession)
        model_session = OnnxModelSession(session)
        self.logger.debug("Running model predictions")

        try:
            y_pred = model_session.predict(x_test_np, probabilities=False)
        except Exception:
            self.logger.exception(
                "Model prediction failed for input shape %s", x_test_np.shape
            )
            raise

        self.logger.debug("Predictions shape: %s", y_pred.shape)

        df = self.get_input_data("test-dataset")
        assert isinstance(df, pd.DataFrame)
        dates_masks = list(dataframe_iter(df, date_feature, frequency, window_size))
        iterations = len(dates_masks)
        self.logger.info("Processing %d time windows", iterations)

        results = []
        for i, (date, mask) in enumerate(dates_masks, start=1):
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
                results.append(
                    self._calculate_metrics(y_true[mask], y_pred[mask], date=date)
                )
            except Exception:
                self.logger.exception(
                    "Metric calculation failed for window %d/%d (date=%s)",
                    i,
                    iterations,
                    date,
                )
                raise

            self.report_progress(
                TaskProgress(progress=i / iterations, extra={"iteration": i})
            )

        self.logger.info("Regression evaluation completed")
        return group_metrics(results)

    def get_metric_visualizations(self, config_data: dict) -> list[MetricVisualization]:
        config = self.validate_config_form_data(config_data)

        table = MetricVisualization(
            chart_type=ChartType.TABLE, metrics=self.get_metrics()
        )

        is_multivalued = config.date_feature and config.frequency and config.window_size
        chart_type = ChartType.LINE if is_multivalued else ChartType.BARS
        vis = MetricVisualization(
            chart_type=chart_type, metrics=self.performance_metric_names
        )

        return [table, vis]
