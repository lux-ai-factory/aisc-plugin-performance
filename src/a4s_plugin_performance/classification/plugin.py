from datetime import datetime
from functools import partial
from typing import TYPE_CHECKING, Any

from a4s_plugin_interface import TaskProgress
from a4s_plugin_interface.models.measure import MetricVisualization, ChartType

from ..utils import add_metrics, merge_dicts
from ..base_performance_plugin import BasePerformanceEvaluationPlugin
from ..data_input_provider import DataFrameProvider
from ..model_input_provider import OnnxInputProvider

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


@add_metrics
class ClassificationPerformancePlugin(BasePerformanceEvaluationPlugin):
    plugin_name = "Classification Performance"

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

    @property
    def display_icon(self) -> str:
        return "category"

    def _calculate_metrics(
        self,
        y_true: "npt.NDArray[np.integer[Any]]",
        y_pred_proba: "npt.NDArray[np.floating[Any]]",
        y_pred: "npt.NDArray[np.integer[Any]]",
        date: datetime | None = None,
    ) -> dict[str, dict[str, Any]]:
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix,
            matthews_corrcoef,
        )
        from .calibration_metrics import classification_calibration_score_metrics

        performance_metric_functions = [
            accuracy_score,
            partial(precision_score, zero_division=0),
            partial(recall_score, zero_division=0),
            partial(f1_score, zero_division=0),
            confusion_matrix,
            lambda y_true, y_pred: (matthews_corrcoef(y_true, y_pred) + 1) / 2,
        ]

        metrics = {
            name: {"score": fct(y_true, y_pred), "date": date}
            for name, fct in zip(
                self.performance_metric_names, performance_metric_functions
            )
        }
        calibration_dicts = classification_calibration_score_metrics(
            y_true, y_pred_proba, y_pred
        )
        for d in calibration_dicts:
            metrics[d["name"]] = {"score": d["score"], "date": date}

        return metrics

    def evaluate(self, config_data: dict[str, Any]) -> dict[str, dict[str, list[Any]]]:
        import numpy as np
        import pandas as pd

        self.logger.info("Starting classification evaluation")
        config = self.validate_config_form_data(config_data)

        target_col = config.target_feature
        date_feature = config.date_feature
        frequency = config.frequency
        window_size = config.window_size
        # TODO: add `date_round`

        columns_features = [
            f.name for f in config.features if f.name not in (target_col, date_feature)
        ]
        self.logger.debug(
            "Config: target=%s, date_feature=%s, frequency=%s, window=%s, features=%d",
            target_col,
            date_feature,
            frequency,
            window_size,
            len(columns_features),
        )

        df_test: pd.DataFrame = self.get_dataset()["test"]
        x_test_np = df_test[columns_features].to_numpy()
        y_true = df_test[target_col].to_numpy()
        self.logger.debug(
            "Test data shape: %s, target shape: %s", x_test_np.shape, y_true.shape
        )

        assert isinstance(self.model_input_provider, OnnxInputProvider)
        self.logger.debug("Running model predictions")
        y_pred_proba = self.model_input_provider.predict(x_test_np, probabilities=True)
        y_pred = np.argmax(y_pred_proba, axis=1)
        self.logger.debug(
            "Predictions shape: %s, probabilities shape: %s",
            y_pred.shape,
            y_pred_proba.shape,
        )

        assert isinstance(self.dataset_input_provider, DataFrameProvider)
        dates_masks = list(
            self.dataset_input_provider.iter(date_feature, frequency, window_size)
        )
        iterations = len(dates_masks)
        self.logger.info("Processing %d time windows", iterations)

        results = []
        for i, (date, mask) in enumerate(dates_masks, start=1):
            self.logger.debug(
                "Processing window %d/%d (date=%s, samples=%d)",
                i,
                iterations,
                date,
                mask.sum(),
            )
            results.append(
                self._calculate_metrics(
                    y_true[mask], y_pred_proba[mask], y_pred[mask], date=date
                )
            )
            self.report_progress(
                TaskProgress(progress=i / iterations, extra={"iteration": i})
            )

        self.logger.info("Classification evaluation completed")
        return merge_dicts(results)

    def get_metric_visualizations(self, config_data: dict) -> list[MetricVisualization]:
        config = self.validate_config_form_data(config_data)

        table = MetricVisualization(
            chart_type=ChartType.TABLE, metrics=self.get_metrics()
        )

        performance_chart_metrics = [
            metric_name
            for metric_name in self.performance_metric_names
            if "Matrix" not in metric_name
        ]

        # df_date_iterator = self.dataset_input_provider.iter(
        #     config.date_feature, config.frequency, config.window_size
        # )
        # is_multivalued = len(list(islice(df_date_iterator, 2))) > 1

        is_multivalued = config.date_feature and config.frequency and config.window_size
        per_chart_type = ChartType.LINE if is_multivalued else ChartType.RADAR
        cal_chart_type = ChartType.LINE if is_multivalued else ChartType.BARS

        charts = [
            MetricVisualization(
                chart_type=per_chart_type, metrics=performance_chart_metrics
            ),
            MetricVisualization(
                chart_type=cal_chart_type, metrics=["Confusion-Matrix"]
            ),
            MetricVisualization(
                chart_type=cal_chart_type, metrics=self.calibration_metric_names
            ),
        ]

        return [table, *charts]
