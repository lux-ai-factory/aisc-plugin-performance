from a4s_plugin_interface import TaskProgress
from a4s_plugin_interface.models.measure import Measure, MetricVisualization, ChartType

from .utils import PerformancePluginFromDatasetConfig, add_metrics, merge_dicts


@add_metrics
class RegressionPerformancePlugin(PerformancePluginFromDatasetConfig):
    performance_metric_names = [
        "Mean Absolute Error",
        "Mean Squared Error",
        "Root Mean Squared Error",
        "Explained Variance",
        "R2",
    ]

    @classmethod
    def metric_names(cls):
        return cls.performance_metric_names

    @property
    def display_icon(self) -> str:
        return "trending_up"

    def _calculate_metrics(self, y_true, y_pred, date=None):
        from sklearn.metrics import (
            mean_absolute_error,
            root_mean_squared_error,
            mean_squared_error,
            explained_variance_score,
            r2_score,
        )

        performance_metric_functions = [
            mean_absolute_error,
            root_mean_squared_error,
            mean_squared_error,
            explained_variance_score,
            r2_score,
        ]

        return {
            name: {"score": fct(y_true, y_pred), "date": date}
            for name, fct in zip(
                self.performance_metric_names, performance_metric_functions
            )
        }

    def _get_y_pred(self, session, x_test_np):
        import numpy as np

        expected = session.get_inputs()[0].type  # e.g., "tensor(double)"
        if "tensor(double)" in expected:
            dtype = np.float64
        elif "tensor(float)" in expected:
            dtype = np.float32
        else:
            dtype = x_test_np.dtype  # fallback

        X = np.asarray(x_test_np, dtype=dtype)
        X = np.ascontiguousarray(X)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        pred_onx = session.run([output_name], {input_name: X})[0]
        y_pred = np.asarray(pred_onx).squeeze(-1)
        return y_pred

    def evaluate(self, config_data: dict) -> list[Measure]:
        import pandas as pd
        from onnxruntime import InferenceSession

        config = self.validate_config_form_data(config_data)

        target_col = config.target_feature
        date_feature = config.date_feature
        frequency = config.frequency
        window_size = config.window_size
        # TODO: add `date_round`

        columns_features = [
            f.name for f in config.features if f.name not in (target_col, date_feature)
        ]

        df_test: pd.DataFrame = self.get_dataset()["test"]
        x_test_np = df_test[columns_features].to_numpy()
        y_true = df_test[target_col].to_numpy()

        session: InferenceSession = self.get_model()
        y_pred = self._get_y_pred(session, x_test_np)

        dates_masks = list(
            self.dataset_input_provider.iter(date_feature, frequency, window_size)
        )
        iterations = len(dates_masks)

        results = []
        for i, (date, mask) in enumerate(dates_masks, start=1):
            results.append(
                self._calculate_metrics(y_true[mask], y_pred[mask], date=date)
            )
            self.report_progress(
                TaskProgress(progress=i / iterations, extra={"iteration": i})
            )

        return merge_dicts(results)

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
