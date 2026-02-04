from a4s_plugin_interface.models.measure import Measure, MetricVisualization, ChartType

from .iterators import DateIterator
from .utils import PerformancePluginFromDatasetConfig, add_metrics, merge_dicts


@add_metrics
class RegressionPerformancePlugin(PerformancePluginFromDatasetConfig):
    metric_names = [
        "Mean Absolute Error",
        "Mean Squared Error",
        "Root Mean Squared Error",
        "Explained Variance",
        "R2",
    ]

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

        metric_functions = [
            mean_absolute_error,
            root_mean_squared_error,
            mean_squared_error,
            explained_variance_score,
            r2_score,
        ]

        return {
            name: {"score": fct(y_true, y_pred), "date": date}
            for name, fct in zip(self.metric_names, metric_functions)
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

        df_test: pd.DataFrame = self.get_dataset()
        session: InferenceSession = self.get_model()

        target_col = config.target_feature
        date_col = config.date_feature
        frequency = config.frequency
        window_size = config.frequency
        # TODO: add `date_round`

        columns_features = [
            f.name for f in config.features if f.name not in (target_col, date_col)
        ]

        if date_col is not None:
            df_test[date_col] = pd.to_datetime(df_test[date_col])

        x_test_np = df_test[columns_features].to_numpy()
        y_true = df_test[target_col].to_numpy()

        y_pred = self._get_y_pred(session, x_test_np)

        df_date_iterator = DateIterator(df_test, date_col, frequency, window_size)
        return merge_dicts(
            [
                self._calculate_metrics(y_true[mask], y_pred[mask], date=date)
                for date, mask in df_date_iterator
            ]
        )

    def get_metric_visualizations(
        self, is_multivalued: bool = False
    ) -> list[MetricVisualization]:
        table = MetricVisualization(
            chart_type=ChartType.TABLE, metrics=self.get_metrics()
        )

        chart_type = ChartType.LINE if is_multivalued else ChartType.BARS
        vis = MetricVisualization(chart_type=chart_type, metrics=self.metric_names)

        return [table, vis]
