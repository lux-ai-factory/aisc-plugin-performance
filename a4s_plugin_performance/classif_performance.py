from functools import partial

from a4s_plugin_interface.models.measure import Measure, MetricVisualization, ChartType

from .iterators import DateIterator
from .utils import PerformancePluginFromDatasetConfig, add_metrics, merge_dicts


@add_metrics
class ClassificationPerformancePlugin(PerformancePluginFromDatasetConfig):
    metric_names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "Confusion-Matrix",
        "MCC",
    ]

    @property
    def display_icon(self) -> str:
        return "category"

    def _calculate_metrics(self, y_true, y_pred, date=None):
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix,
            matthews_corrcoef,
        )

        metric_functions = [
            accuracy_score,
            partial(precision_score, zero_division=0),
            partial(recall_score, zero_division=0),
            partial(f1_score, zero_division=0),
            confusion_matrix,
            matthews_corrcoef,
        ]

        return {
            name: {"score": fct(y_true, y_pred), "date": date}
            for name, fct in zip(self.metric_names, metric_functions)
        }

    def _get_y_pred_probs(self, session, x_test_np):
        import numpy as np

        # Cast to expected dtype
        expected = session.get_inputs()[0].type  # e.g., 'tensor(double)'
        dtype = (
            np.float32
            if "tensor(float)" in expected
            else (np.float64 if "tensor(double)" in expected else x_test_np.dtype)
        )
        x = np.ascontiguousarray(x_test_np.astype(dtype, copy=False))

        # Pick a robust probability output
        out_candidates = session.get_outputs()
        # Prefer a 'prob'/'probab' output if available
        idx = next(
            (i for i, o in enumerate(out_candidates) if "prob" in o.name.lower()), None
        )
        if idx is None:
            idx = 1 if len(out_candidates) >= 2 else 0
        label_name = out_candidates[idx].name

        raw = session.run([label_name], {session.get_inputs()[0].name: x})[0]

        # Handle ZipMap (list of dicts) -> array
        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            keys = list(raw[0].keys())
            probs = np.array(
                [[row.get(k, 0.0) for k in keys] for row in raw], dtype=np.float32
            )
            return probs

        # Otherwise assume it's already a tensor
        arr = np.array(raw)
        if arr.ndim == 1:
            arr = arr[:, None]
        return arr.astype(np.float32, copy=False)

    def evaluate(self, config_data: dict) -> list[Measure]:
        from onnxruntime import InferenceSession
        import numpy as np
        import pandas as pd

        config = self.validate_config_form_data(config_data)

        df_test: pd.DataFrame = self.get_dataset()
        session: InferenceSession = self.get_model()

        target_col = config.target_feature
        date_col = config.date_feature
        frequency = config.frequency
        window_size = config.window_size
        # TODO: add `date_round`

        columns_features = [
            f.name for f in config.features if f.name not in (target_col, date_col)
        ]

        if date_col is not None:
            df_test[date_col] = pd.to_datetime(df_test[date_col])

        x_test_np = df_test[columns_features].to_numpy()
        y_true = df_test[target_col].to_numpy()

        y_pred_proba = self._get_y_pred_probs(session, x_test_np)
        y_pred = np.argmax(y_pred_proba, axis=1)

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

        line_chart_metrics = [
            metric_name
            for metric_name in self.metric_names
            if "Matrix" not in metric_name
        ]

        chart_type = ChartType.LINE if is_multivalued else ChartType.RADAR
        vis = MetricVisualization(chart_type=chart_type, metrics=line_chart_metrics)

        return [table, vis]
