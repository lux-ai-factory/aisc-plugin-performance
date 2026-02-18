from functools import partial

from a4s_plugin_interface import TaskProgress
from a4s_plugin_interface.models.measure import Measure, MetricVisualization, ChartType

from .utils import PerformancePluginFromDatasetConfig, add_metrics, merge_dicts


@add_metrics
class ClassificationPerformancePlugin(PerformancePluginFromDatasetConfig):
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
    def metric_names(cls):
        return cls.performance_metric_names + cls.calibration_metric_names

    @property
    def display_icon(self) -> str:
        return "category"

    def _calculate_metrics(self, y_true, y_pred_proba, y_pred, date=None):
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
            matthews_corrcoef,
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
        y_pred_proba = self._get_y_pred_probs(session, x_test_np)
        y_pred = np.argmax(y_pred_proba, axis=1)

        dates_masks = list(
            self.dataset_input_provider.iter(date_feature, frequency, window_size)
        )
        iterations = len(dates_masks)

        results = []
        for i, (date, mask) in enumerate(dates_masks, start=1):
            results.append(
                self._calculate_metrics(
                    y_true[mask], y_pred_proba[mask], y_pred[mask], date=date
                )
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
