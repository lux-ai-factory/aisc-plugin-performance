"""Test that sklearn predictions match ONNX model provider predictions."""

from pathlib import Path

import numpy as np

DATASETS_DIR = Path(__file__).parent.parent / "datasets"


def train_sklearn_converted_to_onnx(X_train, y_train, model_type="classification"):
    """Train a sklearn model and convert to ONNX with zipmap=False."""
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    if model_type == "classification":
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            class_weight="balanced",
            n_estimators=100,
            max_depth=17,
            min_samples_leaf=15,
            min_samples_split=15,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
    else:
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=17,
            min_samples_leaf=15,
            min_samples_split=15,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )

    model.fit(X_train, y_train)

    initial_type = [("float_input", FloatTensorType([None, X_train.shape[1]]))]
    options = {id(model): {"zipmap": False}} if model_type == "classification" else None
    onnx_model = convert_sklearn(model, initial_types=initial_type, options=options)
    return model, onnx_model.SerializeToString()


class TestSklearnVsOnnx:
    def _run_comparison(self, dataset_name, model_type, target_col, drop_cols=None):
        from predictive_insights.model_input_provider import (
            OnnxModelSession,
            OnnxInputProvider,
        )
        import pandas as pd

        train_df = pd.read_csv(DATASETS_DIR / dataset_name / "training_data.csv")
        test_df = pd.read_csv(DATASETS_DIR / dataset_name / "testing_data.csv")

        if len(train_df) > 1000:
            train_df = train_df.sample(n=1000, random_state=42)

        drop_cols = drop_cols or []
        feature_cols = [
            c for c in train_df.columns if c not in drop_cols + [target_col]
        ]
        X_train = train_df[feature_cols].to_numpy()
        y_train = train_df[target_col].to_numpy()
        X_test = test_df[feature_cols].to_numpy()

        sk_model, onnx_bytes = train_sklearn_converted_to_onnx(
            X_train, y_train, model_type
        )

        provider = OnnxInputProvider(onnx_bytes)
        session = OnnxModelSession(provider.get_data())

        if model_type == "classification":
            sk_pred = sk_model.predict(X_test).astype(np.float32)
            sk_proba = sk_model.predict_proba(X_test).astype(np.float32)
            onnx_pred = session.predict(X_test, probabilities=False)
            onnx_proba = session.predict(X_test, probabilities=True)

            assert np.array_equal(sk_pred, onnx_pred), (
                f"{dataset_name}: Class predictions don't match\n"
                f"  sklearn: {sk_pred[:10]}\n"
                f"  onnx:    {onnx_pred[:10]}"
            )
            assert np.allclose(sk_proba, onnx_proba, atol=1e-5), (
                f"{dataset_name}: Probabilities don't match\n"
                f"  max diff: {np.abs(sk_proba - onnx_proba).max()}"
            )
        else:
            sk_pred = sk_model.predict(X_test).astype(np.float32)
            onnx_pred = session.predict(X_test, probabilities=False)

            assert np.allclose(sk_pred, onnx_pred, atol=1e-5), (
                f"{dataset_name}: Regression predictions don't match\n"
                f"  max diff: {np.abs(sk_pred - onnx_pred).max()}"
            )

    def test_classification_dataset(self):
        self._run_comparison(
            dataset_name="classification",
            model_type="classification",
            target_col="target",
            drop_cols=[],
        )

    def test_classification_time_series(self):
        self._run_comparison(
            dataset_name="time_series",
            model_type="classification",
            target_col="charged_off",
            drop_cols=["issue_d"],
        )

    def test_regression_dataset(self):
        self._run_comparison(
            dataset_name="regression",
            model_type="regression",
            target_col="target",
            drop_cols=[],
        )
