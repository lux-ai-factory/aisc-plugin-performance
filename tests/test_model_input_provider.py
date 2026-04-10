"""Tests for ONNX model input provider."""

import pytest
import numpy as np

from vera_plugin_performance.model_input_provider import (
    OnnxInputProvider,
    OnnxModelSession,
)


class TestOnnxInputProvider:
    """Tests for OnnxInputProvider class."""

    @pytest.fixture
    def simple_onnx_model(self):
        """Create a simple ONNX model for testing."""
        from sklearn.linear_model import LogisticRegression
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        # Train a simple model
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        model = LogisticRegression()
        model.fit(X, y)

        # Convert to ONNX
        initial_type = [("float_input", FloatTensorType([None, 2]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        return onnx_model.SerializeToString()

    @pytest.fixture
    def regression_onnx_model(self):
        """Create a simple regression ONNX model for testing."""
        from sklearn.linear_model import LinearRegression
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0, 4.0])
        model = LinearRegression()
        model.fit(X, y)

        initial_type = [("float_input", FloatTensorType([None, 2]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        return onnx_model.SerializeToString()

    def test_load_onnx_model(self, simple_onnx_model):
        provider = OnnxInputProvider(simple_onnx_model)
        data = provider.get_data()
        assert data is not None

    def test_predict_with_numpy_array(self, simple_onnx_model):
        provider = OnnxInputProvider(simple_onnx_model)
        model_session = OnnxModelSession(provider.get_data())
        X = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
        predictions = model_session.predict(X, probabilities=False)
        assert predictions.shape[0] == 2
        assert predictions.dtype == np.float32
        # Class predictions should be 0 or 1
        assert all(p in [0.0, 1.0] for p in predictions)

    def test_predict_with_dataframe(self, simple_onnx_model):
        import pandas as pd

        provider = OnnxInputProvider(simple_onnx_model)
        model_session = OnnxModelSession(provider.get_data())
        X = pd.DataFrame({"a": [1.5, 3.5], "b": [2.5, 4.5]})
        predictions = model_session.predict(X, probabilities=False)
        assert predictions.shape[0] == 2

    def test_predict_probabilities(self, simple_onnx_model):
        provider = OnnxInputProvider(simple_onnx_model)
        model_session = OnnxModelSession(provider.get_data())
        X = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
        proba = model_session.predict(X, probabilities=True)
        assert proba.shape[0] == 2
        assert proba.shape[1] == 2  # Binary classification
        # Probabilities should sum to ~1
        np.testing.assert_array_almost_equal(proba.sum(axis=1), [1.0, 1.0], decimal=5)

    def test_predict_regression(self, regression_onnx_model):
        provider = OnnxInputProvider(regression_onnx_model)
        model_session = OnnxModelSession(provider.get_data())
        X = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
        predictions = model_session.predict(X, probabilities=False)
        assert predictions.shape[0] == 2
        assert predictions.dtype == np.float32

    def test_predict_invalid_input_type(self, simple_onnx_model):
        provider = OnnxInputProvider(simple_onnx_model)
        model_session = OnnxModelSession(provider.get_data())
        with pytest.raises(ValueError, match="x_test should be np.ndarray"):
            model_session.predict([[1, 2], [3, 4]], probabilities=False)

    def test_predict_without_loading_model(self):
        """OnnxInputProvider with None raises TypeError during initialization."""
        with pytest.raises(TypeError, match="Unable to load from type"):
            OnnxInputProvider(None)  # ty: ignore[invalid-argument-type]
