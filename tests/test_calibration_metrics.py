"""Tests for calibration metrics."""

import pytest
import numpy as np

from predictive_insights.classification.calibration_metrics import (
    classification_calibration_score_metrics,
    expected_calibration_error,
    maximum_calibration_error,
    static_calibration_error,
    _compute_bin_stats,
)


class TestComputeBinStats:
    """Tests for _compute_bin_stats helper function."""

    def test_uniform_distribution(self):
        confidences = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        accuracies = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        bin_counts, avg_conf, avg_acc, nonzero = _compute_bin_stats(
            confidences, accuracies, n_bins=5
        )
        assert len(bin_counts) == 5
        assert len(avg_conf) == 5
        assert len(avg_acc) == 5
        assert nonzero.dtype == bool

    def test_empty_bins(self):
        # All samples in one bin
        confidences = np.array([0.95, 0.96, 0.97, 0.98, 0.99])
        accuracies = np.array([1.0, 1.0, 0.0, 1.0, 1.0])
        bin_counts, avg_conf, avg_acc, nonzero = _compute_bin_stats(
            confidences, accuracies, n_bins=10
        )
        # Most bins should be empty (nonzero=False)
        assert (~nonzero).sum() > 0
        # Empty bins should have 0 avg_conf and avg_acc
        assert all(avg_conf[~nonzero] == 0)
        assert all(avg_acc[~nonzero] == 0)


class TestExpectedCalibrationError:
    """Tests for ECE metric."""

    def test_perfectly_calibrated(self):
        """Perfectly calibrated predictions should have ECE close to 0."""
        # When confidence equals accuracy, ECE should be 0
        bin_counts = np.array([10, 10, 10])
        avg_conf = np.array([0.3, 0.5, 0.8])
        avg_acc = np.array([0.3, 0.5, 0.8])
        nonzero = np.array([True, True, True])
        n_samples = 30
        ece = expected_calibration_error(
            bin_counts, avg_conf, avg_acc, nonzero, n_samples
        )
        assert ece == pytest.approx(0.0, abs=1e-6)

    def test_overconfident_predictions(self):
        """Overconfident predictions should have positive ECE."""
        bin_counts = np.array([100])
        avg_conf = np.array([0.9])  # High confidence
        avg_acc = np.array([0.5])  # But only 50% accurate
        nonzero = np.array([True])
        n_samples = 100
        ece = expected_calibration_error(
            bin_counts, avg_conf, avg_acc, nonzero, n_samples
        )
        assert ece == pytest.approx(0.4, abs=1e-6)  # |0.9 - 0.5| = 0.4


class TestMaximumCalibrationError:
    """Tests for MCE metric."""

    def test_mce_returns_max_gap(self):
        bin_counts = np.array([10, 10, 10])
        avg_conf = np.array([0.2, 0.5, 0.9])
        avg_acc = np.array([0.2, 0.3, 0.5])  # Largest gap at last bin
        nonzero = np.array([True, True, True])
        mce = maximum_calibration_error(bin_counts, avg_conf, avg_acc, nonzero)
        assert mce == pytest.approx(0.4, abs=1e-6)  # |0.9 - 0.5| = 0.4

    def test_mce_ignores_empty_bins(self):
        bin_counts = np.array([10, 0, 10])
        avg_conf = np.array([0.5, 0.99, 0.7])  # Middle bin would have large gap
        avg_acc = np.array([0.5, 0.1, 0.6])  # But it's empty
        nonzero = np.array([True, False, True])
        mce = maximum_calibration_error(bin_counts, avg_conf, avg_acc, nonzero)
        # Should ignore empty middle bin
        assert mce == pytest.approx(0.1, abs=1e-6)  # |0.7 - 0.6| = 0.1


class TestStaticCalibrationError:
    """Tests for SCE metric."""

    def test_sce_binary_classification(self):
        # Binary classification
        y_true = np.array([0, 0, 1, 1])
        y_pred_proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8]])
        sce = static_calibration_error(y_true, y_pred_proba, n_bins=10)
        assert isinstance(sce, float)
        assert sce >= 0.0

    def test_sce_multiclass(self):
        # 3-class classification
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred_proba = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
                [0.7, 0.2, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.7],
            ]
        )
        sce = static_calibration_error(y_true, y_pred_proba, n_bins=5)
        assert isinstance(sce, float)
        assert sce >= 0.0


class TestClassificationCalibrationScoreMetrics:
    """Tests for the combined calibration metrics function."""

    @pytest.fixture
    def binary_classification_data(self):
        np.random.seed(42)
        y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
        y_pred_proba = np.array(
            [
                [0.9, 0.1],
                [0.8, 0.2],
                [0.7, 0.3],
                [0.2, 0.8],
                [0.3, 0.7],
                [0.1, 0.9],
                [0.6, 0.4],
                [0.4, 0.6],
                [0.85, 0.15],
                [0.25, 0.75],
            ]
        )
        y_pred = np.argmax(y_pred_proba, axis=1)
        return y_true, y_pred_proba, y_pred

    def test_returns_all_metrics(self, binary_classification_data):
        y_true, y_pred_proba, y_pred = binary_classification_data
        results = classification_calibration_score_metrics(y_true, y_pred_proba, y_pred)
        assert len(results) == 3
        metric_names = [r["name"] for r in results]
        assert "SCE" in metric_names
        assert "ECE" in metric_names
        assert "MCE" in metric_names

    def test_metrics_are_floats(self, binary_classification_data):
        y_true, y_pred_proba, y_pred = binary_classification_data
        results = classification_calibration_score_metrics(y_true, y_pred_proba, y_pred)
        for result in results:
            assert isinstance(result["score"], float)

    def test_metrics_are_non_negative(self, binary_classification_data):
        y_true, y_pred_proba, y_pred = binary_classification_data
        results = classification_calibration_score_metrics(y_true, y_pred_proba, y_pred)
        for result in results:
            assert result["score"] >= 0.0

    def test_y_pred_optional(self, binary_classification_data):
        """y_pred should be computed from y_pred_proba if not provided."""
        y_true, y_pred_proba, _ = binary_classification_data
        results = classification_calibration_score_metrics(
            y_true, y_pred_proba, y_pred=None
        )
        assert len(results) == 3
