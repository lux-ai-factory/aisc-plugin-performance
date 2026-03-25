from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


def classification_calibration_score_metrics(
    y_true: "npt.NDArray[np.integer[Any]]",
    y_pred_proba: "npt.NDArray[np.floating[Any]]",
    y_pred: "npt.NDArray[np.integer[Any]] | None" = None,
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    """
    Computes ECE, MCE, and SCE.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred_proba : array-like
        Predicted probabilities.
    y_pred : array-like, optional
        Predicted class labels.
    n_bins : int, default=10
        Number of probability bins.

    Returns
    -------
    - "SCE": Static Calibration Error is the weighted average of the absolute
        confidence–accuracy gap across probability bins and classes
    - "ECE": Expected Calibration Error is the weighted average absolute difference
        between predicted confidence and empirical accuracy across probability bins.
    - "MCE": Maximum Calibration Error is the largest absolute difference across bins.
    """
    import numpy as np

    if y_pred is None:
        y_pred = np.argmax(y_pred_proba, axis=1)

    n_samples = y_true.shape[0]

    confidences = np.max(y_pred_proba, axis=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    accuracies = (y_pred == y_true).astype(np.float32)

    bin_counts, avg_conf, avg_acc, nonzero = _compute_bin_stats(
        confidences, accuracies, n_bins
    )

    gap = np.abs(avg_acc - avg_conf)

    return [
        dict(
            name="SCE",
            score=static_calibration_error(y_true, y_pred_proba, n_bins),
        ),
        dict(
            name="ECE",
            score=expected_calibration_error(
                bin_counts, avg_conf, avg_acc, nonzero, n_samples, gap
            ),
        ),
        dict(
            name="MCE",
            score=maximum_calibration_error(
                bin_counts, avg_conf, avg_acc, nonzero, gap
            ),
        ),
    ]


def _compute_bin_stats(
    confidences: "npt.NDArray[np.floating]",
    accuracies: "npt.NDArray[np.floating]",
    n_bins: int,
) -> tuple[
    "npt.NDArray[np.intp]",
    "npt.NDArray[np.floating]",
    "npt.NDArray[np.floating]",
    "npt.NDArray[np.bool_]",
]:
    """
    Returns per-bin counts, average confidence, average accuracy, and non-zero bins
    """
    import numpy as np

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(confidences, bin_edges, right=True) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    conf_sum = np.bincount(bin_indices, weights=confidences, minlength=n_bins)
    acc_sum = np.bincount(bin_indices, weights=accuracies, minlength=n_bins)

    nonzero = bin_counts > 0

    avg_conf = np.zeros(n_bins)
    avg_acc = np.zeros(n_bins)

    avg_conf[nonzero] = conf_sum[nonzero] / bin_counts[nonzero]
    avg_acc[nonzero] = acc_sum[nonzero] / bin_counts[nonzero]

    return bin_counts, avg_conf, avg_acc, nonzero


def maximum_calibration_error(
    bin_counts: "npt.NDArray[np.intp]",
    avg_conf: "npt.NDArray[np.floating]",
    avg_acc: "npt.NDArray[np.floating]",
    nonzero: "npt.NDArray[np.bool_]",
    gap: "npt.NDArray[np.floating] | None" = None,
) -> float:
    """
    Maximum Calibration Error (MCE)
    https://ojs.aaai.org/index.php/AAAI/article/view/9602
    """
    import numpy as np

    if gap is None:
        gap = np.abs(avg_acc - avg_conf)

    return float(np.max(gap[nonzero])) if np.any(nonzero) else 0.0


def expected_calibration_error(
    bin_counts: "npt.NDArray[np.intp]",
    avg_conf: "npt.NDArray[np.floating]",
    avg_acc: "npt.NDArray[np.floating]",
    nonzero: "npt.NDArray[np.bool_]",
    n_samples: int,
    gap: "npt.NDArray[np.floating] | None" = None,
) -> float:
    """
    Expected Calibration Error (ECE)
    https://ojs.aaai.org/index.php/AAAI/article/view/9602
    """
    import numpy as np

    if gap is None:
        gap = np.abs(avg_acc - avg_conf)

    return float(np.sum((bin_counts[nonzero] / n_samples) * gap[nonzero]))


def static_calibration_error(
    y_true: "npt.NDArray[np.integer[Any]]",
    y_pred_proba: "npt.NDArray[np.floating[Any]]",
    n_bins: int,
) -> float:
    """
    Static Calibration Error (SCE)
    https://arxiv.org/abs/1904.01685
    """
    import numpy as np

    n_samples, n_classes = y_pred_proba.shape
    sce = 0.0

    for c in range(n_classes):
        class_probs = y_pred_proba[:, c]
        class_acc = (y_true == c).astype(np.float32)

        bin_counts, avg_conf, avg_acc, nonzero = _compute_bin_stats(
            class_probs, class_acc, n_bins
        )

        gap = np.abs(avg_acc - avg_conf)
        sce += np.sum((bin_counts[nonzero] / n_samples) * gap[nonzero])

    return float(sce)
