"""
Evaluation metrics for copper price forecasting models.

Provides standard regression metrics and time-series-specific metrics
for assessing forecast quality.
"""

import logging
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


def mean_absolute_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Compute Mean Absolute Error (MAE).

    Args:
        actual: Ground truth values.
        predicted: Predicted values.

    Returns:
        MAE value.
    """
    actual, predicted = _validate_inputs(actual, predicted)
    return float(np.mean(np.abs(actual - predicted)))


def mean_squared_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Compute Mean Squared Error (MSE).

    Args:
        actual: Ground truth values.
        predicted: Predicted values.

    Returns:
        MSE value.
    """
    actual, predicted = _validate_inputs(actual, predicted)
    return float(np.mean((actual - predicted) ** 2))


def root_mean_squared_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error (RMSE).

    Args:
        actual: Ground truth values.
        predicted: Predicted values.

    Returns:
        RMSE value.
    """
    return float(np.sqrt(mean_squared_error(actual, predicted)))


def mean_absolute_percentage_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Compute Mean Absolute Percentage Error (MAPE).

    Args:
        actual: Ground truth values (must be non-zero).
        predicted: Predicted values.

    Returns:
        MAPE value as a percentage.
    """
    actual, predicted = _validate_inputs(actual, predicted)
    # Avoid division by zero
    mask = actual != 0
    if not mask.any():
        logger.warning("All actual values are zero; MAPE is undefined")
        return float("inf")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Compute R-squared (coefficient of determination).

    Args:
        actual: Ground truth values.
        predicted: Predicted values.

    Returns:
        R² value (1.0 = perfect, 0.0 = mean baseline, negative = worse than mean).
    """
    actual, predicted = _validate_inputs(actual, predicted)
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def directional_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Compute Directional Accuracy (DA) - the fraction of times the predicted
    direction of change matches the actual direction.

    Args:
        actual: Ground truth values.
        predicted: Predicted values.

    Returns:
        Directional accuracy as a fraction [0.0, 1.0].
    """
    actual, predicted = _validate_inputs(actual, predicted)
    if len(actual) < 2:
        return 0.0

    actual_direction = np.sign(np.diff(actual))
    predicted_direction = np.sign(np.diff(predicted))
    matches = (actual_direction == predicted_direction).sum()
    return float(matches / len(actual_direction))


def evaluate_all(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute all available evaluation metrics.

    Args:
        actual: Ground truth values.
        predicted: Predicted values.

    Returns:
        Dictionary with all metric names and values.
    """
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "MSE": mean_squared_error(actual, predicted),
        "RMSE": root_mean_squared_error(actual, predicted),
        "MAPE": mean_absolute_percentage_error(actual, predicted),
        "R2": r_squared(actual, predicted),
        "DirectionalAccuracy": directional_accuracy(actual, predicted),
    }


def _validate_inputs(actual: np.ndarray, predicted: np.ndarray) -> tuple:
    """Validate and convert inputs to numpy arrays."""
    actual = np.asarray(actual, dtype=float).flatten()
    predicted = np.asarray(predicted, dtype=float).flatten()
    if actual.shape != predicted.shape:
        raise ValueError(
            f"Shape mismatch: actual {actual.shape} vs predicted {predicted.shape}"
        )
    if len(actual) == 0:
        raise ValueError("Input arrays must not be empty")
    return actual, predicted
