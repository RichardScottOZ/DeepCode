"""Tests for the copper_forecasting evaluation module."""

import numpy as np
import pytest

from copper_forecasting.evaluation import (
    directional_accuracy,
    evaluate_all,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r_squared,
    root_mean_squared_error,
)


class TestMAE:
    def test_perfect_prediction(self):
        actual = np.array([1.0, 2.0, 3.0])
        assert mean_absolute_error(actual, actual) == 0.0

    def test_known_values(self):
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.5, 2.5, 3.5])
        assert mean_absolute_error(actual, predicted) == pytest.approx(0.5)


class TestMSE:
    def test_perfect_prediction(self):
        actual = np.array([1.0, 2.0, 3.0])
        assert mean_squared_error(actual, actual) == 0.0

    def test_known_values(self):
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([2.0, 3.0, 4.0])
        assert mean_squared_error(actual, predicted) == pytest.approx(1.0)


class TestRMSE:
    def test_perfect_prediction(self):
        actual = np.array([1.0, 2.0, 3.0])
        assert root_mean_squared_error(actual, actual) == 0.0

    def test_known_values(self):
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([2.0, 3.0, 4.0])
        assert root_mean_squared_error(actual, predicted) == pytest.approx(1.0)


class TestMAPE:
    def test_perfect_prediction(self):
        actual = np.array([1.0, 2.0, 3.0])
        assert mean_absolute_percentage_error(actual, actual) == 0.0

    def test_known_values(self):
        actual = np.array([100.0, 200.0])
        predicted = np.array([110.0, 220.0])
        assert mean_absolute_percentage_error(actual, predicted) == pytest.approx(10.0)

    def test_zero_actual(self):
        actual = np.array([0.0, 0.0])
        predicted = np.array([1.0, 2.0])
        result = mean_absolute_percentage_error(actual, predicted)
        assert result == float("inf")


class TestRSquared:
    def test_perfect_prediction(self):
        actual = np.array([1.0, 2.0, 3.0, 4.0])
        assert r_squared(actual, actual) == pytest.approx(1.0)

    def test_mean_prediction(self):
        actual = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = np.full(4, actual.mean())
        assert r_squared(actual, predicted) == pytest.approx(0.0)

    def test_constant_actual(self):
        actual = np.array([5.0, 5.0, 5.0])
        predicted = np.array([5.0, 5.0, 5.0])
        assert r_squared(actual, predicted) == 0.0


class TestDirectionalAccuracy:
    def test_perfect_direction(self):
        actual = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = np.array([1.1, 2.1, 3.1, 4.1])
        assert directional_accuracy(actual, predicted) == pytest.approx(1.0)

    def test_opposite_direction(self):
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([3.0, 2.0, 1.0])
        assert directional_accuracy(actual, predicted) == pytest.approx(0.0)

    def test_single_element(self):
        assert directional_accuracy(np.array([1.0]), np.array([2.0])) == 0.0


class TestEvaluateAll:
    def test_returns_all_metrics(self):
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = np.array([1.1, 2.2, 2.9, 4.1, 5.0])
        results = evaluate_all(actual, predicted)
        assert "MAE" in results
        assert "MSE" in results
        assert "RMSE" in results
        assert "MAPE" in results
        assert "R2" in results
        assert "DirectionalAccuracy" in results

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            evaluate_all(np.array([1.0, 2.0]), np.array([1.0]))

    def test_empty_arrays(self):
        with pytest.raises(ValueError, match="empty"):
            evaluate_all(np.array([]), np.array([]))
