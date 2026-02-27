"""Tests for the copper_forecasting models module."""

import numpy as np
import pytest

from copper_forecasting.models import (
    ArimaModel,
    ExponentialSmoothingModel,
    GradientBoostingModel,
    LinearRegressionModel,
    RandomForestModel,
    RidgeRegressionModel,
    get_available_models,
)


@pytest.fixture()
def regression_data():
    """Simple regression dataset for testing."""
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = 3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] + np.random.normal(0, 0.1, 100)
    return X, y


@pytest.fixture()
def time_series_data():
    """Simple time series dataset for testing."""
    np.random.seed(42)
    n = 100
    y = np.cumsum(np.random.normal(0.01, 0.1, n)) + 100
    X = np.arange(n).reshape(-1, 1)
    return X, y


class TestLinearRegressionModel:
    def test_fit_predict(self, regression_data):
        X, y = regression_data
        model = LinearRegressionModel()
        model.fit(X[:80], y[:80])
        predictions = model.predict(X[80:])
        assert predictions.shape == (20,)
        assert model.is_fitted

    def test_predict_before_fit(self):
        model = LinearRegressionModel()
        with pytest.raises(RuntimeError, match="fitted"):
            model.predict(np.array([[1, 2, 3]]))


class TestRidgeRegressionModel:
    def test_fit_predict(self, regression_data):
        X, y = regression_data
        model = RidgeRegressionModel(alpha=0.5)
        model.fit(X[:80], y[:80])
        predictions = model.predict(X[80:])
        assert predictions.shape == (20,)

    def test_params(self):
        model = RidgeRegressionModel(alpha=2.0)
        params = model.get_params()
        assert params["alpha"] == 2.0


class TestRandomForestModel:
    def test_fit_predict(self, regression_data):
        X, y = regression_data
        model = RandomForestModel(n_estimators=10, random_state=42)
        model.fit(X[:80], y[:80])
        predictions = model.predict(X[80:])
        assert predictions.shape == (20,)

    def test_feature_importances(self, regression_data):
        X, y = regression_data
        model = RandomForestModel(n_estimators=10, random_state=42)
        model.fit(X, y)
        importances = model.feature_importances
        assert importances is not None
        assert len(importances) == 5


class TestGradientBoostingModel:
    def test_fit_predict(self, regression_data):
        X, y = regression_data
        model = GradientBoostingModel(n_estimators=10, random_state=42)
        model.fit(X[:80], y[:80])
        predictions = model.predict(X[80:])
        assert predictions.shape == (20,)

    def test_feature_importances(self, regression_data):
        X, y = regression_data
        model = GradientBoostingModel(n_estimators=10, random_state=42)
        model.fit(X, y)
        assert model.feature_importances is not None


class TestExponentialSmoothingModel:
    def test_fit_predict(self, time_series_data):
        X, y = time_series_data
        model = ExponentialSmoothingModel(trend="add")
        model.fit(X[:80], y[:80])
        predictions = model.predict(X[80:])
        assert predictions.shape == (20,)

    def test_predict_before_fit(self):
        model = ExponentialSmoothingModel()
        with pytest.raises(RuntimeError, match="fitted"):
            model.predict(np.array([[1]]))


class TestArimaModel:
    def test_fit_predict(self, time_series_data):
        X, y = time_series_data
        model = ArimaModel(order=(1, 1, 0))
        model.fit(X[:80], y[:80])
        predictions = model.predict(X[80:])
        assert predictions.shape == (20,)

    def test_predict_before_fit(self):
        model = ArimaModel()
        with pytest.raises(RuntimeError, match="fitted"):
            model.predict(np.array([[1]]))


class TestGetAvailableModels:
    def test_returns_list(self):
        models = get_available_models()
        assert isinstance(models, list)
        assert len(models) >= 6

    def test_model_dict_keys(self):
        models = get_available_models()
        for m in models:
            assert "name" in m
            assert "class" in m
            assert "default_params" in m
