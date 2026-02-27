"""
Forecasting models for copper price prediction.

Provides a unified interface for multiple forecasting approaches including
statistical models (ARIMA, Exponential Smoothing) and machine learning
models (Linear Regression, Ridge, Random Forest, Gradient Boosting).
"""

import abc
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseForecastModel(abc.ABC):
    """Abstract base class for all forecasting models."""

    def __init__(self, name: str, **kwargs: Any):
        self.name = name
        self.params = kwargs
        self._fitted = False

    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseForecastModel":
        """Fit the model to training data."""

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def get_params(self) -> Dict[str, Any]:
        return {"name": self.name, **self.params}


class LinearRegressionModel(BaseForecastModel):
    """Linear Regression forecasting model."""

    def __init__(self, **kwargs: Any):
        super().__init__(name="LinearRegression", **kwargs)
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionModel":
        from sklearn.linear_model import LinearRegression

        self._model = LinearRegression(**self.params)
        self._model.fit(X, y)
        self._fitted = True
        logger.info("Fitted LinearRegression model")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self._model.predict(X)


class RidgeRegressionModel(BaseForecastModel):
    """Ridge Regression forecasting model with L2 regularization."""

    def __init__(self, alpha: float = 1.0, **kwargs: Any):
        super().__init__(name="RidgeRegression", alpha=alpha, **kwargs)
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegressionModel":
        from sklearn.linear_model import Ridge

        self._model = Ridge(**self.params)
        self._model.fit(X, y)
        self._fitted = True
        logger.info("Fitted Ridge model (alpha=%.4f)", self.params.get("alpha", 1.0))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self._model.predict(X)


class RandomForestModel(BaseForecastModel):
    """Random Forest regression model for price forecasting."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        random_state: Optional[int] = 42,
        **kwargs: Any,
    ):
        super().__init__(
            name="RandomForest",
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs,
        )
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestModel":
        from sklearn.ensemble import RandomForestRegressor

        self._model = RandomForestRegressor(**self.params)
        self._model.fit(X, y)
        self._fitted = True
        logger.info(
            "Fitted RandomForest model (%d estimators)", self.params["n_estimators"]
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self._model.predict(X)

    @property
    def feature_importances(self) -> Optional[np.ndarray]:
        if self._fitted and self._model is not None:
            return self._model.feature_importances_
        return None


class GradientBoostingModel(BaseForecastModel):
    """Gradient Boosting regression model for price forecasting."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        random_state: Optional[int] = 42,
        **kwargs: Any,
    ):
        super().__init__(
            name="GradientBoosting",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs,
        )
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingModel":
        from sklearn.ensemble import GradientBoostingRegressor

        self._model = GradientBoostingRegressor(**self.params)
        self._model.fit(X, y)
        self._fitted = True
        logger.info(
            "Fitted GradientBoosting model (lr=%.4f)", self.params["learning_rate"]
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self._model.predict(X)

    @property
    def feature_importances(self) -> Optional[np.ndarray]:
        if self._fitted and self._model is not None:
            return self._model.feature_importances_
        return None


class ExponentialSmoothingModel(BaseForecastModel):
    """
    Exponential Smoothing model for time series forecasting.

    Uses Holt-Winters method with configurable trend and seasonal components.
    """

    def __init__(
        self,
        trend: Optional[str] = "add",
        seasonal: Optional[str] = None,
        seasonal_periods: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(
            name="ExponentialSmoothing",
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            **kwargs,
        )
        self._model = None
        self._result = None
        self._train_series = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ExponentialSmoothingModel":
        """
        Fit Exponential Smoothing model.

        Note: For this model, X is ignored; only the target series y is used.
        """
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        self._train_series = pd.Series(y.flatten())
        model_params = {
            k: v
            for k, v in self.params.items()
            if k in ("trend", "seasonal", "seasonal_periods")
        }
        self._model = ExponentialSmoothing(self._train_series, **model_params)
        self._result = self._model.fit()
        self._fitted = True
        logger.info("Fitted ExponentialSmoothing model")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Forecast the next n steps where n = len(X)."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")
        n_steps = X.shape[0]
        forecast = self._result.forecast(steps=n_steps)
        return np.asarray(forecast)


class ArimaModel(BaseForecastModel):
    """
    ARIMA / SARIMAX model for time series forecasting.

    Supports both non-seasonal ARIMA and seasonal SARIMAX configurations.
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        **kwargs: Any,
    ):
        super().__init__(
            name="ARIMA",
            order=order,
            seasonal_order=seasonal_order,
            **kwargs,
        )
        self._result = None
        self._train_series = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ArimaModel":
        """
        Fit ARIMA/SARIMAX model.

        Note: For this model, X is ignored; only the target series y is used.
        """
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        self._train_series = pd.Series(y.flatten())
        order = self.params["order"]
        seasonal_order = self.params.get("seasonal_order") or (0, 0, 0, 0)

        model = SARIMAX(
            self._train_series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._result = model.fit(disp=False)
        self._fitted = True
        logger.info("Fitted ARIMA%s model", order)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Forecast the next n steps where n = len(X)."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")
        n_steps = X.shape[0]
        forecast = self._result.forecast(steps=n_steps)
        return np.asarray(forecast)


def get_available_models() -> List[Dict[str, Any]]:
    """
    Return a list of all available forecasting models with default parameters.

    Returns:
        List of dicts with 'name', 'class', and 'default_params' keys.
    """
    return [
        {
            "name": "LinearRegression",
            "class": LinearRegressionModel,
            "default_params": {},
        },
        {
            "name": "RidgeRegression",
            "class": RidgeRegressionModel,
            "default_params": {"alpha": 1.0},
        },
        {
            "name": "RandomForest",
            "class": RandomForestModel,
            "default_params": {"n_estimators": 100, "random_state": 42},
        },
        {
            "name": "GradientBoosting",
            "class": GradientBoostingModel,
            "default_params": {"n_estimators": 100, "learning_rate": 0.1},
        },
        {
            "name": "ExponentialSmoothing",
            "class": ExponentialSmoothingModel,
            "default_params": {"trend": "add"},
        },
        {
            "name": "ARIMA",
            "class": ArimaModel,
            "default_params": {"order": (1, 1, 1)},
        },
    ]
