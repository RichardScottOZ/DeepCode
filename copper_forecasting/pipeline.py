"""
End-to-end forecasting pipeline for copper prices.

Orchestrates data loading, preprocessing, feature engineering, model training,
evaluation, and visualization into a single configurable pipeline.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from copper_forecasting.data_loader import generate_sample_data, load_copper_csv
from copper_forecasting.evaluation import evaluate_all
from copper_forecasting.features import add_technical_indicators, create_lag_features
from copper_forecasting.models import (
    BaseForecastModel,
    GradientBoostingModel,
    LinearRegressionModel,
    RandomForestModel,
    RidgeRegressionModel,
)
from copper_forecasting.preprocessing import clean_data, split_data

logger = logging.getLogger(__name__)


class ForecastingPipeline:
    """
    End-to-end copper price forecasting pipeline.

    Orchestrates the full workflow from data loading through model evaluation,
    supporting multiple models and configurable feature engineering.

    Example:
        >>> pipeline = ForecastingPipeline()
        >>> results = pipeline.run()
        >>> print(results["best_model"])
    """

    def __init__(
        self,
        csv_path: Optional[str] = None,
        target_column: str = "Price",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        indicators: Optional[List[str]] = None,
        lags: Optional[List[int]] = None,
        models: Optional[List[BaseForecastModel]] = None,
    ):
        """
        Initialize the forecasting pipeline.

        Args:
            csv_path: Path to copper price CSV. If None, uses generated sample data.
            target_column: Name of the target price column.
            train_ratio: Fraction of data for training.
            val_ratio: Fraction of data for validation.
            test_ratio: Fraction of data for testing.
            indicators: Technical indicators to compute. None = all available.
            lags: Lag periods for feature engineering.
            models: List of model instances. If None, uses default ML models.
        """
        self.csv_path = csv_path
        self.target_column = target_column
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.indicators = indicators
        self.lags = lags
        self.models = models or self._default_models()
        self._results: Dict[str, Any] = {}

    @staticmethod
    def _default_models() -> List[BaseForecastModel]:
        """Return default set of ML models."""
        return [
            LinearRegressionModel(),
            RidgeRegressionModel(alpha=1.0),
            RandomForestModel(n_estimators=100, random_state=42),
            GradientBoostingModel(n_estimators=100, learning_rate=0.1, random_state=42),
        ]

    def load_data(self) -> pd.DataFrame:
        """Load copper price data from CSV or generate sample data."""
        if self.csv_path:
            df = load_copper_csv(self.csv_path)
        else:
            df = generate_sample_data(n_days=1000, seed=42)
        logger.info("Loaded data with shape %s", df.shape)
        return df

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering to the raw data."""
        df = clean_data(df)
        df = add_technical_indicators(
            df, price_column=self.target_column, include=self.indicators
        )
        df = create_lag_features(df, columns=[self.target_column], lags=self.lags)
        df = df.dropna()
        logger.info(
            "Feature engineering complete: %d rows, %d columns",
            len(df),
            len(df.columns),
        )
        return df

    def _prepare_xy(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Separate features (X) and target (y) from a DataFrame."""
        feature_cols = [c for c in df.columns if c != self.target_column]
        X = df[feature_cols].values
        y = df[self.target_column].values
        return X, y, feature_cols

    def run(self) -> Dict[str, Any]:
        """
        Execute the full forecasting pipeline.

        Returns:
            Dictionary containing:
                - 'model_results': Dict of model name -> evaluation metrics
                - 'best_model': Name of the best model by RMSE
                - 'best_metrics': Metrics dict for the best model
                - 'predictions': Dict of model name -> predicted values on test set
                - 'test_actual': Actual test set values
                - 'test_dates': Test set date index
                - 'feature_names': List of feature column names
        """
        # Load and prepare data
        raw_data = self.load_data()
        featured_data = self.prepare_features(raw_data)

        # Split data
        train_df, val_df, test_df = split_data(
            featured_data,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
        )

        X_train, y_train, feature_names = self._prepare_xy(train_df)
        X_val, y_val, _ = self._prepare_xy(val_df)
        X_test, y_test, _ = self._prepare_xy(test_df)

        # Train and evaluate models
        model_results: Dict[str, Dict[str, float]] = {}
        predictions: Dict[str, np.ndarray] = {}

        for model in self.models:
            try:
                logger.info("Training model: %s", model.name)
                model.fit(X_train, y_train)

                # Evaluate on test set
                y_pred = model.predict(X_test)
                metrics = evaluate_all(y_test, y_pred)
                model_results[model.name] = metrics
                predictions[model.name] = y_pred

                logger.info(
                    "Model %s - RMSE: %.4f, R²: %.4f",
                    model.name,
                    metrics["RMSE"],
                    metrics["R2"],
                )
            except Exception as e:
                logger.error("Failed to train model %s: %s", model.name, e)
                model_results[model.name] = {"error": str(e)}

        # Find best model by RMSE
        valid_results = {k: v for k, v in model_results.items() if "RMSE" in v}
        best_model = min(valid_results, key=lambda k: valid_results[k]["RMSE"])

        self._results = {
            "model_results": model_results,
            "best_model": best_model,
            "best_metrics": model_results[best_model],
            "predictions": predictions,
            "test_actual": y_test,
            "test_dates": test_df.index,
            "feature_names": feature_names,
        }

        logger.info("Pipeline complete. Best model: %s", best_model)
        return self._results

    @property
    def results(self) -> Dict[str, Any]:
        """Access the results from the last pipeline run."""
        return self._results
