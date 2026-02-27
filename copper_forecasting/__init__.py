"""
Copper Price Forecasting Package

A sophisticated forecasting toolkit for copper commodity prices, providing
data loading, feature engineering, multiple forecasting models, evaluation
metrics, visualization, and an end-to-end pipeline.

Modules:
    data_loader: Load and generate copper price datasets
    preprocessing: Data cleaning, normalization, and splitting
    features: Technical indicator and feature engineering
    models: Forecasting model implementations
    evaluation: Model evaluation metrics
    visualization: Plotting and charting utilities
    pipeline: End-to-end forecasting pipeline
"""

from copper_forecasting.data_loader import load_copper_csv, generate_sample_data
from copper_forecasting.preprocessing import (
    clean_data,
    normalize,
    denormalize,
    split_data,
)
from copper_forecasting.features import add_technical_indicators, create_lag_features
from copper_forecasting.models import (
    LinearRegressionModel,
    RidgeRegressionModel,
    RandomForestModel,
    GradientBoostingModel,
    ExponentialSmoothingModel,
    ArimaModel,
    get_available_models,
)
from copper_forecasting.evaluation import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
    r_squared,
    directional_accuracy,
    evaluate_all,
)
from copper_forecasting.pipeline import ForecastingPipeline

__all__ = [
    # Data loading
    "load_copper_csv",
    "generate_sample_data",
    # Preprocessing
    "clean_data",
    "normalize",
    "denormalize",
    "split_data",
    # Features
    "add_technical_indicators",
    "create_lag_features",
    # Models
    "LinearRegressionModel",
    "RidgeRegressionModel",
    "RandomForestModel",
    "GradientBoostingModel",
    "ExponentialSmoothingModel",
    "ArimaModel",
    "get_available_models",
    # Evaluation
    "mean_absolute_error",
    "mean_squared_error",
    "root_mean_squared_error",
    "mean_absolute_percentage_error",
    "r_squared",
    "directional_accuracy",
    "evaluate_all",
    # Pipeline
    "ForecastingPipeline",
]
