"""
Visualization utilities for copper price forecasting.

Provides plotting functions for price charts, predictions vs actuals,
residual analysis, and feature importance visualization.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def plot_price_history(
    df: pd.DataFrame,
    price_column: str = "Price",
    title: str = "Copper Price History",
    figsize: tuple = (14, 6),
    save_path: Optional[str] = None,
) -> Any:
    """
    Plot historical copper price data.

    Args:
        df: DataFrame with DatetimeIndex and price column.
        price_column: Name of the price column.
        title: Plot title.
        figsize: Figure size as (width, height).
        save_path: If provided, save figure to this path.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        df.index, df[price_column], linewidth=1.2, color="#B87333", label="Copper Price"
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD/MT)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved price history plot to %s", save_path)

    return fig


def plot_predictions(
    actual: np.ndarray,
    predicted: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
    title: str = "Copper Price: Actual vs Predicted",
    figsize: tuple = (14, 6),
    save_path: Optional[str] = None,
) -> Any:
    """
    Plot actual vs predicted copper prices.

    Args:
        actual: Ground truth price values.
        predicted: Model predicted values.
        dates: Optional date index for the x-axis.
        title: Plot title.
        figsize: Figure size.
        save_path: If provided, save figure to this path.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    x = dates if dates is not None else np.arange(len(actual))

    ax.plot(x, actual, label="Actual", linewidth=1.5, color="#2196F3")
    ax.plot(
        x, predicted, label="Predicted", linewidth=1.5, color="#FF5722", linestyle="--"
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date" if dates is not None else "Time Step")
    ax.set_ylabel("Price (USD/MT)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved predictions plot to %s", save_path)

    return fig


def plot_residuals(
    actual: np.ndarray,
    predicted: np.ndarray,
    title: str = "Forecast Residuals",
    figsize: tuple = (14, 8),
    save_path: Optional[str] = None,
) -> Any:
    """
    Plot residual analysis: residuals over time and distribution.

    Args:
        actual: Ground truth values.
        predicted: Predicted values.
        title: Plot title.
        figsize: Figure size.
        save_path: If provided, save figure to this path.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    residuals = np.asarray(actual).flatten() - np.asarray(predicted).flatten()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Residual plot
    axes[0].plot(residuals, linewidth=0.8, color="#607D8B")
    axes[0].axhline(y=0, color="red", linestyle="--", alpha=0.7)
    axes[0].set_title("Residuals Over Time")
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Residual")
    axes[0].grid(True, alpha=0.3)

    # Histogram
    axes[1].hist(residuals, bins=30, color="#607D8B", edgecolor="white", alpha=0.8)
    axes[1].axvline(x=0, color="red", linestyle="--", alpha=0.7)
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved residuals plot to %s", save_path)

    return fig


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    top_n: int = 20,
    title: str = "Feature Importance",
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None,
) -> Any:
    """
    Plot feature importance as a horizontal bar chart.

    Args:
        feature_names: Names of the features.
        importances: Feature importance values.
        top_n: Number of top features to display.
        title: Plot title.
        figsize: Figure size.
        save_path: If provided, save figure to this path.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    indices = np.argsort(importances)[-top_n:]
    sorted_names = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(sorted_names, sorted_importances, color="#B87333", edgecolor="white")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved feature importance plot to %s", save_path)

    return fig


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str = "RMSE",
    title: str = "Model Comparison",
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
) -> Any:
    """
    Plot a comparison of multiple models on a given metric.

    Args:
        results: Dict mapping model names to their evaluation metrics dict.
        metric: Which metric to compare.
        title: Plot title.
        figsize: Figure size.
        save_path: If provided, save figure to this path.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    names = list(results.keys())
    values = [results[name].get(metric, 0) for name in names]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(names, values, color="#B87333", edgecolor="white")
    ax.set_title(f"{title} ({metric})", fontsize=14, fontweight="bold")
    ax.set_ylabel(metric)
    ax.grid(True, axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved model comparison plot to %s", save_path)

    return fig
