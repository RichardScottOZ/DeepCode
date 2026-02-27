"""
Data preprocessing utilities for copper price forecasting.

Provides functions for cleaning, normalizing, and splitting time series data.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def clean_data(
    df: pd.DataFrame,
    price_column: str = "Price",
    fill_method: str = "ffill",
    drop_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Clean copper price data by handling missing values and outliers.

    Args:
        df: Input DataFrame with price data.
        price_column: Name of the main price column.
        fill_method: Method for filling missing values ('ffill', 'bfill', 'interpolate').
        drop_threshold: Drop columns with more than this fraction of missing values.

    Returns:
        Cleaned DataFrame.
    """
    df = df.copy()

    # Drop columns with too many missing values
    missing_fractions = df.isnull().mean()
    cols_to_drop = missing_fractions[missing_fractions > drop_threshold].index.tolist()
    if cols_to_drop:
        logger.info(
            "Dropping columns with >%.0f%% missing: %s",
            drop_threshold * 100,
            cols_to_drop,
        )
        df = df.drop(columns=cols_to_drop)

    # Fill missing values
    if fill_method == "interpolate":
        df = df.interpolate(method="time")
    elif fill_method == "ffill":
        df = df.ffill()
    elif fill_method == "bfill":
        df = df.bfill()

    # Fill any remaining NaN at the edges
    df = df.bfill().ffill()

    # Remove rows where price is non-positive
    if price_column in df.columns:
        invalid_mask = df[price_column] <= 0
        if invalid_mask.any():
            logger.warning(
                "Removing %d rows with non-positive prices", invalid_mask.sum()
            )
            df = df[~invalid_mask]

    logger.info("Cleaned data: %d rows, %d columns", len(df), len(df.columns))
    return df


def normalize(
    df: pd.DataFrame,
    columns: Optional[list] = None,
    method: str = "minmax",
) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """
    Normalize specified columns of a DataFrame.

    Args:
        df: Input DataFrame.
        columns: Columns to normalize. If None, normalizes all numeric columns.
        method: Normalization method ('minmax' or 'zscore').

    Returns:
        Tuple of (normalized DataFrame, parameters dict for denormalization).
    """
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    params: Dict[str, dict] = {}
    for col in columns:
        if col not in df.columns:
            continue
        if method == "minmax":
            col_min = df[col].min()
            col_max = df[col].max()
            denom = col_max - col_min
            if denom == 0:
                denom = 1.0
            df[col] = (df[col] - col_min) / denom
            params[col] = {"method": "minmax", "min": col_min, "max": col_max}
        elif method == "zscore":
            col_mean = df[col].mean()
            col_std = df[col].std()
            if col_std == 0:
                col_std = 1.0
            df[col] = (df[col] - col_mean) / col_std
            params[col] = {"method": "zscore", "mean": col_mean, "std": col_std}

    return df, params


def denormalize(
    values: np.ndarray,
    column_name: str,
    params: Dict[str, dict],
) -> np.ndarray:
    """
    Reverse normalization for a set of values.

    Args:
        values: Normalized values to denormalize.
        column_name: Name of the column these values correspond to.
        params: Parameters dict returned by normalize().

    Returns:
        Denormalized values as numpy array.

    Raises:
        KeyError: If column_name is not found in params.
    """
    if column_name not in params:
        raise KeyError(f"No normalization parameters found for column '{column_name}'")

    p = params[column_name]
    values = np.asarray(values, dtype=float)
    if p["method"] == "minmax":
        return values * (p["max"] - p["min"]) + p["min"]
    elif p["method"] == "zscore":
        return values * p["std"] + p["mean"]
    return values


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into train, validation, and test sets chronologically.

    Args:
        df: Input DataFrame (should be sorted by date).
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        test_ratio: Fraction of data for testing.

    Returns:
        Tuple of (train_df, val_df, test_df).

    Raises:
        ValueError: If ratios don't sum to approximately 1.0 or data is too small.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Ratios must sum to ~1.0, got {total:.3f}")

    n = len(df)
    if n < 3:
        raise ValueError(f"Need at least 3 data points, got {n}")

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    # Ensure each split has at least 1 row
    train_end = max(1, train_end)
    val_end = max(train_end + 1, val_end)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    logger.info(
        "Split data: train=%d, val=%d, test=%d",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    return train_df, val_df, test_df
