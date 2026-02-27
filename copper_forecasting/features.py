"""
Feature engineering for copper price forecasting.

Generates technical indicators, lag features, and rolling statistics
from historical copper price data.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_technical_indicators(
    df: pd.DataFrame,
    price_column: str = "Price",
    include: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Add technical indicators to copper price data.

    Available indicators: sma, ema, rsi, macd, bollinger, atr, roc, obv.

    Args:
        df: DataFrame with at least a price column.
        price_column: Name of the price column.
        include: List of indicators to add. If None, adds all available.

    Returns:
        DataFrame with new indicator columns appended.
    """
    df = df.copy()
    price = df[price_column]

    all_indicators = {"sma", "ema", "rsi", "macd", "bollinger", "atr", "roc", "obv"}
    selected = set(include) if include else all_indicators

    if "sma" in selected:
        for window in [5, 10, 20, 50]:
            df[f"SMA_{window}"] = price.rolling(window=window).mean()

    if "ema" in selected:
        for span in [12, 26]:
            df[f"EMA_{span}"] = price.ewm(span=span, adjust=False).mean()

    if "rsi" in selected:
        df["RSI_14"] = _compute_rsi(price, period=14)

    if "macd" in selected:
        ema_12 = price.ewm(span=12, adjust=False).mean()
        ema_26 = price.ewm(span=26, adjust=False).mean()
        df["MACD"] = ema_12 - ema_26
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    if "bollinger" in selected:
        sma_20 = price.rolling(window=20).mean()
        std_20 = price.rolling(window=20).std()
        df["BB_Upper"] = sma_20 + 2 * std_20
        df["BB_Middle"] = sma_20
        df["BB_Lower"] = sma_20 - 2 * std_20
        df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]

    if "atr" in selected and "High" in df.columns and "Low" in df.columns:
        high = df["High"]
        low = df["Low"]
        close_prev = price.shift(1)
        tr = pd.concat(
            [high - low, (high - close_prev).abs(), (low - close_prev).abs()],
            axis=1,
        ).max(axis=1)
        df["ATR_14"] = tr.rolling(window=14).mean()

    if "roc" in selected:
        for period in [5, 10, 20]:
            df[f"ROC_{period}"] = price.pct_change(periods=period) * 100

    if "obv" in selected and "Volume" in df.columns:
        df["OBV"] = _compute_obv(price, df["Volume"])

    logger.info("Added technical indicators: %s", sorted(selected))
    return df


def create_lag_features(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    lags: Optional[List[int]] = None,
    include_rolling: bool = True,
) -> pd.DataFrame:
    """
    Create lagged features and rolling statistics for time series forecasting.

    Args:
        df: Input DataFrame.
        columns: Columns to create lag features for. Defaults to ['Price'].
        lags: List of lag periods. Defaults to [1, 2, 3, 5, 7, 14].
        include_rolling: Whether to add rolling mean/std features.

    Returns:
        DataFrame with lag features appended.
    """
    df = df.copy()
    if columns is None:
        columns = ["Price"] if "Price" in df.columns else [df.columns[0]]
    if lags is None:
        lags = [1, 2, 3, 5, 7, 14]

    for col in columns:
        if col not in df.columns:
            logger.warning("Column '%s' not found, skipping lag features", col)
            continue
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

        if include_rolling:
            for window in [5, 10, 20]:
                df[f"{col}_rolling_mean_{window}"] = (
                    df[col].shift(1).rolling(window).mean()
                )
                df[f"{col}_rolling_std_{window}"] = (
                    df[col].shift(1).rolling(window).std()
                )

    # Add day-of-week and month features from index if it's a DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        df["day_of_week"] = df.index.dayofweek
        df["month"] = df.index.month

    logger.info("Created lag features for columns: %s with lags: %s", columns, lags)
    return df


def _compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _compute_obv(prices: pd.Series, volume: pd.Series) -> pd.Series:
    """Compute On-Balance Volume."""
    direction = np.sign(prices.diff())
    obv = (direction * volume).fillna(0).cumsum()
    return obv
