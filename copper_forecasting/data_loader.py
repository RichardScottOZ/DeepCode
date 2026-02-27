"""
Data loading utilities for copper price forecasting.

Provides functions to load historical copper price data from CSV files
and to generate synthetic sample data for testing and demonstration.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_copper_csv(
    filepath: str,
    date_column: str = "Date",
    price_column: str = "Price",
    parse_dates: bool = True,
) -> pd.DataFrame:
    """
    Load copper price data from a CSV file.

    Args:
        filepath: Path to the CSV file.
        date_column: Name of the column containing dates.
        price_column: Name of the column containing prices.
        parse_dates: Whether to parse the date column as datetime.

    Returns:
        A DataFrame with a DatetimeIndex and at least a 'Price' column.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns are missing.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    df = pd.read_csv(filepath)

    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found in CSV")
    if price_column not in df.columns:
        raise ValueError(f"Price column '{price_column}' not found in CSV")

    if parse_dates:
        df[date_column] = pd.to_datetime(df[date_column])

    df = df.set_index(date_column).sort_index()

    # Rename to standard column name if needed
    if price_column != "Price":
        df = df.rename(columns={price_column: "Price"})

    logger.info("Loaded %d rows from %s", len(df), filepath)
    return df


def generate_sample_data(
    n_days: int = 1000,
    start_date: str = "2020-01-01",
    base_price: float = 6000.0,
    volatility: float = 0.02,
    trend: float = 0.0001,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate synthetic copper price data using geometric Brownian motion.

    Args:
        n_days: Number of trading days to generate.
        start_date: Start date for the time series.
        base_price: Starting price in USD per metric ton.
        volatility: Daily volatility (standard deviation of returns).
        trend: Daily drift/trend component.
        seed: Random seed for reproducibility.

    Returns:
        A DataFrame with DatetimeIndex and columns:
        Price, Open, High, Low, Volume.
    """
    if seed is not None:
        np.random.seed(seed)

    dates = pd.bdate_range(start=start_date, periods=n_days)
    returns = np.random.normal(loc=trend, scale=volatility, size=n_days)
    prices = base_price * np.exp(np.cumsum(returns))

    # Generate OHLCV data
    daily_range = prices * np.abs(np.random.normal(0.005, 0.003, n_days))
    open_prices = prices + np.random.normal(0, daily_range * 0.3)
    high_prices = np.maximum(prices, open_prices) + daily_range * 0.5
    low_prices = np.minimum(prices, open_prices) - daily_range * 0.5
    volume = np.random.lognormal(mean=10, sigma=0.5, size=n_days).astype(int)

    df = pd.DataFrame(
        {
            "Price": prices,
            "Open": open_prices,
            "High": high_prices,
            "Low": low_prices,
            "Volume": volume,
        },
        index=dates,
    )
    df.index.name = "Date"

    logger.info("Generated %d days of synthetic copper price data", n_days)
    return df
