"""Tests for the copper_forecasting features module."""

import pandas as pd
import pytest

from copper_forecasting.data_loader import generate_sample_data
from copper_forecasting.features import add_technical_indicators, create_lag_features


class TestAddTechnicalIndicators:
    """Tests for add_technical_indicators."""

    @pytest.fixture()
    def sample_data(self):
        return generate_sample_data(n_days=200, seed=42)

    def test_adds_sma_columns(self, sample_data):
        result = add_technical_indicators(sample_data, include=["sma"])
        for w in [5, 10, 20, 50]:
            assert f"SMA_{w}" in result.columns

    def test_adds_ema_columns(self, sample_data):
        result = add_technical_indicators(sample_data, include=["ema"])
        assert "EMA_12" in result.columns
        assert "EMA_26" in result.columns

    def test_adds_rsi(self, sample_data):
        result = add_technical_indicators(sample_data, include=["rsi"])
        assert "RSI_14" in result.columns
        # RSI should be between 0 and 100 (after warmup)
        rsi_valid = result["RSI_14"].dropna()
        assert (rsi_valid >= 0).all() and (rsi_valid <= 100).all()

    def test_adds_macd(self, sample_data):
        result = add_technical_indicators(sample_data, include=["macd"])
        assert "MACD" in result.columns
        assert "MACD_Signal" in result.columns
        assert "MACD_Hist" in result.columns

    def test_adds_bollinger(self, sample_data):
        result = add_technical_indicators(sample_data, include=["bollinger"])
        assert "BB_Upper" in result.columns
        assert "BB_Lower" in result.columns
        # Upper band should be above lower band
        valid = result.dropna(subset=["BB_Upper", "BB_Lower"])
        assert (valid["BB_Upper"] >= valid["BB_Lower"]).all()

    def test_adds_atr(self, sample_data):
        result = add_technical_indicators(sample_data, include=["atr"])
        assert "ATR_14" in result.columns
        # ATR should be non-negative
        atr_valid = result["ATR_14"].dropna()
        assert (atr_valid >= 0).all()

    def test_adds_roc(self, sample_data):
        result = add_technical_indicators(sample_data, include=["roc"])
        for p in [5, 10, 20]:
            assert f"ROC_{p}" in result.columns

    def test_adds_obv(self, sample_data):
        result = add_technical_indicators(sample_data, include=["obv"])
        assert "OBV" in result.columns

    def test_all_indicators(self, sample_data):
        result = add_technical_indicators(sample_data)
        # Should have more columns than original
        assert len(result.columns) > len(sample_data.columns)

    def test_does_not_modify_original(self, sample_data):
        original_cols = list(sample_data.columns)
        add_technical_indicators(sample_data)
        assert list(sample_data.columns) == original_cols


class TestCreateLagFeatures:
    """Tests for create_lag_features."""

    def test_creates_lag_columns(self):
        df = pd.DataFrame(
            {"Price": range(50)},
            index=pd.bdate_range("2023-01-01", periods=50),
        )
        result = create_lag_features(df, lags=[1, 2, 3])
        assert "Price_lag_1" in result.columns
        assert "Price_lag_2" in result.columns
        assert "Price_lag_3" in result.columns

    def test_lag_values_correct(self):
        df = pd.DataFrame({"Price": [10, 20, 30, 40, 50]})
        result = create_lag_features(df, lags=[1, 2], include_rolling=False)
        assert result["Price_lag_1"].iloc[1] == 10
        assert result["Price_lag_2"].iloc[2] == 10

    def test_creates_rolling_features(self):
        df = pd.DataFrame(
            {"Price": range(50)},
            index=pd.bdate_range("2023-01-01", periods=50),
        )
        result = create_lag_features(df, lags=[1], include_rolling=True)
        assert "Price_rolling_mean_5" in result.columns
        assert "Price_rolling_std_5" in result.columns

    def test_adds_calendar_features(self):
        df = pd.DataFrame(
            {"Price": range(50)},
            index=pd.bdate_range("2023-01-01", periods=50),
        )
        result = create_lag_features(df)
        assert "day_of_week" in result.columns
        assert "month" in result.columns

    def test_does_not_modify_original(self):
        df = pd.DataFrame({"Price": range(50)})
        original_cols = list(df.columns)
        create_lag_features(df)
        assert list(df.columns) == original_cols
