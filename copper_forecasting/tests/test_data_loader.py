"""Tests for the copper_forecasting data_loader module."""

import numpy as np
import pandas as pd
import pytest

from copper_forecasting.data_loader import generate_sample_data, load_copper_csv


class TestGenerateSampleData:
    """Tests for generate_sample_data."""

    def test_default_generates_dataframe(self):
        df = generate_sample_data(n_days=100, seed=42)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100

    def test_columns_present(self):
        df = generate_sample_data(n_days=50, seed=0)
        for col in ["Price", "Open", "High", "Low", "Volume"]:
            assert col in df.columns

    def test_index_is_datetime(self):
        df = generate_sample_data(n_days=50, seed=0)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == "Date"

    def test_prices_positive(self):
        df = generate_sample_data(n_days=500, seed=42)
        assert (df["Price"] > 0).all()

    def test_reproducibility(self):
        df1 = generate_sample_data(n_days=100, seed=123)
        df2 = generate_sample_data(n_days=100, seed=123)
        pd.testing.assert_frame_equal(df1, df2)

    def test_custom_base_price(self):
        df = generate_sample_data(n_days=10, base_price=8000.0, seed=0)
        # First price should be close to base_price
        assert abs(df["Price"].iloc[0] - 8000.0) < 1000.0

    def test_high_greater_equal_low(self):
        df = generate_sample_data(n_days=100, seed=42)
        assert (df["High"] >= df["Low"]).all()


class TestLoadCopperCsv:
    """Tests for load_copper_csv."""

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_copper_csv("/nonexistent/path.csv")

    def test_load_valid_csv(self, tmp_path):
        csv_file = tmp_path / "copper.csv"
        dates = pd.bdate_range("2023-01-01", periods=30)
        df = pd.DataFrame({"Date": dates, "Price": np.random.uniform(5000, 7000, 30)})
        df.to_csv(csv_file, index=False)

        result = load_copper_csv(str(csv_file))
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 30
        assert "Price" in result.columns

    def test_missing_date_column(self, tmp_path):
        csv_file = tmp_path / "bad.csv"
        pd.DataFrame({"Price": [100, 200]}).to_csv(csv_file, index=False)
        with pytest.raises(ValueError, match="Date column"):
            load_copper_csv(str(csv_file))

    def test_missing_price_column(self, tmp_path):
        csv_file = tmp_path / "bad.csv"
        pd.DataFrame({"Date": ["2023-01-01", "2023-01-02"]}).to_csv(
            csv_file, index=False
        )
        with pytest.raises(ValueError, match="Price column"):
            load_copper_csv(str(csv_file))

    def test_custom_column_names(self, tmp_path):
        csv_file = tmp_path / "copper.csv"
        df = pd.DataFrame(
            {"Timestamp": ["2023-01-01", "2023-01-02"], "Close": [6000.0, 6100.0]}
        )
        df.to_csv(csv_file, index=False)

        result = load_copper_csv(
            str(csv_file), date_column="Timestamp", price_column="Close"
        )
        assert "Price" in result.columns
        assert len(result) == 2
