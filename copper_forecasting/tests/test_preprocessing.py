"""Tests for the copper_forecasting preprocessing module."""

import numpy as np
import pandas as pd
import pytest

from copper_forecasting.preprocessing import (
    clean_data,
    denormalize,
    normalize,
    split_data,
)


class TestCleanData:
    """Tests for clean_data."""

    def test_fills_missing_values(self):
        df = pd.DataFrame({"Price": [100.0, np.nan, 102.0, np.nan, 104.0]})
        result = clean_data(df)
        assert not result["Price"].isnull().any()

    def test_removes_non_positive_prices(self):
        df = pd.DataFrame({"Price": [100.0, -50.0, 200.0, 0.0, 300.0]})
        result = clean_data(df)
        assert (result["Price"] > 0).all()
        assert len(result) == 3

    def test_drops_columns_with_high_missing(self):
        df = pd.DataFrame(
            {
                "Price": [100.0, 200.0, 300.0, 400.0],
                "Sparse": [np.nan, np.nan, np.nan, 1.0],
            }
        )
        result = clean_data(df, drop_threshold=0.5)
        assert "Sparse" not in result.columns

    def test_interpolate_fill_method(self):
        index = pd.date_range("2023-01-01", periods=5)
        df = pd.DataFrame({"Price": [100.0, np.nan, 300.0, np.nan, 500.0]}, index=index)
        result = clean_data(df, fill_method="interpolate")
        assert not result["Price"].isnull().any()


class TestNormalize:
    """Tests for normalize and denormalize."""

    def test_minmax_range(self):
        df = pd.DataFrame({"Price": [100.0, 200.0, 300.0, 400.0, 500.0]})
        normalized, params = normalize(df, method="minmax")
        assert normalized["Price"].min() == pytest.approx(0.0)
        assert normalized["Price"].max() == pytest.approx(1.0)

    def test_zscore_stats(self):
        df = pd.DataFrame({"Price": [100.0, 200.0, 300.0, 400.0, 500.0]})
        normalized, params = normalize(df, method="zscore")
        assert normalized["Price"].mean() == pytest.approx(0.0, abs=1e-10)
        assert normalized["Price"].std() == pytest.approx(1.0, abs=0.1)

    def test_denormalize_minmax(self):
        original = np.array([100.0, 200.0, 300.0])
        df = pd.DataFrame({"Price": original})
        normalized, params = normalize(df, method="minmax")
        restored = denormalize(normalized["Price"].values, "Price", params)
        np.testing.assert_allclose(restored, original, atol=1e-10)

    def test_denormalize_zscore(self):
        original = np.array([100.0, 200.0, 300.0])
        df = pd.DataFrame({"Price": original})
        normalized, params = normalize(df, method="zscore")
        restored = denormalize(normalized["Price"].values, "Price", params)
        np.testing.assert_allclose(restored, original, atol=1e-10)

    def test_denormalize_missing_key(self):
        with pytest.raises(KeyError):
            denormalize(np.array([1.0]), "Unknown", {})

    def test_constant_column(self):
        df = pd.DataFrame({"Price": [5.0, 5.0, 5.0]})
        normalized, params = normalize(df, method="minmax")
        # Should not produce NaN
        assert not normalized["Price"].isnull().any()


class TestSplitData:
    """Tests for split_data."""

    def test_default_split_proportions(self):
        df = pd.DataFrame({"Price": range(100)})
        train, val, test = split_data(df)
        assert len(train) + len(val) + len(test) == 100

    def test_ratios_sum_check(self):
        df = pd.DataFrame({"Price": range(10)})
        with pytest.raises(ValueError, match="sum"):
            split_data(df, train_ratio=0.5, val_ratio=0.1, test_ratio=0.1)

    def test_too_few_rows(self):
        df = pd.DataFrame({"Price": [1, 2]})
        with pytest.raises(ValueError, match="at least 3"):
            split_data(df)

    def test_chronological_order(self):
        index = pd.date_range("2023-01-01", periods=100)
        df = pd.DataFrame({"Price": range(100)}, index=index)
        train, val, test = split_data(df)
        assert train.index[-1] < val.index[0]
        assert val.index[-1] < test.index[0]
