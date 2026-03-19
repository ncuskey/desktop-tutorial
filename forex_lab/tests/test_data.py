"""Tests for data loading, resampling, and indicators."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from data.loader import generate_synthetic_ohlcv
from data.resampler import resample_ohlcv
from data.indicators import (
    sma, ema, atr, rsi, bollinger_bands, adx, donchian_channels, add_indicators,
)
from data.costs import CostModel, DEFAULT_COSTS


@pytest.fixture
def eurusd_h1():
    return generate_synthetic_ohlcv("EURUSD", start="2020-01-01", end="2021-12-31", freq="h")


class TestDataLoader:
    def test_generates_correct_columns(self, eurusd_h1):
        assert set(eurusd_h1.columns) == {"open", "high", "low", "close", "volume"}

    def test_ohlc_integrity(self, eurusd_h1):
        assert (eurusd_h1["high"] >= eurusd_h1["close"]).all()
        assert (eurusd_h1["high"] >= eurusd_h1["open"]).all()
        assert (eurusd_h1["low"] <= eurusd_h1["close"]).all()
        assert (eurusd_h1["low"] <= eurusd_h1["open"]).all()

    def test_no_negative_prices(self, eurusd_h1):
        assert (eurusd_h1[["open", "high", "low", "close"]] > 0).all().all()

    def test_index_is_datetime(self, eurusd_h1):
        assert isinstance(eurusd_h1.index, pd.DatetimeIndex)

    def test_no_duplicate_index(self, eurusd_h1):
        assert not eurusd_h1.index.duplicated().any()

    def test_sorted_index(self, eurusd_h1):
        assert eurusd_h1.index.is_monotonic_increasing


class TestResampler:
    def test_h4_has_fewer_bars(self, eurusd_h1):
        h4 = resample_ohlcv(eurusd_h1, "H4")
        assert len(h4) < len(eurusd_h1)
        assert len(h4) == pytest.approx(len(eurusd_h1) / 4, abs=5)

    def test_daily_has_fewer_than_h4(self, eurusd_h1):
        h4 = resample_ohlcv(eurusd_h1, "H4")
        d1 = resample_ohlcv(eurusd_h1, "D1")
        assert len(d1) < len(h4)

    def test_ohlc_integrity_after_resample(self, eurusd_h1):
        h4 = resample_ohlcv(eurusd_h1, "H4")
        assert (h4["high"] >= h4["low"]).all()
        assert (h4["high"] >= h4["close"]).all()

    def test_no_nan_close(self, eurusd_h1):
        h4 = resample_ohlcv(eurusd_h1, "H4")
        assert not h4["close"].isna().any()


class TestIndicators:
    def test_sma_length(self, eurusd_h1):
        result = sma(eurusd_h1["close"], 20)
        assert len(result) == len(eurusd_h1)

    def test_sma_nan_warmup(self, eurusd_h1):
        result = sma(eurusd_h1["close"], 20)
        assert result.iloc[:19].isna().all()
        assert result.iloc[19:].notna().all()

    def test_ema_no_all_nan(self, eurusd_h1):
        result = ema(eurusd_h1["close"], 20)
        assert result.notna().sum() > 0

    def test_rsi_range(self, eurusd_h1):
        result = rsi(eurusd_h1["close"], 14)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_atr_positive(self, eurusd_h1):
        result = atr(eurusd_h1, 14)
        assert (result.dropna() > 0).all()

    def test_bollinger_upper_above_lower(self, eurusd_h1):
        upper, mid, lower = bollinger_bands(eurusd_h1["close"], 20, 2.0)
        valid = upper.notna()
        assert (upper[valid] > lower[valid]).all()
        assert (upper[valid] >= mid[valid]).all()
        assert (mid[valid] >= lower[valid]).all()

    def test_adx_range(self, eurusd_h1):
        result = adx(eurusd_h1, 14)
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_add_indicators_adds_columns(self, eurusd_h1):
        df = add_indicators(eurusd_h1)
        for col in ["sma_20", "rsi", "atr", "bb_upper", "bb_lower", "adx", "dc_upper", "macd"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_lookahead_sma(self, eurusd_h1):
        # SMA at index i should only depend on i and prior bars
        close = eurusd_h1["close"].copy()
        result1 = sma(close, 20)
        # Modify future bars — shouldn't affect past values
        close_modified = close.copy()
        close_modified.iloc[50:] *= 10
        result2 = sma(close_modified, 20)
        # First 50 values should be identical
        pd.testing.assert_series_equal(result1.iloc[:50], result2.iloc[:50])


class TestCostModel:
    def test_default_costs_exist(self):
        for symbol in ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]:
            assert symbol in DEFAULT_COSTS

    def test_spread_cost_positive(self):
        model = CostModel("EURUSD", spread_pips=1.0, pip_size=0.0001)
        assert model.spread_cost > 0

    def test_total_cost_positive(self):
        model = DEFAULT_COSTS["EURUSD"]
        assert model.total_round_trip_cost(1.10) > 0

    def test_cost_as_return_small(self):
        model = DEFAULT_COSTS["EURUSD"]
        # Cost should be a small fraction (< 0.1%)
        assert model.cost_as_return(1.10) < 0.001
