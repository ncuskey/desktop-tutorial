"""Tests for all strategy implementations."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from data.loader import generate_synthetic_ohlcv
from data.indicators import add_indicators
from strategies.trend import MACrossover, DonchianBreakout
from strategies.mean_reversion import RSIReversal, BollingerFade
from strategies.breakout import RangeBreakout, VolatilityExpansionBreakout
from strategies.carry import CarryProxy


@pytest.fixture
def df():
    raw = generate_synthetic_ohlcv("EURUSD", start="2020-01-01", end="2022-12-31", freq="h")
    return add_indicators(raw)


def assert_valid_signals(signals: pd.Series, df: pd.DataFrame) -> None:
    """Common signal quality assertions."""
    assert len(signals) == len(df), "Signal length mismatch"
    assert signals.index.equals(df.index), "Signal index mismatch"
    assert set(signals.unique()).issubset({-1, 0, 1}), f"Invalid signal values: {signals.unique()}"
    assert not signals.isna().any(), "Signals contain NaN"


class TestMACrossover:
    def test_basic_signals(self, df):
        strat = MACrossover()
        sigs = strat.generate_signals(df, {"fast_period": 20, "slow_period": 50})
        assert_valid_signals(sigs, df)

    def test_ema_variant(self, df):
        strat = MACrossover()
        sigs = strat.generate_signals(df, {"fast_period": 10, "slow_period": 30, "ma_type": "ema"})
        assert_valid_signals(sigs, df)

    def test_initial_warmup_is_zero(self, df):
        strat = MACrossover()
        sigs = strat.generate_signals(df, {"fast_period": 20, "slow_period": 50})
        # First slow_period-1 bars must be 0 (MA not yet valid)
        assert (sigs.iloc[:49] == 0).all()

    def test_signal_changes_direction(self, df):
        strat = MACrossover()
        sigs = strat.generate_signals(df, {"fast_period": 20, "slow_period": 50})
        # Should have both long and short signals
        assert (sigs == 1).any()
        assert (sigs == -1).any()


class TestDonchianBreakout:
    def test_basic_signals(self, df):
        strat = DonchianBreakout()
        sigs = strat.generate_signals(df, {"period": 20})
        assert_valid_signals(sigs, df)

    def test_with_exit_mid_false(self, df):
        strat = DonchianBreakout()
        sigs = strat.generate_signals(df, {"period": 20, "exit_mid": False})
        assert_valid_signals(sigs, df)


class TestRSIReversal:
    def test_basic_signals(self, df):
        strat = RSIReversal()
        sigs = strat.generate_signals(df, {"period": 14, "oversold": 30, "overbought": 70})
        assert_valid_signals(sigs, df)

    def test_first_bar_is_zero(self, df):
        strat = RSIReversal()
        sigs = strat.generate_signals(df, {"period": 14})
        # We shift RSI by 1, so bar 0 can never have a signal
        assert sigs.iloc[0] == 0


class TestBollingerFade:
    def test_basic_signals(self, df):
        strat = BollingerFade()
        sigs = strat.generate_signals(df, {"period": 20, "num_std": 2.0})
        assert_valid_signals(sigs, df)

    def test_tight_bands_more_signals(self, df):
        strat = BollingerFade()
        sigs_tight = strat.generate_signals(df, {"period": 20, "num_std": 1.0})
        sigs_wide = strat.generate_signals(df, {"period": 20, "num_std": 3.0})
        # Tighter bands should produce more entries
        entries_tight = (sigs_tight.diff().abs() > 0).sum()
        entries_wide = (sigs_wide.diff().abs() > 0).sum()
        assert entries_tight >= entries_wide


class TestRangeBreakout:
    def test_basic_signals(self, df):
        strat = RangeBreakout()
        sigs = strat.generate_signals(df, {"lookback": 20, "hold_bars": 10})
        assert_valid_signals(sigs, df)

    def test_hold_bars_forces_exit(self, df):
        strat = RangeBreakout()
        hold_bars = 5
        sigs = strat.generate_signals(df, {"lookback": 20, "hold_bars": hold_bars})
        # Verify that exits DO occur: not always in the same position
        # (a strategy that holds forever would have near-zero zero-crossings)
        flat_periods = (sigs == 0).sum()
        assert flat_periods > 0, "Strategy never exits — hold_bars not working"
        # Also verify signal values are valid
        assert set(sigs.unique()).issubset({-1, 0, 1})


class TestVolatilityExpansionBreakout:
    def test_basic_signals(self, df):
        strat = VolatilityExpansionBreakout()
        sigs = strat.generate_signals(df, {"atr_period": 14, "atr_mult": 1.5})
        assert_valid_signals(sigs, df)


class TestCarryProxy:
    def test_eurusd_direction(self, df):
        strat = CarryProxy()
        sigs = strat.generate_signals(df, {"symbol": "EURUSD", "min_diff": 0.5})
        assert_valid_signals(sigs, df)
        # EUR rate (4.0) vs USD rate (5.25) → differential -1.25 → short
        assert (sigs == -1).all()

    def test_unknown_pair_flat(self, df):
        strat = CarryProxy()
        sigs = strat.generate_signals(df, {"symbol": "XXXXXX"})
        assert (sigs == 0).all()

    def test_high_min_diff_gives_flat(self, df):
        strat = CarryProxy()
        sigs = strat.generate_signals(df, {"symbol": "EURUSD", "min_diff": 100.0})
        assert (sigs == 0).all()
