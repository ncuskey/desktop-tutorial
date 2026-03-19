"""Regime-based orchestrator — classify market regime and switch strategies."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..strategies.base import Strategy


class RegimeOrchestrator:
    """Simple volatility / trend-strength regime classifier.

    Regimes:
        trending — ADX above *adx_high* AND volatility (ATR percentile)
                   is not extreme.
        volatile — ATR percentile above *vol_high*.
        range    — everything else.

    Each regime maps to a strategy.
    """

    def __init__(
        self,
        regime_strategies: dict[str, tuple[Strategy, dict[str, Any]]],
        adx_high: float = 25.0,
        vol_high_pct: float = 75.0,
        atr_lookback: int = 100,
    ):
        self.regime_strategies = regime_strategies
        self.adx_high = adx_high
        self.vol_high_pct = vol_high_pct
        self.atr_lookback = atr_lookback

    def classify_regime(self, df: pd.DataFrame) -> pd.Series:
        """Return a regime label per bar."""
        if "adx" not in df.columns or "atr" not in df.columns:
            raise ValueError("DataFrame must contain 'adx' and 'atr' columns")

        atr_pct = df["atr"].rolling(self.atr_lookback, min_periods=self.atr_lookback).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

        regime = pd.Series("range", index=df.index)
        regime[(df["adx"] > self.adx_high) & (atr_pct <= self.vol_high_pct / 100)] = "trending"
        regime[atr_pct > self.vol_high_pct / 100] = "volatile"

        return regime

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        regime = self.classify_regime(df)

        all_signals = {}
        for regime_name, (strat, params) in self.regime_strategies.items():
            all_signals[regime_name] = strat.generate_signals(df, params)

        signal = pd.Series(0, index=df.index, dtype=int)
        for regime_name in self.regime_strategies:
            mask = regime == regime_name
            if mask.any() and regime_name in all_signals:
                signal[mask] = all_signals[regime_name][mask]

        return signal
