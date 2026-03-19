"""Regime-based orchestrator — classify market regimes and switch strategies."""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any
from enum import Enum

from ..strategies.base import Strategy


class Regime(Enum):
    TRENDING_HIGH_VOL = "trending_high_vol"
    TRENDING_LOW_VOL = "trending_low_vol"
    RANGING_HIGH_VOL = "ranging_high_vol"
    RANGING_LOW_VOL = "ranging_low_vol"


class RegimeOrchestrator:
    """Classify regimes using ADX (trend strength) and ATR (volatility),
    then route to the appropriate strategy.

    Regime classification:
      - ADX > adx_threshold and ATR_rank > vol_threshold → TRENDING_HIGH_VOL
      - ADX > adx_threshold and ATR_rank <= vol_threshold → TRENDING_LOW_VOL
      - ADX <= adx_threshold and ATR_rank > vol_threshold → RANGING_HIGH_VOL
      - ADX <= adx_threshold and ATR_rank <= vol_threshold → RANGING_LOW_VOL
    """

    def __init__(
        self,
        adx_threshold: float = 25.0,
        vol_threshold: float = 0.5,
        vol_lookback: int = 100,
    ):
        self.adx_threshold = adx_threshold
        self.vol_threshold = vol_threshold
        self.vol_lookback = vol_lookback
        self._regime_strategies: dict[Regime, tuple[Strategy, dict[str, Any]]] = {}

    def set_strategy(
        self,
        regime: Regime,
        strategy: Strategy,
        params: dict[str, Any] | None = None,
    ) -> "RegimeOrchestrator":
        self._regime_strategies[regime] = (strategy, params or strategy.default_params())
        return self

    def classify_regime(self, df: pd.DataFrame) -> pd.Series:
        """Return a Series of Regime enum values."""
        if "adx" not in df.columns or "atr" not in df.columns:
            raise ValueError("DataFrame must contain 'adx' and 'atr' columns.")

        atr_rank = df["atr"].rolling(self.vol_lookback, min_periods=20).rank(pct=True)
        is_trending = df["adx"] > self.adx_threshold
        is_high_vol = atr_rank > self.vol_threshold

        regime = pd.Series(Regime.RANGING_LOW_VOL, index=df.index)
        regime[is_trending & is_high_vol] = Regime.TRENDING_HIGH_VOL
        regime[is_trending & ~is_high_vol] = Regime.TRENDING_LOW_VOL
        regime[~is_trending & is_high_vol] = Regime.RANGING_HIGH_VOL
        regime[~is_trending & ~is_high_vol] = Regime.RANGING_LOW_VOL

        return regime

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate signals by routing each bar to the appropriate regime strategy."""
        regimes = self.classify_regime(df)

        precomputed = {}
        for regime, (strategy, params) in self._regime_strategies.items():
            precomputed[regime] = strategy.generate_signals(df, params)

        signal = pd.Series(0.0, index=df.index)
        for regime in Regime:
            if regime in precomputed:
                mask = regimes == regime
                signal[mask] = precomputed[regime][mask]

        return signal
