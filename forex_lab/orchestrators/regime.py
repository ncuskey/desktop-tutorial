"""
Regime classifier and regime-based orchestrator.

Regime classification using ATR (volatility) and ADX (trend strength):

  ┌──────────────┬──────────────────────────────────────┐
  │  Regime      │  Conditions                          │
  ├──────────────┼──────────────────────────────────────┤
  │  TRENDING    │  ADX > 25 AND ATR% normal-to-high    │
  │  VOLATILE    │  ATR% very high (> 2× median)        │
  │  RANGING     │  ADX < 20 AND ATR% normal-to-low     │
  │  NEUTRAL     │  Otherwise                           │
  └──────────────┴──────────────────────────────────────┘
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from data.indicators import add_indicators


class Regime(str, Enum):
    TRENDING = "trending"
    VOLATILE = "volatile"
    RANGING = "ranging"
    NEUTRAL = "neutral"


class RegimeClassifier:
    """Classify market regime per bar using ATR and ADX.

    Parameters
    ----------
    adx_trend_threshold:
        ADX above which market is trending (default 25).
    adx_range_threshold:
        ADX below which market is ranging (default 20).
    atr_vol_multiplier:
        ATR% must exceed this multiple of the rolling median to be
        classified as VOLATILE (default 1.8).
    lookback:
        Rolling window for ATR median calculation (default 100).
    """

    def __init__(
        self,
        adx_trend_threshold: float = 25.0,
        adx_range_threshold: float = 20.0,
        atr_vol_multiplier: float = 1.8,
        lookback: int = 100,
    ) -> None:
        self.adx_trend = adx_trend_threshold
        self.adx_range = adx_range_threshold
        self.atr_vol_mult = atr_vol_multiplier
        self.lookback = lookback

    def classify(self, df: pd.DataFrame) -> pd.Series:
        """Return a Series of Regime values aligned to df.index."""
        df = add_indicators(df)
        adx = df.get("adx", pd.Series(np.nan, index=df.index))
        atr_pct = df.get("atr_pct", pd.Series(np.nan, index=df.index))

        atr_median = atr_pct.rolling(self.lookback, min_periods=10).median()
        is_volatile = atr_pct > (self.atr_vol_mult * atr_median)
        is_trending = adx > self.adx_trend
        is_ranging = adx < self.adx_range

        regime = pd.Series(Regime.NEUTRAL.value, index=df.index)
        regime[is_volatile] = Regime.VOLATILE.value
        regime[is_trending & ~is_volatile] = Regime.TRENDING.value
        regime[is_ranging & ~is_volatile] = Regime.RANGING.value

        return regime

    def regime_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return frequency table of regime occurrences."""
        regimes = self.classify(df)
        counts = regimes.value_counts()
        pcts = counts / len(regimes) * 100
        return pd.DataFrame({"count": counts, "pct": pcts.round(1)})


class RegimeOrchestrator:
    """Switch between strategies based on classified market regime.

    Parameters
    ----------
    regime_strategy_map:
        Dict mapping Regime value → (strategy, params) tuple.
    """

    name = "regime_orchestrator"

    def __init__(
        self,
        regime_strategy_map: dict[str, tuple[Any, dict[str, Any]]],
        classifier: RegimeClassifier | None = None,
    ) -> None:
        self.regime_map = regime_strategy_map
        self.classifier = classifier or RegimeClassifier()

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Switch strategies based on per-bar regime.

        Returns
        -------
        pd.Series of int8 {-1, 0, +1}.
        """
        df = add_indicators(df)
        regimes = self.classifier.classify(df)

        # Pre-compute signals for all strategies in the map
        strat_signals: dict[str, pd.Series] = {}
        for regime_val, (strat, params) in self.regime_map.items():
            strat_signals[regime_val] = strat.generate_signals(df, params)

        combined = pd.Series(0, index=df.index, dtype="int8")
        for i in range(len(df)):
            regime_val = regimes.iloc[i]
            if regime_val in strat_signals:
                combined.iloc[i] = int(strat_signals[regime_val].iloc[i])

        return combined
