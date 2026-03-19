"""Rule-based and adaptive orchestration logic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from forex_research_lab.combinations.engine import weighted_ensemble_signal


class RuleBasedOrchestrator:
    """Switch between trend and mean-reversion using ADX."""

    def __init__(self, *, adx_threshold: float = 25.0) -> None:
        self.adx_threshold = adx_threshold

    def combine(
        self,
        *,
        adx: pd.Series,
        trend_signal: pd.Series,
        mean_reversion_signal: pd.Series,
    ) -> pd.Series:
        return trend_signal.where(adx >= self.adx_threshold, mean_reversion_signal).fillna(0.0)


class PerformanceBasedOrchestrator:
    """Allocate to strategies with the best recent rolling Sharpe."""

    def __init__(self, *, lookback: int = 63) -> None:
        self.lookback = lookback

    def allocate(self, strategy_returns: pd.DataFrame) -> pd.DataFrame:
        rolling_mean = strategy_returns.rolling(self.lookback).mean()
        rolling_std = strategy_returns.rolling(self.lookback).std(ddof=0).replace(0.0, np.nan)
        rolling_sharpe = rolling_mean / rolling_std
        weights = rolling_sharpe.clip(lower=0.0)
        normalized = weights.div(weights.sum(axis=1), axis=0).fillna(0.0)
        return normalized

    def combine(self, signal_map: dict[str, pd.Series], strategy_returns: pd.DataFrame) -> pd.Series:
        weights = self.allocate(strategy_returns)
        combined = []
        for strategy_name, signal in signal_map.items():
            strategy_weight = weights.get(strategy_name, pd.Series(0.0, index=signal.index))
            combined.append(signal.fillna(0.0) * strategy_weight.reindex(signal.index).fillna(0.0))
        aggregate = sum(combined)
        result = pd.Series(0.0, index=aggregate.index)
        result = result.mask(aggregate > 0.05, 1.0)
        result = result.mask(aggregate < -0.05, -1.0)
        return result


class RegimeClassifierOrchestrator:
    """Classify regimes via ATR and ADX, then switch strategy sleeves."""

    def __init__(self, *, adx_threshold: float = 25.0, atr_quantile: float = 0.6) -> None:
        self.adx_threshold = adx_threshold
        self.atr_quantile = atr_quantile

    def classify(self, *, atr: pd.Series, adx: pd.Series) -> pd.Series:
        atr_threshold = atr.expanding().quantile(self.atr_quantile)
        labels = pd.Series("range", index=atr.index, dtype="object")
        labels = labels.mask((adx >= self.adx_threshold) & (atr >= atr_threshold), "trend")
        labels = labels.mask((adx < self.adx_threshold) & (atr >= atr_threshold), "volatile_mean_reversion")
        labels = labels.mask((adx >= self.adx_threshold) & (atr < atr_threshold), "steady_trend")
        return labels

    def combine(
        self,
        *,
        atr: pd.Series,
        adx: pd.Series,
        trend_signal: pd.Series,
        mean_reversion_signal: pd.Series,
        breakout_signal: pd.Series | None = None,
    ) -> pd.Series:
        regimes = self.classify(atr=atr, adx=adx)
        breakout_component = breakout_signal if breakout_signal is not None else trend_signal
        return pd.Series(
            np.select(
                [
                    regimes.eq("trend"),
                    regimes.eq("steady_trend"),
                    regimes.eq("volatile_mean_reversion"),
                ],
                [
                    breakout_component.reindex(regimes.index).fillna(0.0),
                    trend_signal.reindex(regimes.index).fillna(0.0),
                    mean_reversion_signal.reindex(regimes.index).fillna(0.0),
                ],
                default=weighted_ensemble_signal(
                    {
                        "trend": trend_signal.reindex(regimes.index).fillna(0.0),
                        "mean_reversion": mean_reversion_signal.reindex(regimes.index).fillna(0.0),
                    }
                ),
            ),
            index=regimes.index,
            dtype=float,
        )
