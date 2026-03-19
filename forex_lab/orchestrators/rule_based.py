"""
Rule-based orchestrator.

Uses market-state indicators (ADX, ATR, RSI) to select the appropriate
strategy on each bar.

Default rules:
  ADX > 25  → trend strategy
  ADX < 20  → mean-reversion strategy
  20 ≤ ADX ≤ 25 → flat (neutral zone)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from data.indicators import add_indicators


@dataclass
class Rule:
    """A single dispatch rule."""
    condition: str       # 'adx_high' | 'adx_low' | 'high_vol' | 'low_vol' | 'always'
    strategy: Any        # strategy instance
    params: dict[str, Any]
    priority: int = 0    # higher priority wins when multiple rules match


class RuleBasedOrchestrator:
    """Dispatch to different strategies based on market state rules.

    Parameters
    ----------
    rules:
        List of Rule objects (evaluated in priority order).
    adx_high_threshold:
        ADX level above which markets are considered trending.
    adx_low_threshold:
        ADX level below which markets are considered ranging.
    atr_high_pct:
        ATR% above which volatility is considered high.
    """

    name = "rule_based_orchestrator"

    def __init__(
        self,
        rules: list[Rule],
        adx_high_threshold: float = 25.0,
        adx_low_threshold: float = 20.0,
        atr_high_pct: float = 0.006,
    ) -> None:
        self.rules = sorted(rules, key=lambda r: r.priority, reverse=True)
        self.adx_high = adx_high_threshold
        self.adx_low = adx_low_threshold
        self.atr_high_pct = atr_high_pct

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate orchestrated signals bar by bar.

        Returns
        -------
        pd.Series of int8 {-1, 0, +1}.
        """
        df = add_indicators(df)

        # Pre-compute all strategy signals
        strategy_signals: dict[int, pd.Series] = {}
        for rule in self.rules:
            if id(rule) not in strategy_signals:
                strategy_signals[id(rule)] = rule.strategy.generate_signals(df, rule.params)

        combined = pd.Series(0, index=df.index, dtype="int8")

        for i in range(len(df)):
            row = df.iloc[i]
            active_signal = 0

            for rule in self.rules:
                if self._evaluate_condition(rule.condition, row):
                    active_signal = int(strategy_signals[id(rule)].iloc[i])
                    break  # first matching rule (by priority) wins

            combined.iloc[i] = active_signal

        return combined

    def _evaluate_condition(self, condition: str, row: pd.Series) -> bool:
        adx_val = float(row.get("adx", 0) or 0)
        atr_pct = float(row.get("atr_pct", 0) or 0)

        if condition == "always":
            return True
        elif condition == "adx_high":
            return adx_val > self.adx_high
        elif condition == "adx_low":
            return adx_val < self.adx_low
        elif condition == "adx_mid":
            return self.adx_low <= adx_val <= self.adx_high
        elif condition == "high_vol":
            return atr_pct > self.atr_high_pct
        elif condition == "low_vol":
            return atr_pct <= self.atr_high_pct
        return False
