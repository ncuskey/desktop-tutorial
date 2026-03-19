"""Rule-based orchestrator — select strategies using indicator thresholds."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..strategies.base import Strategy


class RuleBasedOrchestrator:
    """Switch between trend and mean-reversion strategies based on ADX.

    When ADX > *adx_threshold* the market is deemed trending and the
    trend strategy is used; otherwise the mean-reversion strategy is used.
    """

    def __init__(
        self,
        trend_strategy: Strategy,
        trend_params: dict[str, Any],
        mr_strategy: Strategy,
        mr_params: dict[str, Any],
        adx_threshold: float = 25.0,
    ):
        self.trend_strategy = trend_strategy
        self.trend_params = trend_params
        self.mr_strategy = mr_strategy
        self.mr_params = mr_params
        self.adx_threshold = adx_threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if "adx" not in df.columns:
            raise ValueError("DataFrame must contain an 'adx' column — run add_indicators first")

        trend_signals = self.trend_strategy.generate_signals(df, self.trend_params)
        mr_signals = self.mr_strategy.generate_signals(df, self.mr_params)

        trending = df["adx"] > self.adx_threshold
        signal = pd.Series(0, index=df.index, dtype=int)
        signal[trending] = trend_signals[trending]
        signal[~trending] = mr_signals[~trending]
        return signal
