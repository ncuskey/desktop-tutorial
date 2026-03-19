"""Rule-based: if ADX high use trend, if ADX low use mean reversion."""

from typing import Any, Optional

import pandas as pd

from forex_lab.strategies.base import BaseStrategy


class RuleBasedOrchestrator:
    """
    Switch strategy based on simple rules (e.g. ADX threshold).
    """

    def __init__(
        self,
        trend_strategy: BaseStrategy,
        mean_reversion_strategy: BaseStrategy,
        adx_threshold: float = 25.0,
        adx_column: str = "adx",
    ):
        self.trend_strategy = trend_strategy
        self.mean_reversion_strategy = mean_reversion_strategy
        self.adx_threshold = adx_threshold
        self.adx_column = adx_column

    def run(
        self,
        df: pd.DataFrame,
        trend_params: dict[str, Any],
        mean_reversion_params: dict[str, Any],
    ) -> pd.Series:
        """
        Use trend when ADX >= threshold, else mean reversion.
        """
        if self.adx_column not in df.columns:
            raise ValueError(f"DataFrame must have '{self.adx_column}' column")

        adx = df[self.adx_column].fillna(0)
        use_trend = adx >= self.adx_threshold

        trend_sig = self.trend_strategy.generate_signals(df, trend_params)
        mr_sig = self.mean_reversion_strategy.generate_signals(df, mean_reversion_params)

        position = trend_sig.where(use_trend, mr_sig)
        return position.fillna(0).astype(int)
