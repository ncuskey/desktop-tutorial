"""Rule-based orchestrator — switch strategies based on indicator thresholds."""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any

from ..strategies.base import Strategy


class RuleBasedOrchestrator:
    """Switch between strategies based on hard-coded indicator rules.

    Default logic:
      - ADX > high_adx_threshold → use trend strategy
      - ADX <= low_adx_threshold → use mean reversion strategy
      - otherwise → flat

    Custom rules can be provided via the `rules` parameter.
    """

    def __init__(
        self,
        trend_strategy: Strategy,
        mean_reversion_strategy: Strategy,
        trend_params: dict[str, Any] | None = None,
        mr_params: dict[str, Any] | None = None,
        high_adx: float = 30.0,
        low_adx: float = 20.0,
    ):
        self.trend_strategy = trend_strategy
        self.mr_strategy = mean_reversion_strategy
        self.trend_params = trend_params or trend_strategy.default_params()
        self.mr_params = mr_params or mean_reversion_strategy.default_params()
        self.high_adx = high_adx
        self.low_adx = low_adx

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate signals by switching strategies based on ADX level."""
        if "adx" not in df.columns:
            raise ValueError("DataFrame must contain 'adx' column. Run compute_indicators first.")

        trend_signals = self.trend_strategy.generate_signals(df, self.trend_params)
        mr_signals = self.mr_strategy.generate_signals(df, self.mr_params)

        signal = pd.Series(0.0, index=df.index)
        trend_regime = df["adx"] > self.high_adx
        mr_regime = df["adx"] <= self.low_adx

        signal[trend_regime] = trend_signals[trend_regime]
        signal[mr_regime] = mr_signals[mr_regime]

        return signal
