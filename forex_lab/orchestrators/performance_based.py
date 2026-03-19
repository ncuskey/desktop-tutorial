"""Performance-based orchestrator — allocate to best recent performers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..strategies.base import Strategy
from ..metrics.performance import ANNUALIZATION_FACTORS


class PerformanceBasedOrchestrator:
    """Select the strategy with the best rolling Sharpe ratio.

    At each bar the orchestrator looks back *lookback* bars, computes the
    Sharpe ratio of each strategy's returns, and follows the winner's signal.
    """

    def __init__(
        self,
        strategies: list[Strategy],
        params_list: list[dict[str, Any]],
        lookback: int = 252,
        freq: str = "h",
    ):
        self.strategies = strategies
        self.params_list = params_list
        self.lookback = lookback
        self.freq = freq

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        ann = ANNUALIZATION_FACTORS.get(self.freq, np.sqrt(252))
        signal_sets = {}
        return_sets = {}

        for strat, params in zip(self.strategies, self.params_list):
            sig = strat.generate_signals(df, params)
            signal_sets[strat.name] = sig

            price_ret = df["close"].pct_change().fillna(0)
            strat_ret = sig.shift(1).fillna(0) * price_ret
            return_sets[strat.name] = strat_ret

        ret_df = pd.DataFrame(return_sets)

        rolling_sharpe = (
            ret_df.rolling(self.lookback, min_periods=self.lookback).mean()
            / ret_df.rolling(self.lookback, min_periods=self.lookback).std()
        ) * ann

        best_strat = rolling_sharpe.idxmax(axis=1)

        final = pd.Series(0, index=df.index, dtype=int)
        for i in range(self.lookback, len(df)):
            winner = best_strat.iloc[i]
            if pd.notna(winner):
                final.iloc[i] = signal_sets[winner].iloc[i]

        return final
