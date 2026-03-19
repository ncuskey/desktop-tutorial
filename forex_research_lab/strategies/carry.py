"""Carry-style placeholder strategy."""

from __future__ import annotations

import pandas as pd

from .base import Strategy


class CarryDifferentialStrategy(Strategy):
    name = "carry_differential_proxy"

    def generate_signals(self, df: pd.DataFrame, params: dict[str, float]) -> pd.Series:
        long_bias = float(params.get("long_bias", 0.0))

        if "carry_proxy" in df.columns:
            proxy = df["carry_proxy"]
        else:
            lookback = int(params.get("lookback", 20))
            proxy = df["close"].pct_change(lookback).rolling(lookback).mean()

        signal = pd.Series(0.0, index=df.index)
        signal = signal.mask(proxy > long_bias, 1.0)
        signal = signal.mask(proxy < -long_bias, -1.0)
        return self._coerce_signal(signal.ffill().fillna(0.0))
