"""Mean-reversion strategies."""

from __future__ import annotations

import pandas as pd

from .base import Strategy


def _stateful_mean_reversion(
    long_entries: pd.Series,
    short_entries: pd.Series,
    long_exits: pd.Series,
    short_exits: pd.Series,
) -> pd.Series:
    state = 0.0
    positions: list[float] = []

    for idx in long_entries.index:
        if state == 0.0:
            if bool(long_entries.loc[idx]):
                state = 1.0
            elif bool(short_entries.loc[idx]):
                state = -1.0
        elif state > 0.0 and bool(long_exits.loc[idx]):
            state = 0.0
        elif state < 0.0 and bool(short_exits.loc[idx]):
            state = 0.0

        positions.append(state)

    return pd.Series(positions, index=long_entries.index, dtype=float)


class RSIReversalStrategy(Strategy):
    name = "rsi_reversal"

    def generate_signals(self, df: pd.DataFrame, params: dict[str, float]) -> pd.Series:
        window = int(params.get("window", 14))
        oversold = float(params.get("oversold", 30.0))
        overbought = float(params.get("overbought", 70.0))
        exit_level = float(params.get("exit_level", 50.0))

        rsi = df["rsi"] if "rsi" in df.columns else self._compute_fallback_rsi(df["close"], window)
        signal = _stateful_mean_reversion(
            long_entries=rsi < oversold,
            short_entries=rsi > overbought,
            long_exits=rsi >= exit_level,
            short_exits=rsi <= exit_level,
        )
        return self._coerce_signal(signal)

    @staticmethod
    def _compute_fallback_rsi(close: pd.Series, window: int) -> pd.Series:
        delta = close.diff()
        gains = delta.clip(lower=0.0)
        losses = -delta.clip(upper=0.0)
        avg_gain = gains.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
        avg_loss = losses.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0.0, pd.NA)
        return 100 - (100 / (1 + rs))


class BollingerFadeStrategy(Strategy):
    name = "bollinger_fade"

    def generate_signals(self, df: pd.DataFrame, params: dict[str, float]) -> pd.Series:
        window = int(params.get("window", 20))
        num_std = float(params.get("num_std", 2.0))
        rolling_mean = (
            df["bb_mid"] if "bb_mid" in df.columns else df["close"].rolling(window).mean()
        )
        rolling_std = df["close"].rolling(window).std(ddof=0)
        upper = df["bb_upper"] if "bb_upper" in df.columns else rolling_mean + num_std * rolling_std
        lower = df["bb_lower"] if "bb_lower" in df.columns else rolling_mean - num_std * rolling_std

        signal = _stateful_mean_reversion(
            long_entries=df["close"] < lower,
            short_entries=df["close"] > upper,
            long_exits=df["close"] >= rolling_mean,
            short_exits=df["close"] <= rolling_mean,
        )
        return self._coerce_signal(signal)
