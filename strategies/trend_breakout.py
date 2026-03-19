from __future__ import annotations

import numpy as np
import pandas as pd

from .filters import apply_filter


def trend_breakout_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Trend breakout sleeve with optional Donchian + ATR breakout confirmation
    and optional ATR trailing stop.
    """
    lookback = int(params.get("donchian_lookback", params.get("lookback", 20)))
    atr_col = str(params.get("atr_col", "atr_14"))
    atr_breakout_mult = float(params.get("atr_breakout_mult", 1.2))
    combine_mode = str(params.get("combine_mode", "either")).lower()  # either / both
    use_donchian = bool(params.get("use_donchian", True))
    use_atr_breakout = bool(params.get("use_atr_breakout", True))
    trailing_stop_atr_mult = params.get("trailing_stop_atr_mult", None)
    if trailing_stop_atr_mult is not None:
        trailing_stop_atr_mult = float(trailing_stop_atr_mult)

    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    atr = pd.to_numeric(df[atr_col], errors="coerce") if atr_col in df.columns else pd.Series(np.nan, index=df.index)

    donchian_upper = high.rolling(lookback, min_periods=lookback).max().shift(1)
    donchian_lower = low.rolling(lookback, min_periods=lookback).min().shift(1)
    donchian_long = close > donchian_upper
    donchian_short = close < donchian_lower

    atr_anchor = close.shift(1)
    atr_long = close > (atr_anchor + (atr_breakout_mult * atr))
    atr_short = close < (atr_anchor - (atr_breakout_mult * atr))

    if use_donchian and use_atr_breakout:
        if combine_mode == "both":
            long_entry = donchian_long & atr_long
            short_entry = donchian_short & atr_short
        else:
            long_entry = donchian_long | atr_long
            short_entry = donchian_short | atr_short
    elif use_donchian:
        long_entry = donchian_long
        short_entry = donchian_short
    elif use_atr_breakout:
        long_entry = atr_long
        short_entry = atr_short
    else:
        raise ValueError("At least one breakout engine must be enabled.")

    signal = pd.Series(0.0, index=df.index, dtype=float)
    current = 0.0
    trailing_stop = np.nan

    for i in range(len(df)):
        price = float(close.iloc[i]) if pd.notna(close.iloc[i]) else np.nan
        atr_i = float(atr.iloc[i]) if pd.notna(atr.iloc[i]) else np.nan
        if not np.isfinite(price):
            signal.iloc[i] = current
            continue

        go_long = bool(long_entry.iloc[i]) if pd.notna(long_entry.iloc[i]) else False
        go_short = bool(short_entry.iloc[i]) if pd.notna(short_entry.iloc[i]) else False

        # Exit / stop logic first.
        if current > 0:
            if trailing_stop_atr_mult is not None and np.isfinite(atr_i):
                new_stop = price - trailing_stop_atr_mult * atr_i
                trailing_stop = max(trailing_stop, new_stop) if np.isfinite(trailing_stop) else new_stop
            stop_hit = np.isfinite(trailing_stop) and price < trailing_stop
            if stop_hit:
                current = 0.0
                trailing_stop = np.nan
        elif current < 0:
            if trailing_stop_atr_mult is not None and np.isfinite(atr_i):
                new_stop = price + trailing_stop_atr_mult * atr_i
                trailing_stop = min(trailing_stop, new_stop) if np.isfinite(trailing_stop) else new_stop
            stop_hit = np.isfinite(trailing_stop) and price > trailing_stop
            if stop_hit:
                current = 0.0
                trailing_stop = np.nan

        # Entry / flip logic.
        if current == 0.0:
            if go_long and not go_short:
                current = 1.0
                if trailing_stop_atr_mult is not None and np.isfinite(atr_i):
                    trailing_stop = price - trailing_stop_atr_mult * atr_i
            elif go_short and not go_long:
                current = -1.0
                if trailing_stop_atr_mult is not None and np.isfinite(atr_i):
                    trailing_stop = price + trailing_stop_atr_mult * atr_i
        elif current > 0 and go_short:
            current = -1.0
            trailing_stop = (
                price + trailing_stop_atr_mult * atr_i
                if trailing_stop_atr_mult is not None and np.isfinite(atr_i)
                else np.nan
            )
        elif current < 0 and go_long:
            current = 1.0
            trailing_stop = (
                price - trailing_stop_atr_mult * atr_i
                if trailing_stop_atr_mult is not None and np.isfinite(atr_i)
                else np.nan
            )

        signal.iloc[i] = current

    warmup_invalid = donchian_upper.isna() & donchian_lower.isna()
    signal[warmup_invalid] = 0.0
    out = signal.astype(int)
    filter_condition = params.get("filter_condition")
    if filter_condition is not None:
        out = apply_filter(out, condition=filter_condition)
    return out


def generate_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    """Standard strategy interface adapter."""
    return trend_breakout_signals(df, params)
