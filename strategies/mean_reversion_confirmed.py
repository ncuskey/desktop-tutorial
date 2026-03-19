from __future__ import annotations

import numpy as np
import pandas as pd

from .filters import apply_filter


def mean_reversion_confirmed_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Mean-reversion sleeve with RSI asymmetry + Bollinger confirmation and configurable exits.
    """
    rsi_col = str(params.get("rsi_col", "rsi_14"))
    lower_col = str(params.get("lower_col", "bb_lower_20_2"))
    upper_col = str(params.get("upper_col", "bb_upper_20_2"))
    mean_col = str(params.get("mean_col", "bb_mid_20"))

    long_entry_rsi = float(params.get("long_entry_rsi", 30.0))
    short_entry_rsi = float(params.get("short_entry_rsi", 70.0))
    require_bb_confirmation = bool(params.get("require_bb_confirmation", True))
    bb_confirm_buffer = float(params.get("bb_confirm_buffer", 0.0))

    exit_mode = str(params.get("exit_mode", "mean_touch")).lower()  # mean_touch / fixed_horizon / time_stop
    fixed_horizon_bars = int(params.get("fixed_horizon_bars", 12))
    time_stop_bars = int(params.get("time_stop_bars", 24))
    neutral_rsi_exit = float(params.get("neutral_rsi_exit", 50.0))
    neutral_band = float(params.get("neutral_band", 2.0))

    close = pd.to_numeric(df["close"], errors="coerce")
    rsi = pd.to_numeric(df[rsi_col], errors="coerce")
    bb_lower = pd.to_numeric(df[lower_col], errors="coerce")
    bb_upper = pd.to_numeric(df[upper_col], errors="coerce")
    mean_line = pd.to_numeric(df[mean_col], errors="coerce")

    long_raw = rsi < long_entry_rsi
    short_raw = rsi > short_entry_rsi

    if require_bb_confirmation:
        long_entry = long_raw & (close <= (bb_lower * (1.0 + bb_confirm_buffer)))
        short_entry = short_raw & (close >= (bb_upper * (1.0 - bb_confirm_buffer)))
    else:
        long_entry = long_raw
        short_entry = short_raw

    signal = pd.Series(0.0, index=df.index, dtype=float)
    current = 0.0
    bars_in_trade = 0

    for i in range(len(df)):
        price = float(close.iloc[i]) if pd.notna(close.iloc[i]) else np.nan
        rsi_i = float(rsi.iloc[i]) if pd.notna(rsi.iloc[i]) else np.nan
        mean_i = float(mean_line.iloc[i]) if pd.notna(mean_line.iloc[i]) else np.nan

        if not np.isfinite(price):
            signal.iloc[i] = current
            continue

        go_long = bool(long_entry.iloc[i]) if pd.notna(long_entry.iloc[i]) else False
        go_short = bool(short_entry.iloc[i]) if pd.notna(short_entry.iloc[i]) else False
        exit_now = False

        if current != 0.0:
            bars_in_trade += 1
            if exit_mode == "mean_touch":
                if current > 0 and np.isfinite(mean_i) and price >= mean_i:
                    exit_now = True
                if current < 0 and np.isfinite(mean_i) and price <= mean_i:
                    exit_now = True
                if np.isfinite(rsi_i) and abs(rsi_i - neutral_rsi_exit) <= neutral_band:
                    exit_now = True
            elif exit_mode == "fixed_horizon":
                if bars_in_trade >= fixed_horizon_bars:
                    exit_now = True
            elif exit_mode == "time_stop":
                if bars_in_trade >= time_stop_bars:
                    exit_now = True
            else:
                raise ValueError(f"Unsupported exit_mode '{exit_mode}'")

        if exit_now:
            current = 0.0
            bars_in_trade = 0

        if current == 0.0:
            if go_long and not go_short:
                current = 1.0
                bars_in_trade = 0
            elif go_short and not go_long:
                current = -1.0
                bars_in_trade = 0
        elif current > 0 and go_short:
            current = -1.0
            bars_in_trade = 0
        elif current < 0 and go_long:
            current = 1.0
            bars_in_trade = 0

        signal.iloc[i] = current

    out = signal.astype(int)
    filter_condition = params.get("filter_condition")
    if filter_condition is not None:
        out = apply_filter(out, condition=filter_condition)
    return out


def generate_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    """Standard strategy interface adapter."""
    return mean_reversion_confirmed_signals(df, params)
