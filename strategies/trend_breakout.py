from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_percentile_rank(series: pd.Series, window: int = 252) -> pd.Series:
    min_periods = max(30, window // 3)
    return series.rolling(window=window, min_periods=min_periods).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )


def trend_breakout_v2_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Hardened trend breakout sleeve for V2.2 candidate testing.

    Adds:
      - volatility compression filter
      - breakout strength filter
      - optional retest entry mode
      - ATR trailing stop
      - time exit
      - volatility contraction exit
      - optional partial take-profit
      - min bars between trades
    """
    lookback = int(params.get("lookback", 20))
    atr_col = str(params.get("atr_col", "atr_14"))
    atr_norm_col = str(params.get("atr_norm_col", "atr_norm"))
    vol_rank_col = str(params.get("atr_pct_rank_col", "atr_norm_pct_rank"))

    vol_compression_max_pct = float(params.get("vol_compression_max_pct", 0.40))
    breakout_strength_atr_mult = float(params.get("breakout_strength_atr_mult", 0.25))

    retest_entry_mode = bool(params.get("retest_entry_mode", False))
    retest_expiry_bars = int(params.get("retest_expiry_bars", 10))
    retest_tolerance_atr_mult = float(params.get("retest_tolerance_atr_mult", 0.15))
    retest_confirm_buffer_atr_mult = float(params.get("retest_confirm_buffer_atr_mult", 0.05))

    trailing_stop_atr_mult = float(params.get("trailing_stop_atr_mult", 1.8))
    max_holding_bars = int(params.get("max_holding_bars", 72))
    vol_contraction_exit_mult = float(params.get("vol_contraction_exit_mult", 0.80))
    vol_contraction_window = int(params.get("vol_contraction_window", 20))

    partial_take_profit_rr = float(params.get("partial_take_profit_rr", 0.0))
    partial_take_profit_size = float(params.get("partial_take_profit_size", 0.5))
    partial_take_profit_size = float(np.clip(partial_take_profit_size, 0.1, 1.0))

    min_bars_between_trades = int(params.get("min_bars_between_trades", 6))

    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    atr = pd.to_numeric(df.get(atr_col, np.nan), errors="coerce")
    atr_norm = pd.to_numeric(df.get(atr_norm_col, np.nan), errors="coerce")

    if vol_rank_col in df.columns:
        atr_rank = pd.to_numeric(df[vol_rank_col], errors="coerce")
    else:
        atr_rank = _rolling_percentile_rank(atr_norm, window=252)

    donchian_upper = high.rolling(lookback, min_periods=lookback).max().shift(1)
    donchian_lower = low.rolling(lookback, min_periods=lookback).min().shift(1)
    atr_ma = atr_norm.rolling(vol_contraction_window, min_periods=max(10, vol_contraction_window // 2)).mean()

    signal = pd.Series(0.0, index=df.index, dtype=float)
    current_pos = 0.0
    entry_price = np.nan
    entry_atr = np.nan
    bars_in_trade = 0
    trail_stop = np.nan
    partial_tp_taken = False
    last_exit_i = -10_000_000

    pending_dir = 0
    pending_level = np.nan
    pending_expiry_i = -1

    for i in range(len(df)):
        px = float(close.iloc[i]) if pd.notna(close.iloc[i]) else np.nan
        atr_i = float(atr.iloc[i]) if pd.notna(atr.iloc[i]) else np.nan
        atrn_i = float(atr_norm.iloc[i]) if pd.notna(atr_norm.iloc[i]) else np.nan
        atr_rank_i = float(atr_rank.iloc[i]) if pd.notna(atr_rank.iloc[i]) else np.nan
        up_i = float(donchian_upper.iloc[i]) if pd.notna(donchian_upper.iloc[i]) else np.nan
        dn_i = float(donchian_lower.iloc[i]) if pd.notna(donchian_lower.iloc[i]) else np.nan
        atrn_ma_i = float(atr_ma.iloc[i]) if pd.notna(atr_ma.iloc[i]) else np.nan

        if not np.isfinite(px):
            signal.iloc[i] = current_pos
            continue

        can_open_new = (i - last_exit_i) >= min_bars_between_trades
        compression_ok = (not np.isfinite(atr_rank_i)) or (atr_rank_i <= vol_compression_max_pct)

        long_break = False
        short_break = False
        if np.isfinite(up_i) and np.isfinite(atr_i):
            long_break = px > up_i and (px - up_i) > (breakout_strength_atr_mult * atr_i)
        if np.isfinite(dn_i) and np.isfinite(atr_i):
            short_break = px < dn_i and (dn_i - px) > (breakout_strength_atr_mult * atr_i)

        long_entry = False
        short_entry = False

        if retest_entry_mode:
            # Setup pending retest after breakout impulse.
            if can_open_new and compression_ok and long_break:
                pending_dir = 1
                pending_level = up_i
                pending_expiry_i = i + retest_expiry_bars
            elif can_open_new and compression_ok and short_break:
                pending_dir = -1
                pending_level = dn_i
                pending_expiry_i = i + retest_expiry_bars

            if pending_dir != 0 and i <= pending_expiry_i and np.isfinite(atr_i):
                tol = retest_tolerance_atr_mult * atr_i
                confirm = retest_confirm_buffer_atr_mult * atr_i
                if pending_dir == 1:
                    touched = float(low.iloc[i]) <= (pending_level + tol)
                    confirmed = px >= (pending_level + confirm)
                    if touched and confirmed and can_open_new and compression_ok:
                        long_entry = True
                        pending_dir = 0
                else:
                    touched = float(high.iloc[i]) >= (pending_level - tol)
                    confirmed = px <= (pending_level - confirm)
                    if touched and confirmed and can_open_new and compression_ok:
                        short_entry = True
                        pending_dir = 0
            if pending_dir != 0 and i > pending_expiry_i:
                pending_dir = 0
        else:
            long_entry = can_open_new and compression_ok and long_break
            short_entry = can_open_new and compression_ok and short_break

        # Exit logic.
        exit_now = False
        if current_pos != 0.0:
            bars_in_trade += 1

            if np.isfinite(atr_i):
                if current_pos > 0:
                    new_stop = px - trailing_stop_atr_mult * atr_i
                    trail_stop = max(trail_stop, new_stop) if np.isfinite(trail_stop) else new_stop
                    if np.isfinite(trail_stop) and px < trail_stop:
                        exit_now = True
                else:
                    new_stop = px + trailing_stop_atr_mult * atr_i
                    trail_stop = min(trail_stop, new_stop) if np.isfinite(trail_stop) else new_stop
                    if np.isfinite(trail_stop) and px > trail_stop:
                        exit_now = True

            if bars_in_trade >= max_holding_bars:
                exit_now = True

            if np.isfinite(atrn_i) and np.isfinite(atrn_ma_i):
                if atrn_i <= (vol_contraction_exit_mult * atrn_ma_i):
                    exit_now = True

            if (
                partial_take_profit_rr > 0
                and not partial_tp_taken
                and np.isfinite(entry_price)
                and np.isfinite(entry_atr)
            ):
                if current_pos > 0 and (px - entry_price) >= (partial_take_profit_rr * entry_atr):
                    current_pos = partial_take_profit_size
                    partial_tp_taken = True
                elif current_pos < 0 and (entry_price - px) >= (partial_take_profit_rr * entry_atr):
                    current_pos = -partial_take_profit_size
                    partial_tp_taken = True

        if exit_now:
            current_pos = 0.0
            entry_price = np.nan
            entry_atr = np.nan
            bars_in_trade = 0
            trail_stop = np.nan
            partial_tp_taken = False
            last_exit_i = i

        # Entry / flip.
        if current_pos == 0.0:
            if long_entry and not short_entry:
                current_pos = 1.0
                entry_price = px
                entry_atr = atr_i
                bars_in_trade = 0
                trail_stop = px - trailing_stop_atr_mult * atr_i if np.isfinite(atr_i) else np.nan
                partial_tp_taken = False
            elif short_entry and not long_entry:
                current_pos = -1.0
                entry_price = px
                entry_atr = atr_i
                bars_in_trade = 0
                trail_stop = px + trailing_stop_atr_mult * atr_i if np.isfinite(atr_i) else np.nan
                partial_tp_taken = False
        elif current_pos > 0 and short_entry:
            current_pos = -1.0
            entry_price = px
            entry_atr = atr_i
            bars_in_trade = 0
            trail_stop = px + trailing_stop_atr_mult * atr_i if np.isfinite(atr_i) else np.nan
            partial_tp_taken = False
        elif current_pos < 0 and long_entry:
            current_pos = 1.0
            entry_price = px
            entry_atr = atr_i
            bars_in_trade = 0
            trail_stop = px - trailing_stop_atr_mult * atr_i if np.isfinite(atr_i) else np.nan
            partial_tp_taken = False

        signal.iloc[i] = current_pos

    warmup_invalid = donchian_upper.isna() | donchian_lower.isna()
    signal[warmup_invalid] = 0.0
    return signal.astype(float).clip(-1.0, 1.0)


def generate_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    return trend_breakout_v2_signals(df, params)
